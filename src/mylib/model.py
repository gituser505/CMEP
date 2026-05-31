import numpy as np
import keras as k
from keras import ops, losses, metrics, layers, Model, Input, optimizers
from keras import saving
import tensorflow as tf

@saving.register_keras_serializable(package="mylib")
class Sampling(layers.Layer):
    def __init__(self, seed, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = k.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        z_mean, z_log_var = inputs
        if training is True:
            epsilon = k.random.normal(ops.shape(z_mean), seed=self.seed_generator)
            return z_mean + ops.exp(0.5 * z_log_var) * epsilon
        else:
            return z_mean

    def get_config(self):
        return {"seed": self.seed, **super().get_config()}


@saving.register_keras_serializable(package="mylib")
class PeriodicPadding2D(layers.Layer):
    def __init__(self, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.k = kernel_size
        self.s = strides

    def call(self, x):
        L = ops.shape(x)[1]
        out_dim = (L + self.s - 1) // self.s
        pad_total = ops.maximum(0, (out_dim - 1) * self.s + self.k - L)
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        top = x[:, L - pad_beg : L, :, :]
        bottom = x[:, 0 : pad_end, :, :]
        x = ops.concatenate([top, x, bottom], axis=1)

        left = x[:, :, L - pad_beg : L, :]
        right = x[:, :, 0 : pad_end, :]
        x = ops.concatenate([left, x, right], axis=2)
        return x

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.k, "strides": self.s}


@saving.register_keras_serializable(package="mylib")
class FiLMLayer(layers.Layer):
    def __init__(self, filters, hidden_units=32, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.hidden_units = hidden_units
        self.dense_hidden = layers.Dense(self.hidden_units, activation='relu')
        self.dense_gamma = layers.Dense(self.filters)
        self.dense_delta = layers.Dense(self.filters)
        self.reshape = layers.Reshape((1, 1, self.filters))
        self.multiply = layers.Multiply()
        self.add = layers.Add()

    def call(self, inputs):
        x_features, beta = inputs
        film_hidden = self.dense_hidden(beta)
        gamma = self.reshape(self.dense_gamma(film_hidden))
        delta = self.reshape(self.dense_delta(film_hidden))
        x_modulated = self.multiply([x_features, gamma])
        x_modulated = self.add([x_modulated, delta])
        return x_modulated

    def get_config(self):
        return {**super().get_config(), "filters": self.filters, "hidden_units": self.hidden_units}


@saving.register_keras_serializable(package="mylib")
class CVAE(Model):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams

        for key, value in hparams.items():
            setattr(self, key, value)
        
        self.enc_cnn_out_shape = None
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.loss_tracker = [
            metrics.Mean(name="total_loss"), 
            metrics.Mean(name="recon_loss"), 
            metrics.Mean(name="kl_loss")]
        
        if self.use_physics_loss:
            self.delta = k.Variable(0.0, trainable=False, dtype="float32")
            self.gamma = k.Variable(0.0, trainable=False, dtype="float32")
            
            self.loss_tracker.extend([
                metrics.Mean(name="m_loss"),
                metrics.Mean(name="e_loss")])
            
        if self.use_gumbel:
            self.tau = k.Variable(1.0, trainable=False, dtype="float32")


    def _build_encoder(self):
        spins_in = Input(shape=(self.L, self.L, 1), name='spins_in')
        beta = Input(shape=(1,), name='beta_condition')
        
        beta_reshape = layers.Reshape((1, 1, 1))(beta) 
        beta_spatial = beta_reshape * ops.ones_like(spins_in)
        x_cnn = layers.Concatenate(axis=-1)([spins_in, beta_spatial])

        for i, (f,k,s) in enumerate( zip(self.enc_filters, self.kernels, self.strides) ):
            #x_cnn = PeriodicPadding2D(k,1)(x_cnn)
            #x_cnn = layers.Conv2D(f,k, padding='valid', use_bias=False, name=f'enc_conv_{i}')(x_cnn)
            x_cnn = layers.Conv2D(f,k,s, padding='same', use_bias=False, name=f'enc_conv_{i}')(x_cnn)
            x_cnn = layers.BatchNormalization()(x_cnn)
            #x_cnn = layers.LayerNormalization()(x_cnn)
            #x_cnn = layers.AveragePooling2D(s)(x_cnn)
            x_cnn = layers.LeakyReLU(self.lrlu_slope)(x_cnn)

        self.enc_cnn_out_shape = x_cnn.shape[1:]
        x_latent = layers.Flatten()(x_cnn)

        for i,units in enumerate( self.mlp_units ):
            x_latent = layers.Dense(units, use_bias=False, name=f'enc_mlp_{i}')(x_latent)
            x_latent = layers.BatchNormalization()(x_latent)
            #x_latent = layers.LayerNormalization()(x_latent)
            x_latent = layers.LeakyReLU(self.lrlu_slope)(x_latent)

        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x_latent)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x_latent)
        z = Sampling(seed=self.seed, name='z')([z_mean, z_log_var])

        return Model([spins_in, beta], [z_mean, z_log_var, z], name='encoder')

    def _build_decoder(self):
        latent_space = Input(shape=(self.latent_dim,), name='latent_space')
        beta = Input(shape=(1,), name='beta_condition')

        x = layers.Concatenate()([latent_space, beta])
        cnn_units = int(np.prod(self.enc_cnn_out_shape))

        for i,units in enumerate( reversed(self.mlp_units) ):
            x = layers.Dense(units, use_bias=False, name=f'dec_mlp_{i}')(x)
            x = layers.BatchNormalization()(x)
            #x = layers.LayerNormalization()(x)
            x = layers.LeakyReLU(self.lrlu_slope)(x)
        x = layers.Dense(cnn_units, name='dec_mlp')(x)        
        x_cnn = layers.Reshape(self.enc_cnn_out_shape)(x)

        for i, (f,s,k) in enumerate( zip(self.dec_filters, reversed(self.strides), reversed(self.kernels)) ):
            #x_cnn = layers.UpSampling2D(s, interpolation="nearest")(x_cnn)
            #x_cnn = PeriodicPadding2D(k, 1)(x_cnn)
            #x_cnn = layers.Conv2D(f,k, padding='valid', use_bias=False, name=f'dec_conv_{i}')(x_cnn)
            x_cnn = layers.Conv2DTranspose(f,k,s, padding='same', use_bias=False, name=f'dec_conv_{i}')(x_cnn)
            x_cnn = layers.BatchNormalization()(x_cnn)
            #x_cnn = layers.LayerNormalization()(x_cnn)
            x_cnn = FiLMLayer(f)([x_cnn, beta])
            x_cnn = layers.LeakyReLU(self.lrlu_slope)(x_cnn)
        spins_out = layers.Conv2D(1, 3, padding='same', name='dec_out')(x_cnn)
        
        return Model([latent_space, beta], spins_out, name='decoder')

    def call(self, inputs, training=True):
        spins_in, beta = inputs
        z_mean, z_log_var, z = self.encoder([spins_in, beta], training=training)
        spins_out = self.decoder([z, beta], training=training)
        return [spins_out, z_mean, z_log_var]

    @tf.function(jit_compile=True)
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)
            losses_list = self.compute_losses(inputs, outputs)
            total_loss = losses_list[0]
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return self.metric_updates(losses_list)

    @tf.function(jit_compile=True)
    def test_step(self, inputs):
        outputs = self(inputs, training=False)
        losses_list = self.compute_losses(inputs, outputs)
        return self.metric_updates(losses_list)

    def compute_losses(self, inputs, outputs):
        spins_in, _ = inputs
        spins_out, z_mean, z_log_var = outputs
        
        recon_loss = ops.mean(ops.sum(losses.binary_crossentropy(spins_in, spins_out, from_logits=True), axis=[1,2]))
        kl_loss = -0.5 * ops.mean(ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1))       
        total_loss = recon_loss + self.alpha * kl_loss

        if self.use_physics_loss:
            if self.use_gumbel:
                uniform = k.random.uniform(ops.shape(spins_out), minval=1e-5, maxval=1.0 - 1e-5)
                logistic_noise = ops.log(uniform) - ops.log(1.0 - uniform)
                spins_new = ops.sigmoid((spins_out + logistic_noise) / self.tau)
            else:
                spins_new = ops.sigmoid(spins_out)

            m_loss = ops.abs(self.magnetization(spins_new) - self.magnetization(spins_in))
            e_loss = ops.abs(self.energy(spins_new) - self.energy(spins_in))

            total_loss += self.gamma * m_loss + self.delta * e_loss
            return [total_loss, recon_loss, kl_loss, m_loss, e_loss]

        return [total_loss, recon_loss, kl_loss]

    @tf.function(jit_compile=True)
    def generate(self, betas, stochastic=False):
        betas = ops.cast(ops.reshape(betas, [-1, 1]), "float32")
        num_samples = ops.shape(betas)[0]
        z = k.random.normal([num_samples, self.latent_dim])
        spins_logits = self.decoder([z, betas], training=False)
        spins_probs = ops.sigmoid(spins_logits)
        if stochastic is False:
            spins = ops.cast(spins_probs >= 0.5, "int8")
        else:
            uniform = k.random.uniform(ops.shape(spins_probs))
            spins = ops.cast(uniform < spins_probs, "int8")
        return spins

    def magnetization(self, spins):
        M = ops.mean(spins, axis=(1, 2))
        return ops.abs(2.0*M - 1.0)

    def energy(self, spins, J=1.0):
        s = 2 * spins - 1
        right = ops.roll(s, shift=-1, axis=2)
        down = ops.roll(s, shift=-1, axis=1)
        return -J * ops.mean(s * (right + down), axis=(1, 2))
    
    def wasserstein(self, input, output):
        input = ops.sort(input)
        output = ops.sort(output)
        return ops.mean(ops.abs(input - output))

    def mmd1(self, z, z_prior, sigmas=[1.0, 2.0, 5.0, 10.0]):
        z_sq = ops.sum(ops.square(z), axis=1, keepdims=True)
        prior_sq = ops.sum(ops.square(z_prior), axis=1, keepdims=True)
        
        dist_zz = z_sq - 2.0 * ops.matmul(z, ops.transpose(z)) + ops.transpose(z_sq)
        dist_pp = prior_sq - 2.0 * ops.matmul(z_prior, ops.transpose(z_prior)) + ops.transpose(prior_sq)
        dist_zp = z_sq - 2.0 * ops.matmul(z, ops.transpose(z_prior)) + ops.transpose(prior_sq)

        mmd_loss = 0.0
        for sigma in sigmas:
            gamma = 1.0 / (2.0 * ops.square(sigma))
            k_zz = ops.exp(-gamma * dist_zz)
            k_pp = ops.exp(-gamma * dist_pp)
            k_zp = ops.exp(-gamma * dist_zp)
            mmd_loss += ops.mean(k_zz) + ops.mean(k_pp) - 2.0 * ops.mean(k_zp)

        return mmd_loss

    def mmd2(self, x, y, sigmas=[0.05, 0.2, 1.0, 5.0]):
        x = ops.reshape(x, [-1, 1])
        y = ops.reshape(y, [-1, 1])

        xx_dist = ops.square(x - ops.transpose(x))
        yy_dist = ops.square(y - ops.transpose(y))
        xy_dist = ops.square(x - ops.transpose(y))
        
        mmd_loss = 0.0
        for sigma in sigmas:
            gamma = 1.0 / (2.0 * (sigma ** 2))
            k_xx = ops.exp(-gamma * xx_dist)
            k_yy = ops.exp(-gamma * yy_dist)
            k_xy = ops.exp(-gamma * xy_dist)
            mmd_loss += ops.mean(k_xx) + ops.mean(k_yy) - 2.0 * ops.mean(k_xy)
            
        return mmd_loss

    @property
    def metrics(self):
        return self.loss_tracker

    def metric_updates(self, losses_list):
        for t,l in zip(self.loss_tracker, losses_list): t.update_state(l)
        return {t.name: t.result() for t in self.loss_tracker}

    def get_config(self):
        return {"hparams": self.hparams, **super().get_config() }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    import os
    from types import SimpleNamespace 

    N = 2
    learning_rate = 0.001

    hparams = {
        "L": 8,
        "latent_dim": 2,
        "enc_filters": [4,8],
        "dec_filters": [4,2],
        "strides": [2,2],
        "kernels": [2,2],
        "mlp_units": [8,4],
        "alpha": 0.1 }
    hp = SimpleNamespace(**hparams)

    if os.path.exists("test_cvae.keras"):
        os.remove("test_cvae.keras")

    cvae = CVAE(hparams)
    cvae.compile(optimizer=optimizers.Adam(learning_rate), jit_compile=True)
    print("Model Instantiation: PASSED")

    spins = np.random.uniform(0, 1, (N, hp.L, hp.L, 1)).astype(np.float32)
    betas = np.random.uniform(0, 1, (N, 1),).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((spins, betas)).batch(N)
    cvae.fit(dataset, epochs=1)

    output_orig = cvae([spins, betas], training=False)
    cvae.save("test_cvae.keras")
    print("Model saved")

    loaded_cvae = saving.load_model("test_cvae.keras")
    output_loaded = loaded_cvae([spins, betas], training=False)
    
    # Iterate through output list to compare tensors
    for o1, o2 in zip(output_orig, output_loaded):
        assert np.allclose(o1, o2, atol=1e-5), "Weights did not load correctly"
        
    os.remove("test_cvae.keras")
    print("Save/Load Cycle: PASSED")
    
