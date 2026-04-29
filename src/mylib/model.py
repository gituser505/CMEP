import numpy as np
import tensorflow as tf
import keras as k
from keras import ops, losses, metrics, layers, Model, Input, optimizers
from keras import saving

from mylib.layers import Sampling, FiLMLayer, PeriodicPadding2D


@saving.register_keras_serializable(package="mylib")
class CVAE(Model):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        self.tau = k.Variable(1.0, trainable=False, dtype="float32", name="softmax_temp")
        
        for key, value in hparams.items():
            setattr(self, key, value)
        
        self.enc_cnn_out_shape = None
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.loss_tracker = [
            metrics.Mean(name="total_loss"), 
            metrics.Mean(name="recon_loss"), 
            metrics.Mean(name="kl_loss"), 
            metrics.Mean(name="m_loss"),
            metrics.Mean(name="e_loss"),
            metrics.Mean(name="unweighted_loss")]

    def _build_encoder(self):
        spins_in = Input(shape=(self.L, self.L, 1), name='spins_in')
        beta = Input(shape=(1,), name='beta_condition')
        
        beta_reshape = layers.Reshape((1, 1, 1))(beta) 
        beta_spatial = beta_reshape * ops.ones_like(spins_in)
        x_cnn = layers.Concatenate(axis=-1)([spins_in, beta_spatial])

        for i, (f,k,s) in enumerate( zip(self.enc_filters, self.kernels, self.strides) ):
            #x_cnn = PeriodicPadding2D(k,s)(x_cnn)
            x_cnn = layers.Conv2D(f,k,s, padding='same', use_bias=False, name=f'enc_conv_{i}')(x_cnn)
            x_cnn = layers.BatchNormalization()(x_cnn)
            x_cnn = layers.LeakyReLU(self.lrlu_slope)(x_cnn)
            #x_cnn = layers.MaxPooling2D(pool_size=s, strides=s, padding='same', name=f'enc_pool_{i}')(x_cnn)

        self.enc_cnn_out_shape = x_cnn.shape[1:]
        x_latent = layers.Flatten()(x_cnn)

        for i,units in enumerate( self.mlp_units ):
            x_latent = layers.Dense(units, use_bias=False, name=f'enc_mlp_{i}')(x_latent)
            x_latent = layers.BatchNormalization()(x_latent)
            x_latent = layers.LeakyReLU(self.lrlu_slope)(x_latent)

        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x_latent)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x_latent)
        z = Sampling(name='z')([z_mean, z_log_var])

        return Model([spins_in, beta], [z_mean, z_log_var, z], name='encoder')

    def _build_decoder(self):
        latent_space = Input(shape=(self.latent_dim,), name='latent_space')
        beta = Input(shape=(1,), name='beta_condition')

        x = layers.Concatenate()([latent_space, beta])
        cnn_units = int(np.prod(self.enc_cnn_out_shape))

        for i,units in enumerate( reversed(self.mlp_units) ):
            x = layers.Dense(units, use_bias=False, name=f'dec_mlp_{i}')(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(self.lrlu_slope)(x)
        x = layers.Dense(cnn_units, name='dec_mlp')(x)        
        x_cnn = layers.Reshape(self.enc_cnn_out_shape)(x)

        for i, (f,s,k) in enumerate( zip(self.dec_filters, reversed(self.strides), reversed(self.kernels)) ):
            #x_cnn = layers.UpSampling2D(s, interpolation="nearest")(x_cnn)
            #x_cnn = PeriodicPadding2D(k,s)(x_cnn)
            x_cnn = layers.Conv2DTranspose(f,k,s, padding='same', use_bias=False, name=f'dec_conv_{i}')(x_cnn)
            #x_cnn = layers.Conv2D(f, k, padding='valid', use_bias=False, name=f'dec_conv_{i}')(x_cnn)
            x_cnn = layers.BatchNormalization()(x_cnn)
            x_cnn = FiLMLayer(f)([x_cnn, beta])
            x_cnn = layers.LeakyReLU(self.lrlu_slope)(x_cnn)
        spins_out = layers.Conv2D(1, 1, name='dec_out')(x_cnn)
        
        return Model([latent_space, beta], spins_out, name='decoder')

    def call(self, inputs, training=False):
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
        
        spins_soft = ops.sigmoid(spins_out)
        spins_hard = ops.cast(spins_soft >= 0.5, "float32") 
        spins_ste = spins_soft + ops.stop_gradient(spins_hard - spins_soft)

        #uniform = k.random.uniform(ops.shape(spins_out), minval=1e-5, maxval=1.0 - 1e-5)
        #logistic_noise = ops.log(uniform) - ops.log(1.0 - uniform)
        #spins_discrete = ops.sigmoid((spins_out + logistic_noise) / self.tau)

        recon_loss = ops.mean(ops.sum(losses.binary_crossentropy(spins_in, spins_out, from_logits=True), axis=[1,2]))
        kl_loss = -0.5 * ops.mean(ops.sum(1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var), axis=1))       
        m_loss = ops.mean(ops.abs(self.magnetization(spins_ste) - self.magnetization(spins_in)))
        e_loss = ops.mean(ops.abs(self.energy(spins_ste) - self.energy(spins_in)))
        total_loss = recon_loss + self.alpha * kl_loss + self.gamma * m_loss + self.delta * e_loss
        unweighted_loss = recon_loss + kl_loss + m_loss + e_loss
        return [total_loss, recon_loss, kl_loss, m_loss, e_loss, unweighted_loss]

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
        M = ops.mean(spins, axis=[1, 2]) 
        return 2*M - 1

    def energy(self, spins, J=1.0):
        s = 2.0 * spins - 1.0  
        right = ops.roll(s, shift=-1, axis=2)
        down = ops.roll(s, shift=-1, axis=1)
        return -J * ops.mean(s * (right + down), axis=[1, 2])

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
    
