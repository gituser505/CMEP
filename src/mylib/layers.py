import keras as k
from keras import ops, layers
from keras import saving


@saving.register_keras_serializable(package="mylib")
class Sampling(layers.Layer):
    def __init__(self, seed=42, **kwargs):
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


@saving.register_keras_serializable(package="projectlib")
class PeriodicPadding2D(layers.Layer):
    def __init__(self, kernel_size, strides=0, **kwargs):
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
