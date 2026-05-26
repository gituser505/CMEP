import keras as k
from keras import ops, layers
from keras import saving

@saving.register_keras_serializable(package="projectlib")
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
