import keras as k
from keras import ops, layers
from keras import saving


@saving.register_keras_serializable(package="mylib")
class Sampling(layers.Layer):
    """Performs the reparameterization trick for Variational Autoencoders.

    This layer takes the mean and log-variance from the encoder and samples 
    a latent vector. During training, it adds Gaussian noise to allow for 
    backpropagation through the random sampling process using the formula:
    z = z_mean + exp(0.5 * z_log_var) * epsilon. 
    During inference, it deterministically returns the mean.

    Args:
        seed (int): Random seed for the noise generator to ensure reproducibility.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """
    def __init__(self, seed, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = k.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        """Samples the latent vector z.

        Args:
            inputs (tuple): A tuple containing two Tensors:
                - z_mean (Tensor): The mean of the latent distribution.
                - z_log_var (Tensor): The log-variance of the latent distribution.
            training (bool, optional): Whether the layer is in training or inference mode. 
                Defaults to False.

        Returns:
            Tensor: The sampled latent vector if training is True, otherwise returns `z_mean`.
        """
        z_mean, z_log_var = inputs
        if training is True:
            epsilon = k.random.normal(ops.shape(z_mean), seed=self.seed_generator)
            return z_mean + ops.exp(0.5 * z_log_var) * epsilon
        else:
            return z_mean

    def get_config(self):
        """Returns the configuration of the layer for serialization.

        Returns:
            dict: A dictionary containing the layer's configuration.
        """
        return {"seed": self.seed, **super().get_config()}


@saving.register_keras_serializable(package="mylib")
class PeriodicPadding2D(layers.Layer):
    """Applies periodic padding to a 2D input tensor.

    This layer periodically padds the input tensor by copying and concatenating
    the opposite ends of each tensor dimension to each other.
    The amount of padding is dynamically calculated to mimic Keras' standard 
    'same' padding behavior based on the provided kernel size and strides.

    Args:
        kernel_size (int or tuple): The size of the convolution window that 
            will follow this layer.
        strides (int or tuple, optional): The stride of the convolution operation 
            that will follow this layer. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """
    def __init__(self, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.k = kernel_size
        self.s = strides

    def call(self, x):
        """Pads the input tensor using periodic boundary conditions.

        Args:
            x (Tensor): The 4D input tensor with shape 
                `(batch_size, height, width, channels)`.

        Returns:
            Tensor: The padded 4D tensor with shape 
                `(batch_size, height + padding, width + padding, channels)`.
        """
        L = ops.shape(x)[1]
        
        # Calculate required padding to match 'same' padding behavior
        out_dim = (L + self.s - 1) // self.s
        pad_total = ops.maximum(0, (out_dim - 1) * self.s + self.k - L)
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        # Pad top and bottom
        top = x[:, L - pad_beg : L, :, :]
        bottom = x[:, 0 : pad_end, :, :]
        x = ops.concatenate([top, x, bottom], axis=1)

        # Pad left and right
        left = x[:, :, L - pad_beg : L, :]
        right = x[:, :, 0 : pad_end, :]
        x = ops.concatenate([left, x, right], axis=2)
        
        return x

    def get_config(self):
        """Returns the configuration of the layer for serialization.

        Returns:
            dict: A dictionary containing the layer's configuration.
        """
        return {**super().get_config(), "kernel_size": self.k, "strides": self.s}


@saving.register_keras_serializable(package="mylib")
class FiLMLayer(layers.Layer):
    """Applies Feature-wise Linear Modulation (FiLM) to input features.

    This layer conditions a neural network by applying an affine transformation 
    to convolutional feature maps based on the conditioning variable.
    It computes shifting (delta) and scaling (gamma) parameters using a small 
    Multi-Layer Perceptron (MLP) of standard size 32 or 64.

    Args:
        filters (int): The number of filters (channels) in the input feature map.
            This determines the size of the generated gamma and delta vectors.
        hidden_units (int, optional): The number of neurons in the hidden layer 
            of the conditioning MLP. Defaults to 32.
        **kwargs: Additional keyword arguments passed to the base Layer class.
    """
    def __init__(self, filters, hidden_units=64, **kwargs):
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
        """Applies the FiLM modulation to the feature maps.

        Args:
            inputs (tuple): A tuple containing two Tensors:
                - x_features (Tensor): The 4D convolutional feature maps to modulate 
                  with shape `(batch_size, height, width, channels)`.
                - beta (Tensor): The conditioning variable (e.g., temperature labels).

        Returns:
            Tensor: The modulated feature maps with the same shape as `x_features`.
        """
        x_features, beta = inputs
        
        film_hidden = self.dense_hidden(beta)
        
        # Generate scale (gamma) and shift (delta) parameters
        gamma = self.reshape(self.dense_gamma(film_hidden))
        delta = self.reshape(self.dense_delta(film_hidden))
        
        # Apply affine transformation: modulated = (features * gamma) + delta
        x_modulated = self.multiply([x_features, gamma])
        x_modulated = self.add([x_modulated, delta])
        
        return x_modulated

    def get_config(self):
        """Returns the configuration of the layer for serialization.

        Returns:
            dict: A dictionary containing the layer's configuration.
        """
        return {**super().get_config(), "filters": self.filters, "hidden_units": self.hidden_units}