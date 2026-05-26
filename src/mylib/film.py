import keras as k
from keras import ops, layers
from keras import saving

@saving.register_keras_serializable(package="projectlib")
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
