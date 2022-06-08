#
# Implementing Keras layers by hand, as in Chapter 3
#
import tensorflow as tf
from tensorflow import keras


# A basic fully-connected layer, same as NaiveDense in Chapter 2,
# but implemneted according to Keras API
class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """Method Keras API uses to prepare weights"""
        input_dim = input_shape[-1]

        # self.add_weight is a method on keras.layers.Layer
        self.W = self.add_weight(
            shape=(input_dim, self.units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.units,), initializer="zeros")

    def call(self, inputs):
        """Keras does main layer calculation, the forward pass in call()"""
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation:
            y = self.activation(y)
        return y
