#
# Implementing Keras layers by hand, as in Chapter 3
#
import tensorflow as tf
from tensorflow import keras


# A basic fully-connected layer, same as NaiveDense in Chapter 2,
# but implemneted according to Keras API
class SimpleDense(keras.layers.Layer):
    def __init__(self, output_units, activation=None):
        super().__init__()
        self.output_units = output_units
        self.activation = activation

    def build(self, input_shape):
        """
        Method Keras API uses to prepare weights. Dedicated method to build
        parameters outside of __init__ allows for input shape inference, i.e.
        the parameters are prepared with build only when the first data is
        passed through the layer
        """
        input_dim = input_shape[-1]

        # self.add_weight is a method on keras.layers.Layer
        self.W = self.add_weight(
            shape=(input_dim, self.output_units), initializer="random_normal"
        )
        self.b = self.add_weight(shape=(self.output_units,), initializer="zeros")

    def call(self, inputs):
        """
        Keras does main layer calculation, the forward pass in call().
        This is not done in the default __call__() because that one is
        already implemented in keras.layers.Layer to run build() on first
        data input, then just run call() to do the computation
        """
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation:
            y = self.activation(y)
        return y
