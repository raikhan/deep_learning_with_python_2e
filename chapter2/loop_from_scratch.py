import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist


#
# Reimplementing the Keras layers
#
class NaiveDense:
    """One dense layer of the DL"""

    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=0.1)
        self.W = tf.Variable(w_initial_value)

        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    """
    Forward pass through layers sequentially
    """

    def __init__(self, layers) -> None:
        self.layers = layers

    def __call__(self, inputs):
        """forward pass"""
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


class BatchGenerator:
    """
    Generate batches of images
    """

    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)

        # batch counter
        self.index = 0

        self.images = images
        self.labels = labels
        self.batch_size = batch_size

        self.num_batches = math.ceil(len(images) / batch_size)

    def next(self):
        """
        Return the next batch
        """

        images = self.images[self.index : self.index + self.batch_size]
        labels = self.labels[self.index : self.index + self.batch_size]

        self.index += self.batch_size

        return images, labels


#
# With those three basic classes, we can run a training loop - changing
# weights based on the predictions generated on a batch of images/labels
#
def one_training_step(model, optimizer, images_batch, labels_batch):
    """
    One step of the training loop:
    - do a forward pass on a batch of images
    - compute the loss by comparing predictions and actuals
    - back-propagate the loss to update weights
    """

    with tf.GradientTape() as tape:

        # forward pass to predict on input images
        predictions = model(images_batch)

        # computing loss
        per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions
        )
        avg_loss = tf.reduce_mean(per_sample_loss)

    # compute gradients
    gradients = tape.gradient(avg_loss, model.weights)

    # update the model weights with computed gradients, using Keras optimizer
    # object
    optimizer.apply_gradients(zip(gradients, model.weights))

    return avg_loss


def fit(model, images, labels, epochs=2, batch_size=128, learning_rate=1e-3):
    """
    The whole training loop
    """

    optimizer = optimizers.SGD(learning_rate=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")

        batch_generator = BatchGenerator(images, labels, batch_size=batch_size)

        # iterate through the training dataset in batches
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()

            loss = one_training_step(model, optimizer, images_batch, labels_batch)

            if batch_counter % 100 == 0:
                print(f"Loss at batch {batch_counter}: {loss:.2f}")


#
# Test out the implementation on MNIST
#
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(f"Loaded data - train:{train_images.shape}, test: {test_images.shape}")


# reshape images to 1D arrays
train_images = train_images.reshape(60000, 28 * 28)
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype("float32") / 255

# declare the model
model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
        NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax),
    ]
)
assert len(model.weights) == 4


# run the training loop
fit(model, train_images, train_labels, epochs=20)


# check the accuracy of the trained model
preds = model(test_images)
preds = preds.numpy()

pred_labels = np.argmax(preds, axis=1)
matches = pred_labels == test_labels
print(f"\nAccuracy: {matches.mean():.2f}")
