#
# Chapter 2: Mathematical building blocks of neural networks
#
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Load MNIST data from Keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Train: ", train_images.shape)
print("Test: ", test_images.shape)

# Set up the simple fully-connected NN with keras layers
model = keras.Sequential(
    [layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")]
)

# compile the model to prepare it for training
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# reshape the data before training
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# fit the model
model.fit(x=train_images, y=train_labels, epochs=5)

# predict with the model
pred_proba = model.predict(test_images)
pred = pred_proba.argmax(axis=1)

# use keras API to evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"{test_acc=}")

#
# 2.2: Tensors
#

# Reload raw images, overwritting their normalized
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

# plot a digit
i = 12
digit = test_images[i]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print(test_labels[i])

#
# 2.3: Neural nets with numpy
#
import numpy as np

# a Dense layer converts input vectors into output vectors by applying the operation:
# x_out = activation(x*W + b)
# When layer is defined as layers.Dense(100, activation="relu"), this means that x_out will have len=100.

# how tensor product reshapes matrices
x = np.random.random((20, 40, 100))
y = np.random.random((100,))
print(np.dot(x, y).shape)  # (20, 40)

x = np.random.random((20, 40, 100))
y = np.random.random((100, 50))
print(np.dot(x, y).shape)  # (20, 40, 50)

#
# Auto-grad in TensorFlow: GradientTape
#

# scalar
x1 = tf.Variable(0.0)  # tf.Variable is differentiable
x2 = tf.Variable(0.0)
with tf.GradientTape() as tape:
    y = 2 * x1 + 4 * x2 + 3
grad_of_y_over_xs = tape.gradient(y, [x1, x2])
print(grad_of_y_over_xs)

# rank-2 tensor
x = tf.Variable(tf.random.uniform((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_over_xs = tape.gradient(y, x)
print(grad_of_y_over_xs)

# combination
W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
x2 = tf.Variable(0.0)
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
grad_of_y = tape.gradient(y, [W, b])
print(grad_of_y)

#
# 2.5.1: Reimplementing the dense NN from scratch in TF
#


class NaiveDense:
    """One layer of the densly connected NN"""

    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.activation = activation

        # initialize weights
        w_shape = (input_size, output_size)
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)
        self.W = tf.Variable(w_initial_value)

        # initialize bias
        b_shape = (output_size,)
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):
        assert inputs.shape[-1] == self.input_size  # I added this

        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):
        return [self.W, self.b]


class NaiveSequential:
    """Class to sequentially apply layers of the NN"""

    def __init__(self, layers) -> None:
        self.layers = layers

    def __call__(self, inputs):
        x = inputs

        # apply layers in order (use their __call__)
        for layer in self.layers:
            x = layer(x)

        return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


# now generate batches for training
class BatchGenerator:
    def __init__(self, images, labels, batch_size=128, randomize=False):
        self.index = 0
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

        if randomize:
            print("Randomizing the data order")
            i = np.arange(len(self.labels))
            np.random.shuffle(i)
            self.images = self.images[i]
            self.labels = self.labels[i]

    def next(self):

        # ran across all examples
        if self.index >= self.labels.size:
            return None

        # last step will be smaller than the requested batch size
        if self.index + self.batch_size > self.labels.size:
            batch_size = self.labels.size - self.index
        else:
            batch_size = self.batch_size

        images = self.images[self.index : self.index + batch_size]
        labels = self.labels[self.index : self.index + batch_size]

        self.index += batch_size

        return images, labels


# One batch update is the core of the training process. In the function below, we will:
# 1) compute the forward pass of the model to get predictions
# 2) compute the loss by comparing prediction labels to actuals
# 3) compute the gradient of loss with respect to all the weights in the model
# 4) shift the weights in the model in the direction opposite to the gradient
def one_batch_training_step(model, images_batch, labels_batch):

    # establish the computational graph so it can be differentiated
    with tf.GradientTape() as tape:
        # forward pass
        predictions = model(images_batch)

        # loss for each sample
        per_sample_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions
        )
        average_loss = tf.reduce_mean(per_sample_loss)

    # compute gradients
    gradients = tape.gradient(average_loss, model.weights)

    # update the model weights
    update_weights(gradients, model.weights)

    return average_loss


learning_rate = 1e-3
# Simplest way to update the weights is to move them by a fraction of the computed
# gradient, i.e. the learning rate
def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)  # assign_sub is -= on tf.Variable


# # However, in practice, we will use more complex optimizers, like this one provided
# # in Keras
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)


# def update_weights(gradients, weights):
#     optimizer.apply_gradients(zip(gradients, weights))


# Finally, the whole training loop
# 1 epoch is a for loop over all data available in the training set, one batch at a time
def fit(model, images, labels, epochs=5, batch_size=128, randomize=False):
    for epoch in range(epochs):
        print(f"{epoch=}")
        batch_generator = BatchGenerator(images, labels, batch_size, randomize)
        batch_counter = 0
        while batch := batch_generator.next():
            images_batch, labels_batch = batch
            loss = one_batch_training_step(model, images_batch, labels_batch)
            batch_counter += 1
            if batch_counter % 100 == 0:
                print(f"Loss at batch {batch_counter}: {loss:.2f}")


# Now let's run our model training

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# combine the naive dense layers into a naive sequential object to define the model
model = NaiveSequential(
    [
        NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
        NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax),
    ]
)
assert len(model.weights) == 4

fit(model, train_images, train_labels, epochs=10, batch_size=128, randomize=True)

# evaluate the model
pred = model(test_images)
pred = pred.numpy()
pred_labels = np.argmax(pred, axis=1)
matches = test_labels == pred_labels
print(f"Accuracy: {matches.mean()}")
# Accuracy is ~0.81, much lower than the full Keras example above
# Accuracy is the same even when using the most naive weights update
# No improvement when randomizing batch generator data
