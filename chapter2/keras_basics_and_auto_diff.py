import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.datasets import mnist

#
# Basic MNIST example
#

# load data
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

# declare the model
model = keras.Sequential(
    [layers.Dense(512, activation="relu"), layers.Dense(10, activation="softmax")]
)
model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


# reshape data for training
train_img = train_img.reshape((60000, 28 * 28))
train_img = train_img.astype("float32") / 255
test_img = test_img.reshape((10000, 28 * 28))
test_img = test_img.astype("float32") / 255


# train the model
model.fit(train_img, train_labels, epochs=5, batch_size=128)

# predict the test images
preds = model.predict(test_img)

# compute accuracy
pred_class = preds.argmax(axis=1)
acc = (pred_class == test_labels).mean()
print(f"MNIST example accuracy: {acc}")


#
# Working with GradientTape automatic differentiation
#

# 1) single variable
x = tf.Variable(0.0)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad = tape.gradient(y, x)
print(grad.numpy())


# 2) array
x = tf.Variable(tf.random.uniform((2, 2)))
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad = tape.gradient(y, x)
print(grad.numpy())


# 3) list of TF variables
W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b
grad = tape.gradient(y, [W, b])
print(f"W: ", grad[0].numpy())
print(f"b: ", grad[1].numpy())
