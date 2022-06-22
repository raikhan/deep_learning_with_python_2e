import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


def multi_hot_encode(sequences, dimension=10000):
    """
    Function to hot-encode the sequence of integers. Hot encoding turns a sequence of
    integers into an array of set length, where value at index i is zero if i is not
    in the sequence and it's one if i is in the sequence
    """

    res = np.zeros((len(sequences), dimension))

    # for every sequence...
    for i, seq in enumerate(sequences):
        # for every integer in the sequence...
        for j in seq:
            # set the value for each integer in the sequence to 1
            # NOTE: multiple instances of a word in a sequence are ignored!
            res[i, j] = 1.0

    return res


# encode features
x_train = multi_hot_encode(train_data)
x_test = multi_hot_encode(test_data)

# convert labels to numpy
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# add a validation set from training set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Defining the model
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])


# fit the model using validation set for monitoring progress
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val),
)
