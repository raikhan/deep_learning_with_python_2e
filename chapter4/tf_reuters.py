#
# Section 4.2 - multiclass classification
#
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from helpers import fit_and_evaluate, multi_hot_encode, plot_histories

(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(
    num_words=10000
)


def to_one_hot(labels, dimension=46):
    """
    One-hot encode the labels too
    """
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.0
    return results


x_train = multi_hot_encode(train_data)
x_test = multi_hot_encode(test_data)

# # we have to one-hot encode the labels too
# y_train = to_one_hot(train_labels)
# y_test = to_one_hot(test_labels)

# the keras way...
y_train = keras.utils.to_categorical(train_labels)
y_test = keras.utils.to_categorical(test_labels)

# validation dataset
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]


# Declare the base model
model_base = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),  # softmax = multi-sigmoid
    ]
)
model_base.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

history_base = fit_and_evaluate(
    model_base, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)


plot_histories(history_base)


model_bottleneck = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(46, activation="softmax"),  # softmax = multi-sigmoid
    ]
)
model_bottleneck.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

history_bottleneck = fit_and_evaluate(
    model_bottleneck, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)

plot_histories(history_base, history_bottleneck)


model_less_layers = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),  # softmax = multi-sigmoid
    ]
)
model_less_layers.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

history_less_layers = fit_and_evaluate(
    model_less_layers, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)


model_more_layers = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),  # softmax = multi-sigmoid
    ]
)
model_more_layers.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

history_more_layers = fit_and_evaluate(
    model_more_layers, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)


plot_histories(history_base, history_less_layers, history_more_layers)
