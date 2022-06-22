import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

from helpers import fit_and_evaluate, plot_histories, multi_hot_encode

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


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

#
# Models
#
#

model_base = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model_base.compile(
    optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"]
)
history_base = fit_and_evaluate(
    model_base, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)

# overfit
model1 = keras.Sequential(
    [
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model1.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history1 = fit_and_evaluate(
    model1, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)


# underfit
model2 = keras.Sequential(
    [
        layers.Dense(4, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model2.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history2 = fit_and_evaluate(
    model2, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
)


plot_histories(history_base)

plot_histories(history_base, history1, history2)
