import matplotlib.pyplot as plt
import numpy as np
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

#
# Models
#
#


def fit_and_evaluate(model):

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
    )

    results = model.evaluate(x_test, y_test)
    print(f"Test loss: {results[0]:.4f}, accuracy: {results[1]:.4f}")

    return history


def plot_histories(*args):

    print(args)
    colors = ["b", "r", "g", "k", "y"]

    for i, history in enumerate(args):

        plt.plot(
            history.epoch,
            history.history["loss"],
            f"{colors[i]}o",
            label=f"Model {i} training",
        )
        plt.plot(
            history.epoch,
            history.history["val_loss"],
            f"{colors[i]}",
            label=f"Model {i} validation",
        )

    plt.title("Training vs validation loss on IMBD")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


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
history_base = fit_and_evaluate(model_base)

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
history1 = fit_and_evaluate(model1)


# underfit
model2 = keras.Sequential(
    [
        layers.Dense(4, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model2.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history2 = fit_and_evaluate(model2)


plot_histories(history_base)

plot_histories(history_base, history1, history2)
