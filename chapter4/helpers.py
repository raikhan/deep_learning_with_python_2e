import numpy as np
import matplotlib.pyplot as plt


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


def fit_and_evaluate(
    model, partial_x_train, partial_y_train, x_val, y_val, x_test, y_test
):

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
