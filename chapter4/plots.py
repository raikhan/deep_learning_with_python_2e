import matplotlib.pyplot as plt


def plot_loss(epochs, train_loss, val_loss, ax=None):

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(epochs, train_loss, "bo", label="Training")
    ax.plot(epochs, val_loss, "b", label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    return ax


def plot_acc(epochs, train_acc, val_acc, ax=None):

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.plot(epochs, train_acc, "ro", label="Training")
    ax.plot(epochs, val_acc, "r", label="Validation")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()

    return ax


def plot_loss_and_acc(epochs, train_loss, val_loss, train_acc, val_acc):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    plot_loss(epochs, train_loss, val_loss, ax=ax1)
    plot_acc(epochs, train_acc, val_acc, ax=ax2)

    fig.show()
