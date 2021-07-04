#
# Chapter 4.1: IMDB sentiment classification with Keras
#
import joblib
import numpy as np
from numpy.lib.index_tricks import _ix__dispatcher
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from plots import plot_loss_and_acc

#
# Parameters
#
n_validation = 10000
epochs = 20
batch_size = 512
learning_rate = 0.01

# load the prepared data
(X_train, y_train), (X_test, y_test) = joblib.load("imdb_ml_data.joblib")

# set asside a validation set from the train set
# Since the labels are already shuffled, we can just split the dataset
X_val = X_train[:n_validation]
X_partial_train = X_train[n_validation:]
y_val = y_train[:n_validation]
y_partial_train = y_train[n_validation:]

# convert to PyTorch tensors and datasets/dataloaders
X_train = torch.from_numpy(X_partial_train.astype("float32"))
y_train = torch.from_numpy(y_partial_train)
train_ds = TensorDataset(X_train, y_train)

X_val = torch.from_numpy(X_val.astype("float32"))
y_val = torch.from_numpy(y_val)
val_ds = TensorDataset(X_val, y_val)

X_test = torch.from_numpy(X_test.astype("float32"))
y_test = torch.from_numpy(y_test)
test_ds = TensorDataset(X_test, y_test)

train_dl = DataLoader(train_ds, batch_size=batch_size)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10000, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # sigmoid computed in loss function

        return x


# training loop
train_loss = []
val_loss = []
train_acc = []
val_acc = []

model = Model()
loss_func = nn.BCEWithLogitsLoss()  # Sigmoid + Binary cross-entropy loss
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


def get_metrics(X, y):
    preds = model(X)
    loss = loss_func(preds, y.unsqueeze(1))
    acc = ((preds.squeeze() > 0.5) == y).float().mean()

    return float(loss), float(acc)


for epoch in range(epochs):

    # train for one epoch
    for i, data in enumerate(train_dl, 0):
        X, y = data

        optimizer.zero_grad()

        outputs = model(X)
        loss = loss_func(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()

    # compute training loss and accuracy
    tr_loss, tr_acc = get_metrics(X_train, y_train)
    train_loss.append(tr_loss)
    train_acc.append(tr_acc)

    # compute validation loss and accuracy
    v_loss, v_acc = get_metrics(X_val, y_val)
    val_loss.append(v_loss)
    val_acc.append(v_acc)

    print(f"Epoch: {epoch} -> train_acc: {tr_acc:.4f}, val_acc: {v_acc:.4f}")

loss, acc = get_metrics(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

x_data = np.arange(1, epochs + 1, 1)
train_loss = np.array(train_loss)
train_acc = np.array(train_acc)
val_loss = np.array(val_loss)
val_acc = np.array(val_acc)

plot_loss_and_acc(x_data, train_loss, val_loss, train_acc, val_acc)
