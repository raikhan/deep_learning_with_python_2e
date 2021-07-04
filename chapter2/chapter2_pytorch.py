#
# Writing a MNIST classifier from scratch using PyTorch instead of TensorFlow
# Based on 60 min intro to PyTorch:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
#
import math
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


data = joblib.load("mnist_keras_data.joblib")

train_images = data["train"]["images"]
train_labels = data["train"]["labels"]
test_images = data["test"]["images"]
test_labels = data["test"]["labels"]

# reshape the data before training
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


# Create TensorDataset for training
train_ds = TensorDataset(
    torch.from_numpy(train_images), torch.from_numpy(train_labels.astype(np.int64))
)


class MyLinear(nn.Module):
    def __init__(self, input_len, output_len):
        super().__init__()

        # initialize the parameters
        self.weights = nn.Parameter(
            torch.randn(input_len, output_len) / math.sqrt(input_len)
        )  # Xavier initialization
        self.bias = nn.Parameter(torch.zeros(output_len))

    def forward(self, xb):
        return xb @ self.weights + self.bias


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.fc1 = nn.Linear(28 * 28, 512)
        # self.fc2 = nn.Linear(512, 10)

        self.fc1 = MyLinear(28 * 28, 512)
        self.fc2 = MyLinear(512, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)

        return x


net = Net()
loss = nn.CrossEntropyLoss()
learning_rate = 1e-2  # NOTE: 10 times smaller loss here
n_epochs = 10

# training loop
for epoch in range(n_epochs):
    print(f"{epoch=}")

    # use torch DataLoader for batching
    train_dl = DataLoader(train_ds, batch_size=128)

    batch_counter = 0
    for pt_images, pt_labels in train_dl:

        # compute loss and do backprop to get gradients
        pred = net(pt_images)
        output = loss(pred, pt_labels)
        net.zero_grad()
        output.backward()

        # update parameters using gradients
        for f in net.parameters():
            f.data.sub_(f.grad.data * learning_rate)

        batch_counter += 1
        if batch_counter % 100 == 0:
            print(f"Batch {batch_counter}, loss: {output}")

# test the model
t = torch.from_numpy(test_images)
pred = net(t).argmax(1).numpy()

res = pred == test_labels
print(f"Accuracy: {res.mean()}")
