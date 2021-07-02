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

# reusing the batch generator from chapter2.py
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
    batch_generator = BatchGenerator(
        train_images, train_labels, batch_size=128, randomize=False
    )
    batch_counter = 0
    while batch := batch_generator.next():

        # prepare data
        pt_images = torch.from_numpy(batch[0])
        pt_labels = torch.from_numpy(batch[1].astype("int64"))

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
