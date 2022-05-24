#
# Writing the Chapter 2 MNIST example with PyTorch
#
# Using the PyTorch docs as a guide:
# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#
import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# load the TF dataset
mnist_data = joblib.load("data/mnist_keras_data.joblib")


# make the Keras data into PyTorch dataloaders
def prepare_dataloader(dataset: str, batch_size: int = 64) -> DataLoader:
    images = torch.Tensor(mnist_data["images"][dataset])
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2]) / 255.0

    labels = torch.Tensor(mnist_data["labels"][dataset])
    labels = labels.type("torch.LongTensor")

    tmp = TensorDataset(images, labels)

    return DataLoader(tmp, batch_size=batch_size)


train_dl = prepare_dataloader("train", batch_size=128)  # same batch size as Chapter 2
test_dl = prepare_dataloader("test", batch_size=128)


# Create model
class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super(NeuralNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 10)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    """
    Train the model for one epoch, i.e. one pass over the training set
    """

    size = len(dataloader.dataset)

    # set the model to train mode - so that the weights can be updated
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()  # zero-out gradients (Something Keras does automatically)
        loss.backward()  # compute gradients
        optimizer.step()  # update weights

        # print the loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    """
    Compute the predictions of the model
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # set the model to evaluation mode
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():  # switching off autograd for evaluation
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


#
# Main training loop
#

model = NeuralNet()
print(model)

loss_fn = nn.CrossEntropyLoss()
# Using the same optimization algo as in Chapter 2
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

print("Before any training")
test(test_dl, model, loss_fn)

for _ in range(5):  # train for 5 epochs
    train(train_dl, model, loss_fn, optimizer)
    test(test_dl, model, loss_fn)

# Final accuracy ~ 97%, very similar to Keras result in keras_basics_and_auto_diff.py
