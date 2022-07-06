import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Lambda


class IMDBDataset(Dataset):
    def __init__(self, dataset):

        raw_data = joblib.load("./data/imdb_data.joblib")[dataset]  # pick train or test

        self.X = raw_data["data"]
        self.y = raw_data["labels"]

        #
        # I put the required transforms here
        #
        self.transform = Lambda(
            lambda y: torch.zeros(10000, dtype=torch.float).scatter_(
                dim=0, index=torch.tensor(y), value=1
            )
        )  # multi-hot encoding for features

        # convert targets to int, as expected by nn.CrossEntropyLoss
        self.target_transform = Lambda(lambda y: torch.tensor(y, dtype=torch.float))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # apply transforms before returning requested data
        X = self.transform(self.X[idx])
        y = self.target_transform(self.y[idx])

        return X, y


class IMDBNet(nn.Module):
    def __init__(self) -> None:
        super(IMDBNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(10000, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
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

            # NOTE adding dummy dimension to y to match X shape
            test_loss += loss_fn(pred, y.unsqueeze(dim=1)).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


torch.manual_seed(42)

#
# Load data
#
ds_train_full = IMDBDataset("train")
ds_test = IMDBDataset("test")

ds_train, ds_valid = random_split(ds_train_full, [15000, 10000])

dl_train = DataLoader(ds_train, batch_size=64, shuffle=False)
dl_valid = DataLoader(ds_valid, batch_size=64, shuffle=False)
dl_test = DataLoader(ds_test, batch_size=64, shuffle=False)

#
# Setup the model
#
model = IMDBNet()
loss_fn = nn.functional.binary_cross_entropy_with_logits
# Using the same optimization algo as in Chapter 2
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)


test(dl_train, model, loss_fn)
