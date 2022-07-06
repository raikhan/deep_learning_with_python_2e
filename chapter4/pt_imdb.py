import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Lambda

import matplotlib.pyplot as plt


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

        # NOTE adding dummy dimension to y to match X shape
        loss = loss_fn(pred, y.unsqueeze(dim=1))

        # Backpropagation
        optimizer.zero_grad()  # zero-out gradients (Something Keras does automatically)
        loss.backward()  # compute gradients
        optimizer.step()  # update weights


def evaluate(dataloader, model, loss_fn):
    """
    Compute the predictions of the model on the suplied dataset
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # set the model to evaluation mode
    model.eval()

    test_loss, accuracy = 0, 0
    with torch.no_grad():  # switching off autograd for evaluation
        for X, y in dataloader:

            # NOTE adding dummy dimension to y to match X shape
            y_reshaped = y.unsqueeze(dim=1)

            pred_logits = model(X)
            pred_proba = nn.Sigmoid()(pred_logits)
            pred = pred_proba > 0.5

            test_loss += loss_fn(pred_logits, y_reshaped).item()
            accuracy += (pred == y_reshaped).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy /= size

    return accuracy, test_loss


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
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)


acc, loss = evaluate(dl_test, model, loss_fn)
print(f"Pre-training -> Accuracy: {acc:.4f}, Loss: {loss:.4f}")

#
# Model training
#
val_loss = []
train_loss = []
val_acc = []
train_acc = []
epochs = list(range(20))

for epoch in epochs:
    train(dl_train, model, loss_fn, optimizer)

    epoch_val_acc, epoch_val_loss = evaluate(dl_valid, model, loss_fn)
    epoch_train_acc, epoch_train_loss = evaluate(dl_train, model, loss_fn)

    print(
        f"Epoch {epoch} -> Train loss: {epoch_train_loss:.4f}, "
        f"Validation loss: {epoch_val_loss:.4f}, "
        f"Validation accuracy: {epoch_val_acc:.4f}"
    )

    val_loss.append(epoch_val_loss)
    train_loss.append(epoch_train_loss)
    val_acc.append(epoch_val_acc)
    train_acc.append(epoch_train_acc)

#
# Final evaluation on test set
#
acc, loss = evaluate(dl_test, model, loss_fn)
print(f"After training -> Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

#
# Make loss plot
#
plt.plot(
    epochs,
    train_loss,
    f"ro",
    label=f"Training",
)
plt.plot(
    epochs,
    val_loss,
    f"b-",
    label=f"Validation",
)

plt.title("Training vs validation loss on IMBD")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-entropy loss")
plt.legend()
plt.show()
