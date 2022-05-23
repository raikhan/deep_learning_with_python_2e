#
# Following along the PyTorch tutorial in the official docs
#
import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor


mnist_data = joblib.load("data/mnist_keras_data.joblib")

# make the Keras data into PyTorch dataloaders
def prepare_dataloader(dataset: str) -> DataLoader:
    images = torch.Tensor(mnist_data["images"][dataset])
    labels = torch.Tensor(mnist_data["labels"][dataset])

    tmp = TensorDataset(images, labels)

    return DataLoader(tmp)


train = prepare_dataloader("train")
test = prepare_dataloader("test")
