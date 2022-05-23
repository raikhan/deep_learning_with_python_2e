#
# Load the data from Keras to use in the PyTorch version of Chapter 2
#
from tensorflow.keras.datasets import mnist

import joblib

print("Exporting MNRAS dataset from Keras")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

data = {
    "images": {"train": train_images, "test": test_images},
    "labels": {"train": train_labels, "test": test_labels},
}

joblib.dump(data, "data/mnras_keras_data.joblib")
print("Done")
