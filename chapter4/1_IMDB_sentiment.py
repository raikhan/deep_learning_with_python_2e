#
# Chapter 4.1: IMDB sentiment classification with Keras
#
import joblib
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from plots import plot_loss_and_acc

# load the prepared data
(X_train, y_train), (X_test, y_test) = joblib.load("imdb_ml_data.joblib")

#
# Parameters
#
n_validation = 10000
epochs = 20
batch_size = 512

# define the Keras model
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# set asside a validation set from the train set
# Since the labels are already shuffled, we can just split the dataset
X_val = X_train[:n_validation]
X_partial_train = X_train[n_validation:]
y_val = y_train[:n_validation]
y_partial_train = y_train[n_validation:]

# fit the model
history = model.fit(
    X_partial_train,
    y_partial_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
)

# plot training vs validation loss
train_loss = np.array(history.history["loss"])
val_loss = np.array(history.history["val_loss"])
train_acc = np.array(history.history["accuracy"])
val_acc = np.array(history.history["val_accuracy"])
x_data = np.array(history.epoch) + 1

plot_loss_and_acc(x_data, train_loss, val_loss, train_acc, val_acc)

# the model overfits and the best iteration is epoch=4
# Train the model for 4 epochs and evaluate on test
model.fit(X_train, y_train, epochs=4, batch_size=batch_size)
results = model.evaluate(X_test, y_test)
print(results)

predictions = model.predict(X_test)
print(predictions)
