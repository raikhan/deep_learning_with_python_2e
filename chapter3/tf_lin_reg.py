#
# Doing linear regression with TF, from Chapter 3
#
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# generate dummy data
num_samples_per_class = 1000
class0 = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)
class1 = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class
)

# combine two classes into a single array
inputs = np.vstack((class0, class1)).astype(np.float32)

# create labels for points
targets = np.vstack(
    (
        np.zeros((num_samples_per_class, 1), dtype="float32"),
        np.ones((num_samples_per_class, 1), dtype="float32"),
    )
)

# look at the generated data
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.title("Generated data")
plt.show()

# split data to train and test
train_inputs, test_inputs, train_targets, test_targets = train_test_split(
    inputs, targets, test_size=0.2, random_state=42
)

# declare weights variables
input_dim = 2
output_dim = 1

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

# make the linear model
def model(inputs):
    return tf.matmul(inputs, W) + b


# loss function: MSE
def square_loss(targets, predictions):
    per_sample_loss = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_loss)


# function defining one training step: forward pass to make predictions, compute losses,
# compute gradients of losses per parameter and update parameters with a step along
# the gradient
def training_loop(inputs, targets, learning_rate=0.1):

    # forward pass
    with tf.GradientTape() as tape:

        preds = model(inputs)
        loss = square_loss(targets, preds)

    # compute gradients with the tape
    grad_loss_W, grad_loss_b = tape.gradient(loss, [W, b])

    # update weights
    W.assign_sub(grad_loss_W * learning_rate)
    b.assign_sub(grad_loss_b * learning_rate)

    return loss


# train the model using training_loop function
# NOTE: training across full batch i.e. the whole training set. This is slower to
# compute but better for learning, so the learning rate should be bigger than in
# the mini-batch case


for step in range(40):
    loss = training_loop(train_inputs, train_targets)
    print(f"Loss at step {step}: {loss:.4f}")

# print accuracy
preds = model(test_inputs) > 0.5
acc = (preds == test_targets).numpy().mean()
print(f"Accuracy: {acc:.4f}")


# plot predictions with the model line
x = np.linspace(-1, 4, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
plt.scatter(test_inputs[:, 0], test_inputs[:, 1], c=preds[:, 0])
plt.title("Predictions on test data set and model line")
plt.show()


#
# Now I will do the same model using Keras API
#
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=["accuracy"])

hist = model.fit(train_inputs, train_targets, epochs=5, batch_size=128, shuffle=False)
pred = model.predict(test_inputs)
acc = ((pred > 0) == test_targets).mean()
print(
    f"Keras accuracy: {acc:.4f}"
)  # WHAT IS HAPPENING??? VERY DIFFERENT RESULTS BETWEEN RUNS
