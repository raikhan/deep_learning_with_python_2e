#
# Section 4.3. Regression of Boston house prices
#
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold


(train_data, train_targets), (
    test_data,
    test_targets,
) = keras.datasets.boston_housing.load_data()


# standardize the data
# NOTE: using only the train dataset to compute the standardizing stats
train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)

train_data -= train_mean
train_data /= train_std
test_data -= train_mean
test_data /= train_std


def build_model():
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),  # NOTE no activation for regression
        ]
    )
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

    return model


# cross validation
all_mae = []
for train_indices, val_indices in KFold(n_splits=4).split(train_data):
    X_train = train_data[train_indices]
    y_train = train_targets[train_indices]

    X_val = train_data[val_indices]
    y_val = train_targets[val_indices]

    model = build_model()
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

    val_mse, val_mae = model.evaluate(X_val, y_val, verbose=0)
    all_mae.append(val_mae)

print(f"Per fold MAE: {all_mae}")
print(f"Mean MAE: {np.mean(all_mae)}")
print(
    f"(for context) Target mean and std: {train_targets.mean()}, {train_targets.std()}"
)


# skiping the results plot, nothing new here compared to two previous examples
