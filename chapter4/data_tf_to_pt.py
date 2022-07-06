#
# Extract keras datasets for use with PT
#

import joblib
from tensorflow.keras.datasets import imdb, reuters, boston_housing


def data_loader(tf_obj, **kwargs):

    data = {}
    (train_data, train_labels), (test_data, test_labels) = tf_obj.load_data(**kwargs)

    data["train"] = {}
    data["train"]["data"] = train_data
    data["train"]["labels"] = train_labels

    data["test"] = {}
    data["test"]["data"] = test_data
    data["test"]["labels"] = test_labels

    return data


# load data
imdb_data = data_loader(imdb, num_words=10000)
reuters_data = data_loader(reuters, num_words=10000)
boston_data = data_loader(boston_housing)

# dump to joblib
joblib.dump(imdb_data, "./data/imdb_data.joblib")
joblib.dump(reuters_data, "./data/reuters_data.joblib")
joblib.dump(boston_data, "./data/boston_housing_data.joblib")
