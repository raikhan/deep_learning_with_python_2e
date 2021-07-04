import joblib
import numpy as np
from numpy.core.numeric import _ones_like_dispatcher
from tensorflow.keras.datasets import imdb, reuters
from tensorflow.keras.utils import to_categorical

#
# Parameters
#
n_words = 10000


#
# Prepare data for model
# Using multi-hot encoding, so vector is the length of dictionary, column is each
# word ID and values are 1 if the word is present otherwise 0
#
def multihot_encoding(sequences, dimension=n_words):
    res = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        for j in seq:
            res[i, j] = 1  # in bag-of-words, that would be the word cound (+=1)
    return res


def prep_classification_data(fname, X_train, y_train, X_test, y_test, one_hot=False):
    """Clean up data for classification models in chapter 4"""

    X_train = multihot_encoding(X_train)
    X_test = multihot_encoding(X_test)
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    data = ((X_train, y_train), (X_test, y_test))
    joblib.dump(data, fname)


#
# 1) IMDB - binary classification
#

# load the IMDB data, limiting the dictionary to top 10k most common words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=n_words)
prep_classification_data("imdb_ml_data.joblib", X_train, y_train, X_test, y_test, False)


#
# 2) Reuters - multiclass classification
#

# load the IMDB data, limiting the dictionary to top 10k most common words
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=n_words)
prep_classification_data(
    "reuters_ml_data.joblib", X_train, y_train, X_test, y_test, True
)
