import joblib
import numpy as np
from tensorflow.keras.datasets import imdb

#
# Parameters
#
n_words = 10000

# load the IMDB data, limiting the dictionary to top 10k most common words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=n_words)

# # read the text by getting the vocabulary mapping
# word_index = imdb.get_word_index()

# # ID to word, defining the special characters 0,1 and 2
# id_to_word = {v + 3: k for k, v in word_index.items()}
# id_to_word[0] = "PADDING"
# id_to_word[1] = "START"
# id_to_word[2] = "UNKNOWN"
# review = " ".join([id_to_word.get(i, "?") for i in X_train[1234]])
# print(review)

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


X_train = multihot_encoding(X_train)
X_test = multihot_encoding(X_test)
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

data = ((X_train, y_train), (X_test, y_test))
joblib.dump(data, "imdb_ml_data.joblib")
