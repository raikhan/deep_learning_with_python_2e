#
# Chapter 4.2: predicting topics of Reuters news articles
#
import joblib


# load prepared data
(X_train, y_train), (X_test, y_test) = joblib.load("reuters_ml_data.joblib")
