import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import re
import sys

path = "datasets/"

# Reading file and creating pandas dataframe
def create_array_from_file(file: str):
    with open (path + file, "r") as f:
        np_array = np.loadtxt(f, dtype=np.float32)
        return np_array

def housing_data_handling():
    housing_training = create_array_from_file("housing_training.txt")
    housing_test = create_array_from_file("housing_test.txt")
    X_training = get_features(housing_training)
    X_test = get_features(housing_test)

    X_training = min_max_normalize(X_training)
    X_test = min_max_normalize(X_test)

    y_training = get_labels(housing_training) 
    y_test = get_labels(housing_test)

    return X_training, X_test, y_training, y_test
    
# Min-max normalization of data, using train min/max to avoid data leakage from test set
def min_max_normalize(data):
    return ((data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)))

def zero_mean_normalize(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std

# Returns all but last column (features) of dataset
def get_features(data):
    return data[:, :-1] 

# Returns last column (labels) of dataset and reshapes into column vector
def get_labels(data):
    y = data[:, -1]
    return y.reshape(y.shape[0], 1)

# Adds bias - x is df, m is number of samples, n is no. of features
# creates column of ones and appends to start of array
def add_bias(x):
    return np.c_[np.ones((x.shape[0], 1)), x]

def mean_sum_of_squares(y, y_hat):
    squared_differences = np.square(y - y_hat)
    mse = np.mean(squared_differences)
    return mse

def spambase_data_handling():
    # Load file to np array
    data = np.loadtxt(path + "spambase.data", delimiter=",", dtype=np.float32)
    n_samples, n_features = data.shape
    n_features -= 1

    # get features and targets
    X = data[:, 0:n_features]
    y = data[:, n_features]
    return X, y


def fetch_20NG_data_handling(categories):
    newsgroups_train = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42, subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(categories=categories, shuffle=True, random_state=42, subset='test', remove=('headers', 'footers', 'quotes'))
    X_train, X_test, y_train, y_test = newsgroups_train.data, newsgroups_test.data, newsgroups_train.target, newsgroups_test.target
    classes = {}
    for idx, cat in enumerate(newsgroups_train.target_names):
        classes[idx] = cat
    return X_train, X_test, y_train, y_test, classes

def percepton_data_handling():
    perceptron_data = create_array_from_file("perceptron_data.txt")
    return get_features(perceptron_data), get_labels(perceptron_data)
