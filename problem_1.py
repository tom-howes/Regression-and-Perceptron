import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from data_handling import zero_mean_normalize, spambase_data_handling, housing_data_handling, add_bias, mean_sum_of_squares
from normal_linear import Normal_linear
from ridge_linear import Ridge_linear
import sys

float_formatter = "{:.3f}".format

np.set_printoptions(formatter={'float_kind':float_formatter})

def print_errors(training_error, test_error):
    print("Training error: ", training_error)
    print("Test error: ", test_error)

def get_accuracy(true, pred):
    for i in range(len(pred)):
        if pred[i] > 0.49:
            pred[i] = 1
        else:
            pred[i] = 0
    accuracy = np.mean(true == pred)
    return accuracy

def print_accuracy(accuracies: list):
    train_accuracy = np.mean(accuracies[:][0])
    test_accuracy = np.mean(accuracies[:][1])
    print("Train Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy)

def kfolds(X, y, modelType: str):
    # Split into 10 folds and shuffle
    kf = KFold(n_splits=10, shuffle=True)

    # Array to store model test/train prediction accuracy over each iteration
    accuracies = []

    # Normal linear model
    model = Normal_linear()
    
    if modelType == "Ridge":
        model = Ridge_linear()

    for (train_index, test_index) in kf.split(X):
        X_train, X_test = X[train_index, :], X[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        
        # Zero mean normalization
        X_train = zero_mean_normalize(X_train)
        X_test = zero_mean_normalize(X_test)

        X_train = add_bias(X_train)

        # Apply normal/ridge equation on training set
        w = []
        if modelType == "Ridge":
            w = model.fit(X_train, y_train, 1.0)
        else:
            w = model.fit(X_train, y_train)
        
        # use weights to predict on training set
        y_train_hat = model.predict(X_train, w)
        
        # use weights to predict on test set
        X_test = add_bias(X_test)
        y_test_hat = model.predict(X_test, w)

        # use threshold to get the accuracy of the predictions
        train_accuracy = get_accuracy(y_train, y_train_hat)
        test_accuracy = get_accuracy(y_test, y_test_hat)

        # append to accuracies array and repeat
        accuracies.append([train_accuracy, test_accuracy])
    
    print_accuracy(accuracies)



def housing_part_a():
    print("Housing Part A)")
    # Extract data from txt files and normalize
    X_training, X_test, y_training, y_test = housing_data_handling()
    
    # reshape y and add bias
    X_training = add_bias(X_training)

    # Init Normal Equation model
    housing_model = Normal_linear()

    # Get w by applying normal equation
    w = housing_model.fit(X_training, y_training)

    # predict on training data with dot product of X_train and w
    y_training_hat = housing_model.predict(X_training, w)

    # Calculate mean sum of squares on training data
    training_error = mean_sum_of_squares(y_training, y_training_hat)

    # Repeat on test dataset
    X_test = add_bias(X_test)
    y_test_hat = housing_model.predict(X_test, w)
    test_error = mean_sum_of_squares(y_test, y_test_hat)

    print_errors(training_error, test_error)

def housing_part_b():
    print("Housing Part B)")

    # Extract data from txt files and normalize
    X_train, X_test, y_train, y_test = housing_data_handling()
    
    # reshape y and add bias
    X_train = add_bias(X_train)

    # init model
    housing_ridge_model = Ridge_linear()

    # get coefficients with alpha value = 1.0
    w = housing_ridge_model.fit(X_train, y_train, 1.0)

    # predict on training data and calculate MSE
    y_train_hat = housing_ridge_model.predict(X_train, w)
    train_error = mean_sum_of_squares(y_train, y_train_hat)

    # Repeat for test data 
    X_test = add_bias(X_test)
    y_test_hat = housing_ridge_model.predict(X_test, w)
    test_error = mean_sum_of_squares(y_test, y_test_hat)

    print_errors(train_error, test_error)
    
def spambase_part_a():
    print("Spambase Part A)")
    X, y = spambase_data_handling()

    # k fold cross validation
    kfolds(X, y, "Normal")

def spambase_part_b():
    print("Spambase Part B)")
    X, y = spambase_data_handling()
    kfolds(X, y, "Ridge")



def main():
    housing_part_a()
    spambase_part_a()
    housing_part_b()
    spambase_part_b()

if __name__ == "__main__":
    main()
