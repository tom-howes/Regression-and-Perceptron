# Regression and Perceptron
A machine learning homework project implementing linear regression with regularization and perceptron classifiers from scratch using NumPy, along with scikit-learn comparisons.
## Overview
This project explores fundamental machine learning algorithms through hands-on implementation:

L1 and L2 Regularized Linear Regression implemented using normal equations
Perceptron classifier built from scratch in NumPy
Comparative analysis with scikit-learn implementations
Applications to real-world datasets (housing prices and spam detection)

## Datasets
### Housing Dataset
Used for regression tasks to predict housing prices based on various features such as location, size, and amenities.
### Spambase Dataset
Used for classification tasks to distinguish spam emails from legitimate ones based on word frequencies and character patterns.
## Implementation Highlights
### Linear Regression with Regularization
The project implements both L1 (Lasso) and L2 (Ridge) regularization from scratch
### L2 Regularization (Ridge):

Adds a penalty term proportional to the square of coefficients
Helps prevent overfitting by shrinking coefficient values
Implemented using closed-form normal equations

### L1 Regularization (Lasso):

Adds a penalty term proportional to the absolute value of coefficients
Performs feature selection by driving some coefficients to zero
Useful for high-dimensional datasets with many irrelevant features

### Perceptron Classifier
A binary linear classifier implemented from scratch:

Iteratively updates weights based on misclassified examples
Simple yet effective algorithm for linearly separable data
Demonstrates fundamental concepts in neural networks
