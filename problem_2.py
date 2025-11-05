from data_handling import housing_data_handling, add_bias, min_max_normalize, fetch_20NG_data_handling
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import sys

# np.set_printoptions(threshold=inf)
def plot_graph(x_axis, y_axis, dataset: str):
    plt.plot(x_axis, y_axis)
    plt.title('L1 Regression Performance')
    plt.xlabel('Lambda Value')
    plt.ylabel(f'{dataset} Performance')
    plt.show()
    
def part_a():
    # Extract data from txt files and normalize
    X_train, X_test, y_train, y_test = housing_data_handling()
    
    # Add bias to X and reshape y
    X_train = add_bias(X_train)
    X_test = add_bias(X_test)

    # Create an array of lambda values ranging 0-1 in 0.01 increments
    alphas = np.arange(0, 1, 0.01, dtype=float)

    # Arrays to store results
    x_axis = []
    y_axis_train = []
    y_axis_test = []
    # init model
    for i in alphas:
        model = Lasso(alpha=i)

        # Get coefficients with given alpha value = 1.0
        model.fit(X_train, y_train)

        # Predict on training set and test set
        y_train_hat = model.predict(X_train)
        y_test_hat = model.predict(X_test)
        train_mse = mean_squared_error(y_train, y_train_hat)
        test_mse = mean_squared_error(y_test, y_test_hat)
        x_axis.append(i)
        y_axis_test.append(test_mse)
        y_axis_train.append(train_mse)
    
    plot_graph(x_axis, y_axis_train, 'Train')
    plot_graph(x_axis, y_axis_test, 'Test')

def print_class_accuracy(classes, y, y_pred, model: str, dataset: str):
        print(model, dataset, f"Accuracy:\n_____")
        total_accuracy = []
        for index, class_name in classes.items():
            class_mask = (y == index)
            class_prediction = y_pred[class_mask]
            class_accuracy = accuracy_score(y[class_mask], class_prediction)
            print("Class:\t", class_name, "\nAccuracy: ", f"{class_accuracy:.3f}")
            total_accuracy.append(class_accuracy)
        print(model, dataset, f"Total Accuracy:\n: ", np.mean(total_accuracy), "\n")

def part_b():
    categories = ["comp.graphics", "misc.forsale", "rec.sport.baseball"
                  , "sci.space", "talk.politics.guns", "talk.politics.mideast"
                  , "sci.med", "soc.religion.christian"]
    
    X_train, X_test, y_train, y_test, classes = fetch_20NG_data_handling(categories)

    vectorizer = TfidfVectorizer(stop_words='english',
                                 token_pattern=r'\b[a-zA-Z]{3,}\b', # Only alpha words with 3+ characters
                                 lowercase=True,
                                 strip_accents='ascii'
    )
                                
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    multiclass_classifier = OneVsRestClassifier(LogisticRegression(penalty='l1', C=1.0, solver='liblinear', max_iter=1000))
    multiclass_classifier.fit(X_train_vec, y_train)

    y_pred = multiclass_classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    feature_names = vectorizer.get_feature_names_out()
    # Extract coefficients from each binary classifier

    n_classes = len(np.unique(y_train))
    n_features = len(feature_names)
    
    # Init matrix to store coefs
    coef_matrix = np.zeros((n_classes, n_features))

    # Obtain all coefficients and add to matrix
    for i, estimator in enumerate(multiclass_classifier.estimators_):
        coef_matrix[i] = estimator.coef_[0]

    # Compute average absolute coefficients for all classes
    avg_abs_coefs = np.mean(np.abs(coef_matrix), axis=0)

    # Sort for the top 200 feature indices
    top_200_indices = np.argsort(avg_abs_coefs)[-200:][::-1]

    # Get feature (word) names
    top_200_features = feature_names[top_200_indices]

    # Get top coefficients
    top_200_avg_coefs = avg_abs_coefs[top_200_indices]

    # Reduce train and test sets to top 200 features
    X_train_top200 = X_train_vec[:, top_200_indices]
    X_test_top200 = X_test_vec[:, top_200_indices]

    # Retrain with l1-regularization and l2_regularization - May need to experiment with C values if I have time
    top200_classifier_l1 = OneVsRestClassifier(LogisticRegression(penalty='l1', C=30.0, solver='liblinear', max_iter=1000))
    top200_classifier_l2 = OneVsRestClassifier(LogisticRegression(penalty='l2', C=50.0, solver='liblinear', max_iter=1000))

    # Preict for l1 and l2 and print accuracies per class
    top200_classifier_l1.fit(X_train_top200, y_train)
    y_train_pred_l1 = top200_classifier_l1.predict(X_train_top200)
    y_test_pred_l1 = top200_classifier_l1.predict(X_test_top200)

    print_class_accuracy(classes, y_train, y_train_pred_l1, 'L1-regularized', 'Train')
    print_class_accuracy(classes, y_test, y_test_pred_l1, 'L1-regularized', 'Test')

    top200_classifier_l2.fit(X_train_top200, y_train)
    y_train_pred_l2 = top200_classifier_l2.predict(X_train_top200)
    y_test_pred_l2 = top200_classifier_l2.predict(X_test_top200)

    print_class_accuracy(classes, y_train, y_train_pred_l2, 'L2-regularized', 'Train')
    print_class_accuracy(classes, y_test, y_test_pred_l2, 'L2-regularized', 'Test')
     

def main():
    part_a()
    part_b()
if __name__ == "__main__":
    main()
    