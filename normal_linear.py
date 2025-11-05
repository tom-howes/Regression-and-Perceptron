import numpy as np

class Normal_linear:

    def fit(self, X, y):
        X_transpose = X.T
        X_transpose_X = X_transpose.dot(X)
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        X_transpose_y = X_transpose.dot(y)
        w = X_transpose_X_inv.dot(X_transpose_y)
        return w
    
    def predict(self, X, w):
        return X.dot(w)
