import numpy as np

class Ridge_linear():

    def fit(self, X, y, alpha):
        I = np.eye(X.shape[1])
        I[0, 0] = 0
        X_transpose = X.T
        X_transpose_X = X_transpose.dot(X)
        X_transpose_X_reg = X_transpose_X + alpha*I
        X_transpose_X_reg_inv = np.linalg.inv(X_transpose_X_reg)
        X_transpose_y = X_transpose.dot(y)
        w = X_transpose_X_reg_inv.dot(X_transpose_y)
        return w
    
    def predict(self, X, w):
        return X.dot(w)    