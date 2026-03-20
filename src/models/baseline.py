import numpy as np


class BaselineModel:
    def fit(self, X, y):
        self.last_value = np.mean(y) if y is not None else 0

    def predict(self, X):
        return np.full(len(X), self.last_value)