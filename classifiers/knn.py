import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    @staticmethod
    def __euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        predictions = [self._predict(x) for x in np.array(X)]
        return np.array(predictions)

    def _predict(self, x):
        distances = [self.__euclidean_distance(x, x_train) for x_train in self.X_train]

        k_nearest_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
