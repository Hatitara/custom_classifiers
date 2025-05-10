import numpy as np
from collections import Counter, defaultdict
from typing import Callable, Optional, List, Union


class KNNClassifier:
    def __init__(self, k: int = 3, distance_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
                 weighted: bool = False):
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.weighted = weighted
        self.distance_func = distance_func or self._euclidean_distance
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.linalg.norm(x1 - x2)

    @staticmethod
    def _manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sum(np.abs(x1 - x2))

    def fit(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must be the same length")
        self.classes_ = np.unique(self.y_train)

    def predict(self, X: Union[np.ndarray, List]) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier has not been fitted")
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X: Union[np.ndarray, List]) -> List[dict]:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier has not been fitted")
        X = np.array(X)
        return [self._predict_single_proba(x) for x in X]

    def _predict_single(self, x: np.ndarray) -> Union[int, str]:
        distances = np.array([self.distance_func(x, x_train)
                             for x_train in self.X_train])
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]

        if self.weighted:
            label_weights = defaultdict(float)
            for label, dist in zip(k_labels, k_distances):
                weight = 1 / (dist + 1e-16)
                label_weights[label] += weight
            top_label = max(label_weights.items(), key=lambda item: (
                item[1], -self._label_rank(item[0])))[0]
        else:
            label_counts = Counter(k_labels)
            top_count = max(label_counts.values())
            tied_labels = [label for label,
                           count in label_counts.items() if count == top_count]
            top_label = min(tied_labels, key=self._label_rank)

        return top_label

    def _predict_single_proba(self, x: np.ndarray) -> dict:
        distances = np.array([self.distance_func(x, x_train)
                             for x_train in self.X_train])
        k_indices = np.argpartition(distances, self.k)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]

        probs = {label: 0.0 for label in self.classes_}

        if self.weighted:
            weights = np.array([1 / (d + 1e-8) for d in k_distances])
            total_weight = np.sum(weights)
            for label, w in zip(k_labels, weights):
                probs[label] += w
            for label in probs:
                probs[label] /= total_weight
        else:
            for label in k_labels:
                probs[label] += 1
            for label in probs:
                probs[label] /= self.k

        return probs

    def score(self, X: Union[np.ndarray, List], y: Union[np.ndarray, List]) -> float:
        preds = self.predict(X)
        return np.mean(preds == np.array(y))

    def _label_rank(self, label):
        return np.where(self.classes_ == label)[0][0]

    def __repr__(self):
        return f"KNNClassifier(k={self.k}, weighted={self.weighted}, distance_func={self.distance_func.__name__})"
