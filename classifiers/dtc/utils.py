'''
Utility functions for Decision Tree Classifier.
'''
from itertools import combinations
from typing import List, Tuple, Optional, Union
import numpy as np

def is_categorical(X: np.ndarray, feature_index: int, threshold: int = 10) -> bool:
    """
    Determine if a feature is likely categorical.

    Args:
        X (np.ndarray): The feature matrix.
        feature_index (int): The index of the feature to check.
        threshold (int, optional): The maximum number of unique values for a feature to be considered categorical. Defaults to 10.

    Returns:
        bool: True if the feature is likely categorical, False otherwise.
    """
    feature_column = X[:, feature_index]
    if not np.issubdtype(feature_column.dtype, np.number):
        return True
    return len(np.unique(feature_column)) <= threshold

def gini(labels):
    '''
    Calculate the Gini impurity for a given set of labels.

    Gini impurity is a measure of how often a randomly chosen element 
    from the set would be incorrectly labeled if it was randomly labeled 
    according to the distribution of labels in the set.

    Parameters:
        labels (array-like): A list or array of class labels.

    Returns:
        float: The Gini impurity, a value between 0 and 1, where 0 indicates 
               perfect purity (all labels are the same) and values closer to 1 
               indicate higher impurity.
    '''
    if len(labels) == 0:
        return 0
    class_counts = np.unique(labels, return_counts=True)[1]
    probabilities = class_counts / len(labels)
    gini_impurity = 1 - np.sum(probabilities**2)
    return gini_impurity

def entropy(labels):
    '''
    Calculate the entropy of a given set of labels.

    Entropy is a measure of the impurity or randomness in a dataset. It is 
    calculated using the formula:
        H = -Σ(p * log2(p))
    where p is the probability of each unique label in the dataset.

    Parameters:
        labels (array-like): A list or array of labels for which the entropy 
                             is to be calculated.

    Returns:
        float: The entropy value. Returns 0 if the input list is empty.

    Notes:
        - A small constant (1e-16) is added to probabilities to avoid 
          numerical issues with log2(0).
    '''
    if len(labels) == 0:
        return 0
    probabilities = np.unique(labels, return_counts=True)[1] / len(labels)
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-16))
    return entropy_value

def split(X, y, feature_index, threshold):
    '''
    Split the dataset into two subsets based on a feature and a threshold.

    Parameters:
        X (numpy.ndarray): The feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): The target vector of shape (n_samples,).
        feature_index (int): The index of the feature to split on.
        threshold (float): The threshold value to split the feature.

    Returns:
        tuple: A tuple containing:
    - left_indices (numpy.ndarray):
        Indices of samples in the left subset (feature value <= threshold).
    - right_indices (numpy.ndarray):
        Indices of samples in the right subset (feature value > threshold).
    - X_left (numpy.ndarray):
        Feature matrix for the left subset.
    - y_left (numpy.ndarray):
        Target vector for the left subset.
    - X_right (numpy.ndarray):
        Feature matrix for the right subset.
    - y_right (numpy.ndarray):
        Target vector for the right subset.
    '''
    left_indices = np.where(X[:, feature_index] <= threshold)[0]
    right_indices = np.where(X[:, feature_index] > threshold)[0]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    return left_indices, right_indices, X_left, y_left, X_right, y_right

def categorical_split(X: np.ndarray, y: np.ndarray, feature_index: int, criterion: str = 'gini') -> Tuple[Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Split the dataset based on a categorical feature to maximize information gain.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target labels.
        feature_index (int): The index of the categorical feature to split on.
        criterion (str, optional): The impurity criterion to use ('gini' or 'entropy'). Defaults to 'gini'.

    Returns:
        Tuple[Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
            Indices of left samples, indices of right samples, left features, left labels, right features, right labels.
            Returns None for all if no beneficial split is found.
    """
    best_gain = -1.0
    best_left_categories = None
    left_indices_best = None
    right_indices_best = None

    n_samples = X.shape[0]
    unique_categories = np.unique(X[:, feature_index])
    if len(unique_categories) <= 1:
        return None, None, None, None, None, None

    parent_impurity = gini(y) if criterion == 'gini' else entropy(y)

    for r in range(1, len(unique_categories)):
        for left_categories in combinations(unique_categories, r):
            left_categories_set = set(left_categories)
            right_categories_set = set(unique_categories) - left_categories_set

            left_indices = np.where(np.isin(X[:, feature_index], list(left_categories_set)))[0]
            right_indices = np.where(np.isin(X[:, feature_index], list(right_categories_set)))[0]

            if len(left_indices) > 0 and len(right_indices) > 0:
                left_y = y[left_indices]
                right_y = y[right_indices]
                n_left = len(left_indices)
                n_right = len(right_indices)

                child_impurity = (n_left / n_samples) * (gini(left_y) if criterion == 'gini' else entropy(left_y)) + \
                                 (n_right / n_samples) * (gini(right_y) if criterion == 'gini' else entropy(right_y))
                information_gain = parent_impurity - child_impurity

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_left_categories = list(left_categories_set)
                    left_indices_best = left_indices
                    right_indices_best = right_indices

    if best_left_categories is not None:
        left_X = X[left_indices_best]
        left_y = y[left_indices_best]
        right_X = X[right_indices_best]
        right_y = y[right_indices_best]
        return left_indices_best, right_indices_best, left_X, left_y, right_X, right_y
    else:
        return None, None, None, None, None, None

def best_split(X: np.ndarray, y: np.ndarray, criterion: str = 'gini', cthreshold: int = 10) -> tuple[Optional[int], Optional[Union[float, List]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find the best split by iterating over all features.

    Handles both continuous and categorical features.

    Args:
        X (np.ndarray): The feature matrix.
        y (np.ndarray): The target labels.
        criterion (str, optional): The impurity criterion to use ('gini' or 'entropy'). Defaults to 'gini'.

    Returns:
        Tuple[Optional[int], Optional[Union[float, List]], Optional[np.ndarray], Optional[np.ndarray]]:
            Index of the best feature, the best threshold (float for continuous, list of categories for categorical),
            indices of left split, indices of right split.
    """
    best_gain = -1.0
    split_feature = None
    split_threshold = None
    left_indices_best = None
    right_indices_best = None

    n_samples, n_features = X.shape
    if n_samples <= 1:
        return split_feature, split_threshold, left_indices_best, right_indices_best

    parent_impurity = gini(y) if criterion == 'gini' else entropy(y)

    for feature_index in range(n_features):
        if is_categorical(X, feature_index, cthreshold):
            (left_indices, right_indices, _, y_left, _, y_right) = categorical_split(X, y, feature_index, criterion)
            if left_indices is not None and right_indices is not None:
                n_left, n_right = len(left_indices), len(right_indices)
                child_impurity = (n_left / n_samples) * (gini(y_left) if criterion == 'gini' else entropy(y_left)) + \
                                 (n_right / n_samples) * (gini(y_right) if criterion == 'gini' else entropy(y_right))
                information_gain = parent_impurity - child_impurity

                if information_gain > best_gain:
                    best_gain = information_gain
                    split_feature = feature_index
                    split_threshold = np.unique(X[left_indices, feature_index]).tolist()
                    left_indices_best = left_indices
                    right_indices_best = right_indices
        else:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices, right_indices, _, y_left, _, y_right = split(X, y, feature_index, threshold)
                if len(y_left) > 0 and len(y_right) > 0:
                    n_left, n_right = len(y_left), len(y_right)
                    child_impurity = (n_left / n_samples) * (gini(y_left) if criterion == 'gini' else entropy(y_left)) + \
                                     (n_right / n_samples) * (gini(y_right) if criterion == 'gini' else entropy(y_right))
                    information_gain = parent_impurity - child_impurity
                    if information_gain > best_gain:
                        best_gain = information_gain
                        split_feature = feature_index
                        split_threshold = threshold
                        left_indices_best = left_indices
                        right_indices_best = right_indices

    return split_feature, split_threshold, left_indices_best, right_indices_best
