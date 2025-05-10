'''
Utility functions for Decision Tree Classifier.
'''
from typing import List, Optional, Union
import numpy as np
import pandas as pd

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

def categorical_split(X, y, feature_idx, criterion='gini', min_samples=0):
    """
    Perform an optimal binary split on a categorical feature.

    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features).
    - y: 1D numpy array of class labels (length n_samples).
    - feature_idx: integer index of the feature column to split on.
    - criterion: 'gini' or 'entropy' (impurity measure).
    - min_samples: threshold for grouping rare categories.
        * If >=1: absolute count threshold;
        * If between 0 and 1: fraction of total samples;
        * If <=0: no grouping (default 0).

    Returns:
    - left_idx: array of row indices sent to left child.
    - right_idx: array of row indices sent to right child.
    - X_left, X_right: feature subsets (rows) for left/right.
    - y_left, y_right: label subsets for left/right.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_samples, n_features = X.shape

    # Basic checks
    if n_samples != len(y):
        raise ValueError(f"X and y must have same number of samples, got {n_samples} and {len(y)}")
    if feature_idx < 0 or feature_idx >= n_features:
        raise IndexError(f"Feature index {feature_idx} is out of bounds")
    if criterion not in ('gini', 'entropy'):
        raise ValueError("criterion must be 'gini' or 'entropy'")

    # Extract and (optionally) copy feature column
    col = X[:, feature_idx]
    col2 = col.astype(str)
    col2 = np.where(pd.isnull(col), 'MISSING', col).astype(str)

    # Group rare categories into 'OTHER'
    if min_samples is not None and min_samples > 0:
        if min_samples < 1:
            thresh = min_samples * n_samples
        else:
            thresh = min_samples
        cats, counts = np.unique(col2, return_counts=True)
        rare_cats = cats[counts < thresh]
        if len(rare_cats) > 0:
            for rc in rare_cats:
                col2[col2 == rc] = 'OTHER'

    # Unique categories after grouping
    categories, counts = np.unique(col2, return_counts=True)
    L = len(categories)
    # If <2 categories, no split possible
    if L <= 1:
        left_idx = np.arange(n_samples)
        right_idx = np.array([], dtype=int)
        return left_idx, right_idx, X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    # Map classes to indices
    classes, _ = np.unique(y, return_counts=True)
    K = len(classes)
    class_to_idx = {c:i for i,c in enumerate(classes)}

    # Count matrix: shape (L categories) x (K classes)
    cat_to_idx = {cat:i for i,cat in enumerate(categories)}
    count_matrix = np.zeros((L, K), dtype=int)
    for i in range(n_samples):
        cat = col2[i]
        cl = y[i]
        j = cat_to_idx[cat]
        k = class_to_idx[cl]
        count_matrix[j, k] += 1

    imp_func = gini if criterion == 'gini' else entropy

    best_impurity = np.inf
    best_splt = None

    if K == 2:
        probs = count_matrix[:,1] / count_matrix.sum(axis=1)
        sorted_idx = np.argsort(probs)
        sorted_counts = count_matrix[sorted_idx]
        cum_counts = np.cumsum(sorted_counts, axis=0)
        for i in range(1, L):  # split after i-1 vs rest
            left_counts = cum_counts[i-1]
            right_counts = cum_counts[-1] - left_counts
            imp = ((left_counts.sum() * imp_func(left_counts)) +
                   (right_counts.sum() * imp_func(right_counts))) / n_samples
            if imp < best_impurity:
                best_impurity = imp
                best_splt = (None, i, sorted_idx)

    else:
        for ci in range(K):
            prob_ci = count_matrix[:, ci] / count_matrix.sum(axis=1)
            sorted_idx = np.argsort(prob_ci)
            sorted_counts = count_matrix[sorted_idx]
            cum_counts = np.cumsum(sorted_counts, axis=0)
            for i in range(1, L):
                left_counts = cum_counts[i-1]
                right_counts = cum_counts[-1] - left_counts
                imp = ((left_counts.sum() * imp_func(left_counts)) +
                       (right_counts.sum() * imp_func(right_counts))) / n_samples
                if imp < best_impurity:
                    best_impurity = imp
                    best_splt = (ci, i, sorted_idx)

    if best_splt is None:
        left_idx = np.arange(n_samples)
        right_idx = np.array([], dtype=int)
        return left_idx, right_idx, X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    _, split_pos, sorted_idx = best_splt
    sorted_cats = [categories[j] for j in sorted_idx]
    left_cats = set(sorted_cats[:split_pos])
    mask_left = np.array([val in left_cats for val in col2], dtype=bool)
    left_idx = np.nonzero(mask_left)[0]
    right_idx = np.nonzero(~mask_left)[0]

    X_left, X_right = X[left_idx], X[right_idx]
    y_left, y_right = y[left_idx], y[right_idx]
    return left_idx, right_idx, X_left, y_left, X_right, y_right

def best_split(X: np.ndarray, y: np.ndarray, criterion: str = 'gini', cthreshold: int = 10, min_samples: int = 0) -> tuple[Optional[int], Optional[Union[float, List]], Optional[np.ndarray], Optional[np.ndarray]]:
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
            (left_indices, right_indices, _, y_left, _, y_right) = categorical_split(X, y, feature_index, criterion, min_samples)
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
            values = np.unique(X[:, feature_index])
            thresholds = (values[:-1] + values[1:]) / 2 
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
