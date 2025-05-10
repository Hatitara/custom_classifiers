'''
Python implementation of a Decision Tree Classifier.
'''
import pandas as pd
from sklearn.metrics import classification_report
from typing import Optional, Union
import numpy as np

from .utils import *

class Node:
    '''
    Represents a node in a decision tree.

    Attributes:
        feature (Optional[int]): The index of the feature used for splitting at this node. 
                                  None if the node is a leaf.
        threshold (Optional[float]): The threshold value for the feature at this node. 
                                      None if the node is a leaf.
        left (Optional['Node']): The left child node. None if the node is a leaf.
        right (Optional['Node']): The right child node. None if the node is a leaf.
        value (Optional[int]): The value or class label at this node if it is a leaf. 
                                None if the node is not a leaf.
    '''
    def __init__(self, feature: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None,
                 value: Optional[int] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    '''
    DecisionTreeClassifier is a custom implementation of a decision tree classifier.
    This class provides functionality to fit a decision tree to a dataset, make predictions, 
    and evaluate performance. The tree is built recursively by splitting the dataset based 
    on the best feature and threshold, using a specified criterion (e.g., 'gini').
    '''
    def __init__(self, min_samples_split: int = 2, max_depth: Optional[int] = None,
                 criterion: str = 'gini', categorical_threshold: int = 10,
                 min_impurity_decrease: float = 0, min_in_category: int = 0):
        """
        Initialize the Decision Tree Classifier.

        Args:
            min_samples_split (int, optional): Minimum samples to split. Defaults to 2.
            max_depth (Optional[int], optional): Maximum depth of the tree. Defaults to None.
            criterion (str, optional): Impurity measure ('gini' or 'entropy'). Defaults to 'gini'.
            categorical_threshold (int, optional): Threshold for categorical features. Defaults to 10.
            min_impurity_decrease (float, optional): Minimum impurity decrease to split. Defaults to 0.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.categorical_threshold = categorical_threshold
        self.min_impurity_decrease = min_impurity_decrease
        self.min_in_category = min_in_category
        self.root: Optional[Node] = None
        self.target_names: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, target_names: Optional[np.ndarray] = None):
        """
        Fit the Decision Tree Classifier to the training data.
        """
        self.target_names = target_names
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively grow the decision tree.
        """
        n_samples, _ = X.shape
        n_labels = len(np.unique(y))

        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        best_feature, best_threshold, left_indices, right_indices = best_split(X, y, self.criterion, self.categorical_threshold,
                                                                                 self.min_in_category)

        if best_feature is None:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)

        parent_impurity = gini(y) if self.criterion == 'gini' else entropy(y)
        left_y = y[left_indices]
        right_y = y[right_indices]
        n_left, n_right = len(left_y), len(right_y)
        n_total = n_left + n_right

        if n_total > 0:
            child_impurity = (n_left / n_total) * (gini(left_y) if self.criterion == 'gini' else entropy(left_y)) + \
                             (n_right / n_total) * (gini(right_y) if self.criterion == 'gini' else entropy(right_y))
            information_gain = parent_impurity - child_impurity

            if information_gain > self.min_impurity_decrease:
                left_X, left_y = X[left_indices], y[left_indices]
                right_X, right_y = X[right_indices], y[right_indices]
                left_node = self._grow_tree(left_X, left_y, depth + 1)
                right_node = self._grow_tree(right_X, right_y, depth + 1)
                return Node(feature=best_feature, threshold=best_threshold, left=left_node, right=right_node)

        leaf_value = np.argmax(np.bincount(y))
        return Node(value=leaf_value)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Predict the class labels for the given input data.

        Args:
            X (np.ndarray): A 2D numpy array where each row represents a sample 
                            and each column represents a feature.

        Returns:
            np.ndarray: A 1D numpy array containing the predicted class labels 
                        for each sample in the input data.
        '''
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Optional[Node]) -> Union[int, None]:
        '''
        Recursively traverses a decision tree to make a prediction for a given input.
        Args:
            x (np.ndarray): The input data point as a NumPy array.
            node (Optional[Node]): The current node in the decision tree.
        Returns:
            Union[int, None]: The predicted value if a leaf node is reached, or None if the node is invalid.
        '''
        if node is None:
            return None
        if node.value is not None:
            return node.value

        if node.feature is not None and node.threshold is not None:
            feature_value = x[node.feature]
            if isinstance(node.threshold, list):
                if feature_value in node.threshold:
                    return self._traverse_tree(x, node.left)
                return self._traverse_tree(x, node.right)
            if feature_value <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        return None

    def summary(self, X_test: np.ndarray, y_test: np.ndarray, target_names: Optional[np.ndarray] = None) -> pd.DataFrame:
        '''
        Generate a summary report for the decision tree classifier's performance on test data.

        Args:
            X_test (np.ndarray): The feature matrix for the test dataset.
            y_test (np.ndarray): The true labels for the test dataset.
            target_names (Optional[np.ndarray]): An optional array of target class names. If provided, 
                it will be used to label the classes in the classification report.

        Returns:
            pd.DataFrame: A DataFrame containing the classification report, including precision, recall, 
            F1-score, and support for each class.

        Raises:
            ValueError: If the model has not been fitted yet.
        '''
        if self.root is None:
            raise ValueError("The model has not been fitted yet.")
        predictions = np.array([self._traverse_tree(x, self.root) for x in X_test])
        return pd.DataFrame(classification_report(y_test, predictions, target_names=target_names, output_dict=True))

    def inspect_tree(self, node: Optional[Node] = None, depth: int = 0):
        """Recursively print the structure of the decision tree, including feature type."""
        if node is None:
            node = self.root

        indent = '  ' * depth
        if node.value is not None:
            print(f"{indent}Leaf: class {node.value}")
        else:
            threshold_str = f"<= {node.threshold:.2f}" if not isinstance(node.threshold, list) else f"in {node.threshold}"
            feature_type = "Categorical" if isinstance(node.threshold, list) else "Continuous"
            print(f"{indent}Feature {node.feature} ({feature_type}) {threshold_str}")
            if node.left:
                print(f"{indent}Left:")
                self.inspect_tree(node.left, depth + 1)
            if node.right:
                print(f"{indent}Right:")
                self.inspect_tree(node.right, depth + 1)
