"""
Simple Logistic Regression Classifier
This is a simple implementation of a logistic regression classifier using numpy.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve


class LogisticRegression:
    """
    Logistic Regression Classifier
    """
    def __init__(self, lr=0.1, max_iter=10000, tol=1e-4,
                 early_stopping=True, n_iter_no_change=5, verbose=False,
                 batch_size=None, decay_rate=0.0, clip_gradients=False, max_grad_norm=None):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.clip_gradients = clip_gradients
        self.max_grad_norm = max_grad_norm

        self.w = None
        self.b = None
        self.loss_history = []

    def _sigmoid(self, z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def _log_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def _get_batches(self, X, y, batch_size):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield X[batch_indices], y.iloc[batch_indices]

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        no_improve_count = 0
        prev_loss = float('inf')

        for i in range(self.max_iter):
            if self.batch_size:
                epoch_loss = []

                for batch_X, batch_y in self._get_batches(X, y, self.batch_size):
                    linear_model = batch_X.dot(self.w) + self.b
                    y_pred = self._sigmoid(linear_model)

                    error = y_pred - batch_y
                    grad_w = (1 / batch_X.shape[0]) * batch_X.T.dot(error)
                    grad_b = (1 / batch_X.shape[0]) * np.sum(error)

                    if self.clip_gradients:
                        grad_norm = np.linalg.norm(grad_w)
                        if self.max_grad_norm and grad_norm > self.max_grad_norm:
                            grad_w = grad_w * (self.max_grad_norm / grad_norm)

                    self.w -= self.lr * grad_w
                    self.b -= self.lr * grad_b

                    batch_loss = self._log_loss(batch_y, y_pred)
                    epoch_loss.append(batch_loss)

                loss = np.mean(epoch_loss)
            else:
                linear_model = X.dot(self.w) + self.b
                y_pred = self._sigmoid(linear_model)

                error = y_pred - y
                grad_w = (1 / n_samples) * X.T.dot(error)
                grad_b = (1 / n_samples) * np.sum(error)

                if self.clip_gradients:
                    grad_norm = np.linalg.norm(grad_w)
                    if self.max_grad_norm and grad_norm > self.max_grad_norm:
                        grad_w = grad_w * (self.max_grad_norm / grad_norm)

                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

                loss = self._log_loss(y, y_pred)

            self.loss_history.append(loss)

            if self.verbose and i % 100 == 0:
                print(f"Iter {i}: LogLoss = {loss:.4f}")

            if self.early_stopping:
                if abs(prev_loss - loss) < self.tol:
                    no_improve_count += 1
                    if no_improve_count >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopped at iteration {i}")
                        break
                else:
                    no_improve_count = 0
                prev_loss = loss

            if self.decay_rate > 0:
                self.lr = self.lr / (1 + self.decay_rate * i)

    def predict_proba(self, X):
        """
        Predict the probabilities of the positive class.
        """
        linear_model = X.dot(self.w) + self.b
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Predict the class labels for the input data.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X, y):
        """
        Calculate the accuracy of the model.
        """
        preds = self.predict(X)
        return np.mean(preds == y)

    def plot_loss_history(self):
        """
        Plot the loss history of the model.
        """
        if not self.loss_history:
            print("No loss history found. Train the model first.")
            return

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=range(len(self.loss_history)), y=self.loss_history)
        plt.title("Log Loss over Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Log Loss")
        plt.tight_layout()
        plt.show()

    def get_params(self):
        """
        Get the parameters of the model.
        """
        return {
            'lr': self.lr,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'early_stopping': self.early_stopping,
            'n_iter_no_change': self.n_iter_no_change,
            'verbose': self.verbose,
            'batch_size': self.batch_size,
            'decay_rate': self.decay_rate,
            'clip_gradients': self.clip_gradients,
            'max_grad_norm': self.max_grad_norm
        }

    def set_params(self, **params):
        """
        Set the parameters of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)

    def evaluate(self, X, y, threshold=0.5):
        """
        Evaluate the model using confusion matrix and classification report.
        """
        preds = self.predict(X, threshold)
        print("Confusion Matrix:")
        print(confusion_matrix(y, preds))
        print("\nClassification Report:")
        print(classification_report(y, preds))

    def reset(self):
        """
        Reset the model parameters.
        """
        self.w = None
        self.b = None
        self.loss_history = []


def best_threshold_from_roc(y_true, y_proba):
    """
    Calculate the best threshold for a binary classification problem using ROC curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    return thresholds[best_idx]
