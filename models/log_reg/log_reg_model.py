import numpy as np
from tqdm import tqdm


class LogisticModel:
    def __init__(
            self,
            lr: float = 0.1,
            max_iter: int = 200,
            reg_lambda: float = 0.1,
            tol=1e-3,
            batch_size=100,
            decay=0.01,
    ) -> None:
        """Initializes Logistic Model with given hyper-parameters
        Args:
            lr (float, optional): Learning rate. Defaults to 0.1.
            max_iter (int, optional): Maximum number of steps/iterations. Defaults to 200.
            reg_lambda (float, optional): Regularizing constant. Equivalent to "C" in sk-learn. Defaults to 0.1.
            tol ([type], optional): Tolerance for minimal gradient update. Defaults to 1e-3.
            batch_size (int, optional): Number of rows in each batch for SGD. Defaults to 100.
            decay (float, optional): Learning rate decay parameter. Defined similarly as in sk-learn. Defaults to 0.01.
        """

        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.lr = lr
        self.batch_size = batch_size
        self.decay = decay

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: np.ndarray = None,
            y_val: np.ndarray = None,
            num_class: int = 2,
    ) -> tuple:
        """Fits the predictor to input matrices X and y. If provided, uses X_val and y_val to return a validation
        accuracies array.
        Args:
            X (np.ndarray): Input Features (n_instances x n_features)
            y (np.ndarray): Labels (n_instance x 1)
            X_val (np.ndarray, optional): Validation Set (n_instances_val x n_features). Defaults to None.
            y_val (np.ndarray, optional): Validation set labels (n_intances_val x 1). Defaults to None.
            num_class (int, optional): Number of unique values in y. Defaults to 3.

        Returns:
            tuple: (losses, training accuracies, validation accuracies). Validation accuracies = [] if X_val and
            y_val not provided.
        """

        with_val = X_val is not None and y_val is not None
        y = get_one_hot(y, num_class)
        self.W = np.random.random(size=(X.shape[1], y.shape[1]))

        losses = []
        train_accs = []
        val_accs = []

        delta_grad = 0
        for step in tqdm(range(self.max_iter), desc="Steps"):
            for start in range(0, X.shape[0], self.batch_size):
                stop = start + self.batch_size
                X_batch = X[start:stop, :]
                y_batch = y[start:stop, :]

                grad = self.gradient(X_batch, y_batch)
                delta_grad = self.decay * delta_grad + self.lr * grad

                if not np.all(abs(delta_grad)) >= self.tol:
                    print(f"Training terminated in iteration {step}")
                    break

                self.W -= delta_grad

            # Update losses
            losses.append(self.loss(self.predict_probs(X), y))

            # Update train accs
            train_accs.append(accuracy(self.predict_labels(X), y))

            if with_val:
                # Update val accs
                val_accs.append(
                    accuracy(self.predict_labels(X_val), y_val)
                )

        print("Train accuracy:  {:.2f}%".format(train_accs[-1] * 100))
        if with_val:
            print("Val accuracy:  {:.2f}%".format(val_accs[-1] * 100))
        return np.array(losses), np.array(train_accs), np.array(val_accs)

    def loss(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy cost function
        Args:
            y_hat (np.ndarray): predictions (one-hot encoded)
            y (np.ndarray): ground-truth labels (one-hot encoded)

        Returns:
            float: Loss as a scalar
        """
        return -np.sum(np.log(y_hat + 1e-15) * y)

    def gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes the gradient of the loss wrt to self.W
        Args:
            X (np.ndarray): Input training features (n_instances x n_features)
            y (np.ndarray): Training labels, one-hot encoded (n_instances x 3)

        Returns:
            np.ndarray: Gradient of the loss. Same dimensions as self.W: n_features x n_classes
        """
        N = X.shape[0]
        errors = y - self.predict_probs(X)
        gd = 1 / N * (X.T @ errors) + self.reg_lambda * self.W
        return gd

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        """Returns the probability distribution of predictions of input features X.
        Args:
            X (np.ndarray): Input features (n_instances x n_features)

        Returns:
            np.ndarray: Softmax probability distribution (n_instances x n_classes)
        """
        Z = -X @ self.W
        return softmax(Z, axis=1)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Returns the predicted labels of input features X
        Args:
            X (np.ndarray): Input features (n_instances x n_features)

        Returns:
            np.ndarray: predicted labels (n_instances x 1)
        """
        return np.argmax(self.predict_probs(X), axis=1)

    def get_params(self) -> dict:
        """Returns the hyperparameters as a dictionary
        """
        params = {
            'max_iter': self.max_iter,
            'reg_lambda': self.reg_lambda,
            'tol': self.tol,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'decay': self.decay
        }
        return params


def softmax(X: np.ndarray, axis: int) -> np.ndarray:
    """Applies softmax normalization to arbitrary matrix X
    Args:
        X (np.ndarray): Input matrix
        axis (int): Axis across which to normalize the softmax

    Returns:
        np.ndarray: Probability normalized x
    """
    X_ = X - X.max(axis=axis, keepdims=True)
    y = np.exp(X_)
    return y / y.sum(axis=axis, keepdims=True)


def get_one_hot(y, num_class):
    """Generates a one-hot encoding of a 1D array
    Args:
        y (np.ndarray): vector to be 1-hot encoded
        num_class (int): Number of class

    Returns:
        np.ndarray: one-hot encoded targets (n_instances x num_class)
    """
    res = np.eye(num_class)[np.array(y).reshape(-1)]
    return res.reshape(list(y.shape) + [num_class])


def accuracy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """Computes the accuracy of a prediction vectors wrt to a vector of ground truths
    Args:
        y_hat (np.ndarray): Predictions
        y (np.ndarray): Ground truths

    Returns:
        float: Accuracy. Returned as a number in [0,1]
    """
    if y.ndim == 2:
        y = np.argmax(y, axis=1)
    return np.sum(y == y_hat) / y.shape[0]
