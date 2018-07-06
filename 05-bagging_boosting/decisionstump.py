import numpy as np
from operator import lt, ge

class DecisionStump:
    """
    A simple decision stump classifier
    """
    def __init__(self, dim=0, value=0, op=lt):
        """
        :param dim: dimension to split in
        :param value: decision value to split on
        :param op: operator to compare (lt = less than)
        """
        self.dim = dim
        self.value = value
        self.op = op

    def update(self, dim = None, value = None, op = None):
        if dim is not None: self.dim = dim
        if value is not None: self.value = value
        if op is not None: self.op = op

    def predict(self, X):
        return np.array([1 if self.op(x, self.value) else -1 for x in X[:, self.dim]])

    def fit(self, X, Y, sample_weights, num_splits=100):
        """
        Fit a decision stump classifier
        :param X: NxD matrix, N observations, d props each
        :param Y: n-dimensional vector, ground truth
        :param num_splits: number of split values to evaluate
        """
        n, d = X.shape
        min_error = np.inf

        for dim in range(d):
            min_dim_err, split_value, op = self._fit_dim( X[:,dim], Y, sample_weights, num_splits)
            if min_dim_err < min_error:
                min_error = min_dim_err
                self.update(dim, split_value, op)

    def _fit_dim( self, X, Y, sample_weights, num_splits):
        """
        Fit a 1D decision stump classifier
        :param X: NxD matrix, n observations with d props each
        :param Y: n-dimensional vector, ground truth
        :param sample_weights: n-dimensional vector, weights of observations
        :param num_splits: number of split values to evaluate
        """

        min_error, best_value, op = np.inf, None, lt
        num_splits = min(num_splits, len(Y)-1)

        # find axis aligned hyperplane to minimize the classification error
        for value in np.linspace(min(X), max(X), num_splits, endpoint=False):
            predictions = [1 if x < value else -1 for x in X]
            idx_err = np.not_equal( predictions, Y )
            Jm = np.dot( sample_weights, idx_err )

            if Jm < min_error:
                min_error, best_value, op = Jm, value, lt

            # consider err for inverted predictions
            Jm = np.dot( sample_weights, ~idx_err)
            if Jm < min_error:
                min_error, best_value, op = Jm, value, ge
        return min_error, best_value, op
