import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt

from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted

from mlcv.templates.base import Solution


def _compute_polynomials(X, n_degrees):
    """Computes polynomial basis functions (for scalars).

    Parameters
    ----------
    X : array, shape (n_samples,)
        Input data.

    n_degrees : int
        n_degrees + 1 (the 0-th order) polynomials will be computed.

    Returns
    -------
    Phi : array, shape (n_samples, n_degrees + 1)
        Each row is a polynomial basis function of one sample.

    """
    X = X.ravel()
    n_samples = len(X)

    # Construct the (polynomial) basis functions
    Phi = np.zeros((n_samples, n_degrees + 1))
    for n_degree in range(n_degrees + 1):
        Phi[:, n_degree] = np.power(X, n_degree)

    return Phi


class PolynomialRegression(Solution):
    """Regression with polynomials as basis functions.

    Parameters
    ----------
    n_degrees : int, optional (default=1)
        The degrees of the polynomial.

    Attributes
    ----------

    weights_ : array, shape (n_degrees + 1, n_features)
        The weights of the basis functions for each feature.

    """

    def __init__(self, n_degrees=1):
        super().__init__()
        self.n_degrees = n_degrees

    def _validate_training_inputs(self, X, y=None):
        """Validate the parameters passed in __init__ and make sure they are
        consistent with X and y.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            There should be at least k + 1 samples.

        y : array, shape (n_samples, ?)
            There should be at least 2 distinct classes

        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """

        if y is not None:
            check_consistent_length(X, y)



        # Check the number of neighbors is a positive integer
        if self.n_degrees < 1:
            raise ValueError("Number of degrees must be at least 1.")

        X = check_array(X)
        y = check_array(y)

        if X.shape[1] > 1:
            raise NotImplementedError("This implementation assumes each "
                                      "sample in `X` has a single feature.")

        return X, y

    def fit(self, X, y=None):
        """

        Parameters
        ----------
        X : array, shape (n_samples_train,)
            Training inputs.

        y : array, shape (n_samples_train, n_features)
            Corresponding training targets.

        Returns
        -------
        solution : PolynomialRegression
            A trained model.

        """

        X, y = self._validate_training_inputs(X, y)

        # Here we assume each sample in X has one feature.
        Phi = _compute_polynomials(X, self.n_degrees)

        # Compute the pseudo-inverse of Phi using SVD with scipy.linalg.pinv2
        Phi_pinv = scipy.linalg.pinv2(Phi)  # shape (n_degrees+1, n_samples)

        # Compute the weights : w = (Phi^T Phi)^{-1} Phi^T t
        self.weights_ = np.dot(Phi_pinv, y) # shape (n_degrees+1, n_features)

        return self

    def _validate_testing_inputs(self, X):
        """Make sure the testing inputs are compatible.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The number of features should be the same as in the training set.

        Raises
        ------
        ValueError : If the inputs or the parameters do not match the expected
        format or their values are not compatible.

        """

        # Make sure the model has been trained first
        check_is_fitted(self, 'weights_')

        # Make sure the testing inputs are in a valid format
        X = check_array(X)
        return X

    def predict(self, X):
        """

        Parameters
        ----------
        X : array, shape (n_samples_test, n_features)
            Testing inputs.

        Returns
        -------
        y : array, shape(n_samples_test,)
            A prediction for each testing input.

        """

        X = self._validate_testing_inputs(X)

        Phi = _compute_polynomials(X, self.n_degrees)

        y_pred = np.dot(Phi, self.weights_)

        return y_pred

    def score(self, y_pred, y_true):
        """Classification accuracy is the ratio of correct predictions.

        Parameters
        ----------
        y_pred : array, shape (n_samples, )
            Predicted labels.

        y_true : array, shape (n_samples, )
            Groundtrith labels.

        Returns
        -------
        score : The true positives rate.

        """
        check_consistent_length(y_pred, y_true)

        sse = np.sum(np.square(y_true - y_pred))

        return sse / len(y_true)

    def visualize(self, X, proba=None, **kwargs):
        pass


def load_quadrocopter_data():
    X = np.array([[2, 0, 1],
                  [1.08, 1.68, 2.38],
                  [-0.83, 1.82, 2.49],
                  [-1.97, 0.28, 2.15],
                  [-1.31, -1.51, 2.59],
                  [0.57, -1.91, 4.32]])

    return X


def plot_trajectory(data):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(xs=data[:, 0], ys=data[:, 1], zs=data[:, 2])
    ax.scatter(data[0, 0], data[0, 1], data[0, 2], c='r', marker='*')


if __name__ == '__main__':

    y = load_quadrocopter_data()
    X = np.arange(0, len(y), step=1)[:, None]

    # Question a)
    # plot_trajectory(y)
    # plt.show()

    # Question b) Constant speed (polynomial of degree 1)
    # model: x(t) = w0 + w1*t

    # Question c) Constant acceleration (polynomial of degree 2)
    # model: x(t) = w0 + w1*t + w2*(t**2)

    for n_degrees in (1, 2):
        print('\nFitting polynomial regression model with polynomials of '
              'degree {}.'.format(n_degrees))
        model = PolynomialRegression(n_degrees=n_degrees)
        model.fit(X, y)

        if n_degrees == 1:
            v = model.weights_[1, :]
            v_norm = np.sqrt(np.dot(v, v))
            print('Speed (per axis): {}'.format(v))
            print('Speed magnitude : {}'.format(v_norm))

        elif n_degrees == 2:
            a = model.weights_[2, :]
            a_norm = np.sqrt(np.dot(a, a))
            print('Acceleration (per axis): {}'.format(a))
            print('Acceleration magnitude : {}'.format(a_norm))

        y_pred = model.predict(X)
        residuals = np.sqrt(np.sum(np.square(y - y_pred), axis=0))
        print('Residual errors (per axis): {}'.format(residuals))


    # Question d) Position in the next second?
    x_new = X[-1] + 1
    position_ml = model.predict([x_new])
    print('Most likely position at t={} is {}.'.format(x_new, position_ml))