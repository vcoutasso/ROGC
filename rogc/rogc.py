"""
This module implements the Robust Optimal Graph Clustering algorithm proposed in https://doi.org/10.1016/j.neucom.2019.07.102
"""

from sklearn.base import BaseEstimator
from sklearn.utils import check_array
import sklearn
import numpy as np
import spams

class ROGC(BaseEstimator):
    """
    Robust Optimal Graph Clustering Algorithm[1].

    Parameters
    ----------
    alpha : float
        Alpha parameter
    beta : float
        Beta parameter
    gamma : float
        Gamma parameter
    n_clusters : int
        Number of clusters

    Attributes
    ----------
    W_ : array-like of shape (n, n)
        Weight matrix (similarity matrix)
    S_ : array-like of shape (m, n)
        Coefficient matrix
    B_ : array-like of shape (d, m)
        Basis matrix
    F_ : array-like of shape (n, c)
        Cluster indicator matrix
    converged_ : bool
        True if convergence was reached in `fit()`, False otherwise

    References
    ----------
        [1] Wang, F., Zhu, L., Liang, C., Li, J., Chang, X., & Lu, K. (2020). Robust optimal graph clustering. Neurocomputing, 378, 153-165.
    """

    def __init__(self, alpha=1, beta=1, gamma=1, n_clusters=3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_clusters = n_clusters


    def _initialize(self, X, B):
        """
        Initializes W by the optimal solution to Eq. (4) in [1]

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset
        """
        n_samples = X.shape[1]

        # TODO: Implement solution to equation 4
        self.W_ = np.empty((n_samples, n_samples))

        if B is None:
            self.B_ = sklearn.decomposition.DictionaryLearning().fit(X.T).components_.T
        else:
            self.B_ = B.copy()


    def _check_params(self):
        # alpha regularization parameter
        if self.alpha < 0:
            raise ValueError(
                f"alpha should be >= 0, got {self.alpha} instead."
            )

        # beta regularization parameter
        if self.beta < 0:
            raise ValueError(
                f"beta should be >= 0, got {self.beta} instead."
            )

        # beta regularization parameter
        if self.gamma < 0:
            raise ValueError(
                f"gamma should be >= 0, got {self.gamma} instead."
            )

        # n_clusters
        if self.n_clusters < 0:
            raise ValueError(
                f"n_clusters should be > 0, got {self.n_clusters} instead."
            )


    def fit(self, X, y=None, B=None):
        """
        Fit the model to X with dictionary matrix B.

        Parameters
        ----------
        X : array-like of shape (d, n)
            Training data
        y : ignored
            Not used, here to comply with Scikit-Learn's API convention
        B : array-like of shape (d, m)
            Initial basis matrix (dictionary of atoms)
        """
        self.fit_predict(X, None, B)
        return self


    def fit_predict(self, X, y=None, B=None):
        """
        Fit the model to X with dictionary matrix B and predict labels for X.

        Parameters
        ----------
        X : array-like of shape (d, n)
            Training data
        B : array-like of shape (d, m)
            Initial basis matrix (dictionary of atoms)
        y : ignored
            Not used, here to comply with Scikit-Learn's API convention
        """
        self._check_params()

        X = check_array(X)

        if not B is None:
            B = check_array(B)

        # Initialize W (and B if not provided)
        self._initialize(X, B)

        m = self.B_.shape[1]

        self.converged_ = False

        while not self.converged_:

            # REVIEW: The constraint on eq. 1 suggests the equation is minimized with respect to both B and S. However, according to Algorithm 1, seems like we should learn just S by solving it.
            # The paper suggests computing the sum of the vector-wise norm of S. spams.lasso computes the norm of S.

            # Solve equation 1 for S
            self.S_ = spams.lasso(np.asfortranarray(X), np.asfortranarray(self.B_), mode=2, lambda1=self.beta, lambda2=0)
            print('sparsity:', np.mean(self.S_ == 0))

            # Reconstruction error
            X_hat = self.B_ @ self.S_
            print('error:', np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1)))

            # Solve equation 18

            # Solve equation 19

            # Solve equation 22

            # Verify convergence
            self.converged_ = True

        self.labels_ = np.random.randint(self.n_clusters, size=X.shape[1])

        # Return predictions
        return self.labels_
