from sklearn.base import BaseEstimator
import numpy as np
import spams

class ROGC(BaseEstimator):
    """
    Robust Optimal Graph Clustering Algorithm[1].

    Parameters
    ----------
    B : array-like of shape (d, m)
        Initial basis matrix (dictionary of atoms)
    alpha : float
        Alpha parameter
    beta : float
        Beta parameter
    gamma : float
        Gamma parameter
    m : int
        Number of basis vectors
    c : int
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

    def __init__(self, B, alpha, beta, gamma, m, c):
        self.B = B
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m = m
        self.c = c


    def _initialize(self, n_samples):
        """
        Initializes W by the optimal solution to Eq. (4) in [1]

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset
        """
        # TODO: Implement solution to equation 4
        self.W_ = np.empty((n_samples, n_samples))


    def fit(self, X):
        """
        Fit the model to X with dictionary matrix B.

        Parameters
        ----------
        X : array-like of shape (d, n)
            Training data
        """
        self.fit_predict(X)
        return self


    def fit_predict(self, X):
        """
        Fit the model to X with dictionary matrix B and predict labels for X.

        Parameters
        ----------
        X : array-like of shape (d, n)
            Training data
        """
        n_samples = X.shape[1]
        self._initialize(n_samples)

        # We preserve the initial dictionary on self.B and copy it to self.B_, where it will be updated in each step
        self.B_ = self.B.copy()
        self.converged_ = False

        while not self.converged_:

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

        # Return predictions
        return None
