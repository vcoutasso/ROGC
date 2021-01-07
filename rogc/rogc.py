import numpy as np

class ROGC:
    """
    Robust Optimal Graph Clustering Algorithm[1].

    Parameters
    ----------
    alpha : float
        Alpha
    beta : float
        Beta
    gamma : float
        Gamma
    m : int
        Number of basis vectors
    c : int
        Number of clusters

    Attributes
    ----------
    W_ : array-like of shape (n_samples, n_samples)
        Weight matrix (similarity matrix)
    S_ : array-like of shape (n_basis_vectors, n_samples)
        Coefficient matrix
    B_ : array-like of shape (n_dimensions, n_samples)
        Basis matrix
    F_ : array-like of shape (n_samples, n_clusters)
        Cluster indicator matrix
    converged_ : bool
        True if convergence was reached in `fit()`, False otherwise

    References
    ----------
    .. [1] Wang, F., Zhu, L., Liang, C., Li, J., Chang, X., & Lu, K. (2020). Robust optimal graph clustering. Neurocomputing, 378, 153-165.
    """

    def __init__(self, alpha, beta, gamma, m, c):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.m = m
        self.c = c


    def _initialize(self, n_samples):
        """
        Initializes W by the optimal solution to Eq. (4) in [1]
        """
        # TODO: Implement solution to equation 4
        self.W_ = np.ones((n_samples, n_samples))


    def fit(self, X, B):
        """
        Fit the model to X with dictionary matrix B.
        """
        self.fit_predict(X, B)
        return self


    def fit_predict(self, X, B):
        """
        Fit the model to X with dictionary matrix B and predict labels for X.
        """
        self.B_ = B

        self.converged_ = False

        self._initialize(X.shape[0])

        while not self.converged_:

            # Solve equation 1

            # Solve equation 18

            # Solve equation 19

            # Solve equation 22

            # Verify convergence
            self.converged_ = True

        # Return predictions
        return None
