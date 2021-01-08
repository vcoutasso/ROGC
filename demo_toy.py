"""
The purpose of this demonstration is to evaluate and visualize the results of this ROGC implementation on some toy datasets.
This demo uses the same datasets as the ones seen in https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html.
"""

import numpy as np

from sklearn import datasets
from sklearn.decomposition import DictionaryLearning

from rogc import ROGC

np.random.seed(0)

# Generate toy data
n_samples = 1500

X, y = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
#X, y = datasets.make_blobs(150, centers=2, n_features=5)

# Learn dictionary B from data X
B = DictionaryLearning(n_components=3, n_jobs=-1).fit(X).components_.T

X = X.T

alpha = 1
beta = 0.5
gamma = 1
m = B.shape[1]
c = np.unique(y).shape[0]

model = ROGC(B, alpha, beta, gamma, m, c)

preds = model.fit_predict(X)
