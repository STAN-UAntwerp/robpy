import numpy as np


def l1median(X: np.ndarray):
    epsilon = 1e-7
    mu_0 = np.mean(X, axis=0)
    while True:
        centered_X = X - mu_0
        d = np.sqrt(np.sum(centered_X * centered_X, axis=1))
        W = 1 / d
        mu_1 = np.sum((X * W[:, np.newaxis]).T / np.sum(W), axis=1)
        if np.sum(np.abs(mu_0 - mu_1)) < epsilon:
            return mu_1
        else:
            mu_0 = mu_1
