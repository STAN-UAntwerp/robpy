import numpy as np


def l1median(X: np.ndarray) -> float:
    """
    Implementation of the L1-median

    Args:
        X (np.ndarray): Data to compute the L1-median on.

    References:
        Fritz, H. and Filzmoser, P. and Croux, C. (2012)
        A comparison of algorithms for the multivariate L1-median.
        Computational Statistics 27, 393–410
    """
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


def weighted_median(X: np.ndarray, weights: np.ndarray) -> float:
    """
    Computes a weighted median.

    References:
        Time-efficient algorithms for two highly robust estimators
        of scale, Christophe Croux and Peter J. Rousseeuw (1992)
    """
    n = len(X)
    wrest = 0
    wtotal = np.sum(weights)
    Xcand = X
    while True:
        k = np.ceil(n / 2).astype("int")
        if n > 1:
            trial = np.partition(X, k - 1)[
                :k
            ].max()  # k^th order statistic, I think this can be programmed better...
        else:
            return Xcand[0]
        wleft = np.sum(weights[X < trial])
        wmid = np.sum(weights[X == trial])
        if (2 * (wrest + wleft)) > wtotal:
            Xcand = X[X < trial]
            weightscand = weights[X < trial]
        elif (2 * (wrest + wleft + wmid)) > wtotal:
            return trial
        else:
            Xcand = X[X > trial]
            weightscand = weights[X > trial]
            wrest = wrest + wleft + wmid
        X = Xcand
        weights = weightscand
        n = len(X)
