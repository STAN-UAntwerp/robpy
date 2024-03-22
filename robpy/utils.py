import numpy as np


def mahalanobis_distance(data: np.ndarray, location: np.ndarray, covariance: np.ndarray):
    """
    Calculate the Mahalanobis distance for multiple data vectors.

    Parameters:
    - data: An array-like object where each row is a data vector.

    Returns:
    - distances: An array of Mahalanobis distances for each data vector.
    """

    cov_inv = np.linalg.inv(covariance)

    centered_data = data - location
    return np.sqrt(np.sum(centered_data.dot(cov_inv) * centered_data, axis=1))


def weighted_median(X: np.array, weights: np.array) -> float:
    """
    Computes a weighted median.
    Based on [Time-efficient algorithms for two highly robust estimators of scale,
    Christophe Croux and Peter J. Rousseeuw (1992)]"""
    n = len(X)
    wrest = 0
    wtotal = np.sum(weights)
    while True:
        k = np.ceil(n / 2).astype("int")
        if n > 1:
            trial = np.partition(X, k)[
                :k
            ].max()  # k^th order statistic, I think this can be programmed better...
        else:
            trial = Xcand
        wleft = np.sum(weights[X < trial])
        wright = np.sum(weights[X > trial])
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
