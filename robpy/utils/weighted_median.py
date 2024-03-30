import numpy as np


def weighted_median(X: np.ndarray, weights: np.ndarray) -> float:
    """
    Computes a weighted median.
    Based on [Time-efficient algorithms for two highly robust estimators of scale,
    Christophe Croux and Peter J. Rousseeuw (1992)]"""
    n = len(X)
    wrest = 0
    wtotal = np.sum(weights)
    Xcand = X
    while True:
        k = np.ceil(n / 2).astype("int")
        if n > 1:
            trial = np.partition(X, k)[
                :k
            ].max()  # k^th order statistic, I think this can be programmed better...
        else:
            trial = Xcand
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
