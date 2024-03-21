import numpy as np
import math as math
from scipy.stats import median_abs_deviation, chi2, gamma


def wrapping_transformation(
    X: np.ndarray,
    b: float = 1.5,
    c: float = 4.0,
    q1: float = 1.540793,
    q2: float = 0.8622731,
    rescale: bool = False,
) -> np.ndarray:
    """
    Implementation of wrapping using this transformation function:

    phi(z) = {
        z                                       if 0 <= |z| < b
        q1 * tanh(q2 * (c - |z|)) * sign(z)     if b <= |z| <= c
        0                                       if c <= |z|
    }

    Args:
        X: data to be transformed, must have shape (N, D)
        b: lower cutoff
        c: upper cutoff
        q1, q2: transformation parameters (see formula)
        rescale: whether to rescale the wrapped data so the robust location and scale
                 of the transformed data are the same as the original data

    Returns:
        transformed data
    """

    median = np.median(X, axis=0)
    mad = median_abs_deviation(X, axis=0)
    mad_no_zero = np.where(mad == 0, 1, mad)

    z = (X - median) / mad_no_zero

    z_wrapped = np.where(
        np.abs(z) < b,
        z,
        np.where(np.abs(z) <= c, q1 * np.tanh(q2 * (c - np.abs(z))) * np.sign(z), 0),
    )
    if rescale:
        z_wrapped_mean = np.mean(z_wrapped, axis=0)
        z_wrapped_std = np.std(z_wrapped, axis=0)
        z_wrapped_std_no_zero = np.where(z_wrapped_std == 0, 1, z_wrapped_std)
        return (
            z_wrapped * (mad / z_wrapped_std_no_zero)
            + median
            - (z_wrapped_mean * (mad / z_wrapped_std_no_zero))
        )
    else:
        return z_wrapped * mad + median


def univariateMCD(
    X: np.array, h_size: float | int | None = None, consistency_correction=True
) -> np.array:
    """
    Implementation of univariate MCD

    [Minimum covariance determinant, Mia Hubert & Michiel Debruyne (2009)]

    Args:
        X: univariate data
        h_size: parameter determining the size of the h-subset
        consistency_correction: whether the estimates should be consistent at the normal model

    Returns:
        raw_var: raw variance estimate
        raw_loc: raw location estimate
        var: reweigthed variance estimate
        loc: reweigthed location estimate
    """
    n = len(X)
    if h_size is None:
        h_size = int(np.floor(n / 2) + 1)
    elif h_size == 1:
        return np.var(X), np.mean(X), np.var(X), np.mean(X)
    elif isinstance(h_size, int) and (1 < h_size <= n):
        pass
    elif isinstance(h_size, float) and (0 < h_size < 1):
        h_size = int(h_size * n)
    else:
        raise ValueError(
            f"h_size must be an integer > 1 and <= n or a float between 0 and 1 "
            f"but received {h_size}"
        )
    var_best = np.inf
    index_best = 1
    X = np.array(sorted(X))
    for i in range(n - h_size + 1):
        var_new = np.var(X[i : (i + h_size)])
        if var_new < var_best:
            var_best = var_new
            index_best = i
    raw_var = var_best
    raw_loc = np.mean(X[index_best : (index_best + h_size)])
    if consistency_correction:
        """[Minimum covariance determinant, Mia Hubert & Michiel Debruyne (2009)]"""
        raw_var = raw_var * (h_size / n) / chi2.cdf(chi2.ppf(h_size / n, df=1), df=3)
    distances = (X - raw_loc) ** 2 / raw_var
    mask = distances < chi2.ppf(0.975, df=1)
    loc = np.mean(X[mask])
    var = np.var(X[mask])
    if consistency_correction:
        """[Influence function and efficiency of the MCD scatter matrix estimator,
        Christophe Croux & Gentiane Haesbroeck (1999)]"""
        delta = np.sum(mask) / n
        var = var * delta * np.reciprocal(gamma.cdf(chi2.ppf(delta, df=1) / 2, a=3 / 2))
    return var, loc, raw_var, raw_loc


def Qn(X: np.array) -> float:
    """
    Implementation of Qn estimator

    [Time-efficient algorithms for two highly robust estimators of scale,
    Christophe Croux and Peter J. Rousseeuw (1992)]
    [Selecting the k^th element in X+Y and X1+...+Xm,
    Donald B. Johnson and Tetsuo Mizoguchi (1978)]

    Args:
        X: univariate data
    Returns:
        scale: robust Qn scale estimator
    """
    n = len(X)
    h = n // 2 + 1
    k = h * (h - 1) // 2
    c = 2.219144
    y = np.sort(X)
    left = n + 1 - np.arange(n)
    right = np.full(n, n)
    jhelp = n * (n + 1) // 2
    knew = k + jhelp
    nL = jhelp
    nR = n * n
    found = False

    while (nR - nL) > n and (not found):
        weight = right - left + 1
        jhelp = (left + weight // 2).astype(int)
        work = y - y[n - jhelp]
        trial = weighted_median(work, weight)
        P = np.searchsorted(-np.flip(y), trial - y, "left", np.arange(n))
        Q = np.searchsorted(-np.flip(y), trial - y, "right", np.arange(n)) + 1
        if knew <= np.sum(P):
            right = P
            nR = np.sum(P)
        elif knew > (np.sum(Q) - n):
            left = Q
            nL = np.sum(Q) - n
        else:
            Qn = trial
            found = True
    if not found:
        work = []
        for i in range(1, n):
            if left[i] <= right[i]:
                for jj in range(int(left[i]), int(right[i] + 1)):
                    work.append(y[i] - y[n - jj - 1 + 1])
        k = int(knew - nL)
        Qn = np.partition(np.array(work), k)[:k].max()
    if n <= 9:
        dn_dict = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844, 6: 0.611, 7: 0.857, 8: 0.669, 9: 0.872}
        dn = dn_dict.get(n)
    else:
        if n % 2 != 0:
            dn = n / (n + 1.4)
        else:
            dn = n / (n + 3.8)
    Qn = dn * c * Qn
    return Qn


def weighted_median(X: np.array, weights: np.array) -> float:
    """based on [Time-efficient algorithms for two highly robust estimators of scale,
    Christophe Croux and Peter J. Rousseeuw (1992)]"""
    n = len(X)
    wrest = 0
    wtotal = np.sum(weights)
    while 1:
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
    return trial
