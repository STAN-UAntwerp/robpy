import numpy as np
from scipy.stats import median_abs_deviation, chi2


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

    Returns:
        raw_var: raw variance estimate
        raw_location: raw location estimate
    """
    n = len(X)
    if h_size is None:
        h_size = int(np.floor(n / 2) + 1)
    elif isinstance(h_size, int) and (1 <= h_size <= n):
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
    X = sorted(X)
    for i in range(n - h_size + 1):
        var_new = np.var(X[i : (i + h_size)])
        if var_new < var_best:
            var_best = var_new
            index_best = i
    raw_var = var_best
    raw_location = np.mean(X[index_best : (index_best + h_size)])
    if consistency_correction:
        raw_var = raw_var * (h_size / n) / chi2.cdf(chi2.ppf(h_size / n, df=1), df=3)
    return raw_var, raw_location
