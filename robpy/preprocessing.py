import numpy as np
from scipy.stats import median_abs_deviation


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
