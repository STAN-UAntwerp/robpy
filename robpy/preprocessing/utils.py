import numpy as np

from typing import Callable
from scipy.stats import median_abs_deviation


def wrapping_transformation(
    X: np.ndarray,
    b: float = 1.5,
    c: float = 4.0,
    q1: float = 1.540793,
    q2: float = 0.8622731,
    rescale: bool = False,
    location_estimator: Callable[[np.ndarray, int], np.ndarray] = np.median,
    scale_estimator: Callable[[np.ndarray, int], np.ndarray] = median_abs_deviation,
) -> np.ndarray:
    """
    Implementation of wrapping using this transformation function:

    .. math::

        \\Psi_{b, c}(z) =
        \\begin{cases}
          z & if \\  0 \\leq |z| < b \\\\
          q_1 \\tanh\\left(q_2 (c - |z|)\\right) \\mathrm{sign}(z) & if \\  b \\leq |z| \\leq c \\\\
          0  & if \\   c < |z|
        \\end{cases}

    Args:
        X: data to be transformed, must have shape (N, D)
        b: lower cutoff
        c: upper cutoff
        q1, q2: transformation parameters (see formula)
        rescale: whether to rescale the wrapped data so the robust location and scale
                 of the transformed data are the same as the original data
        locations: location estimates of the columns of X (optional)
        scales: scale estimates of the columns of X (optional)

    Returns:
        transformed data
    """
    locations = location_estimator(X, axis=0)
    scales = scale_estimator(X, axis=0)
    scales_no_zero = np.where(scales == 0, 1, scales)

    z = (X - locations) / scales_no_zero

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
            z_wrapped * (scales / z_wrapped_std_no_zero)
            + locations
            - (z_wrapped_mean * (scales / z_wrapped_std_no_zero))
        )
    else:
        return z_wrapped * scales + locations
