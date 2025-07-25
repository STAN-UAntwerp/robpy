import numpy as np

from typing import Callable
from scipy.stats import median_abs_deviation


def wrapping_transformation(
    X: np.ndarray,
    b: float = 1.5,
    c: float = 4.0,
    q1: float = 1.540793,
    q2: float = 0.8622731,
    rescale: bool = True,
    location_estimator: Callable[[np.ndarray, int], np.ndarray] = np.median,
    scale_estimator: Callable[[np.ndarray, int], np.ndarray] = median_abs_deviation,
) -> np.ndarray:
    """
    Implementation of the wrapping transformation using the following function:

    .. math::

        \\Psi_{b, c}(z) =
        \\begin{cases}
          z & \\text{if } \\  0 \\leq |z| < b, \\\\
          q_1 \\tanh\\left(q_2 (c - |z|)\\right) \\mathrm{sign}(z) & \\text{if } \\  b \\leq |z|
          \\leq c, \\\\
          0  & \\text{if } \\   c < |z|.
        \\end{cases}

    Args:
        X (np.ndarray):
            Data to be transformed, must have shape (N, D).
        b (float, optional):
                Lower cutoff. Defaults to 1.5.
        c (float, optional):
            Upper cutoff. Defaults to 4.0.
        q1 (float, optional):
            Transformation parameter (see formula). Defaults to 1.540793.
        q2 (float, optional):
            Transformation parameter (see formula). Defaults to 0.8622731.
        rescale (bool, optional):
            Whether to rescale the wrapped data such that the robust location and
            scale of the transformed data are the same as the original data.
            Defaults to True.
        location_estimator (Callable[[np.ndarray, int], np.ndarray], optional):
            Function to estimate the location of the data, should accept an array like input as
            first value and a named argument axis. Defaults to np.median.
        scale_estimator (Callable[[np.ndarray, int], np.ndarray], optional):
            Function to estimate the scale of the data, should accept an array like input as
            first value and a named argument axis. Defaults to median_abs_deviation.

    Returns:
        np.ndarray: The transformed data.

    References:
        - Raymaekers, J., & Rousseeuw, P. J. (2021). Fast robust correlation for
          high-dimensional data. Technometrics, 63(2), 184-198.
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
