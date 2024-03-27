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
