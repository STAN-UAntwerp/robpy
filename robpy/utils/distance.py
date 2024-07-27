import numpy as np
import pandas as pd


def mahalanobis_distance(
    data: np.ndarray | pd.DataFrame, location: np.ndarray, covariance: np.ndarray
):
    """
    Calculate the Mahalanobis distance for multiple data vectors.

    Parameters:
        data (np.ndarray or pd.DataFrame): An array-like object where each row is a data vector.
        location (np.ndarray): the center of the data
        covariance (np.ndarray): the scatter estimator of the data

    Returns:
        np.ndarray: an array of Mahalanobis distances for each data vector.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    cov_inv = np.linalg.inv(covariance)

    centered_data = data - location.flatten()
    return np.sqrt(np.sum(centered_data.dot(cov_inv) * centered_data, axis=1))
