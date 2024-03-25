import numpy as np
import logging


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


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger with the specified name and level.

    Parameters:
    - name: The name of the logger.
    - level: The logging level.

    Returns:
    - logger: A logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    logger.addHandler(handler)
    return logger
