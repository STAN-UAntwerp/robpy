import numpy as np
from sklearn.covariance import MinCovDet


def mahalanobis_distance(data: np.ndarray, robust: bool = False, random_state: int = 101):
    """
    Calculate the Mahalanobis distance for multiple data vectors.

    Parameters:
    - data: An array-like object where each row is a data vector.

    Returns:
    - distances: An array of Mahalanobis distances for each data vector.
    """
    if robust:  # TODO: implement custom MCD (issue #3)
        mcd = MinCovDet(random_state=random_state).fit(data)
        mean = mcd.location_
        cov = mcd.covariance_
    else:
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)

    cov_inv = np.linalg.inv(cov)

    centered_data = data - mean
    return np.sqrt(np.sum(centered_data.dot(cov_inv) * centered_data, axis=1))
