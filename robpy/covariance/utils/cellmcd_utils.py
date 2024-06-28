import numpy as np

from robpy.utils.general import inverse_submatrix


def objective_function(
    X: np.ndarray, W: np.ndarray, location: np.array, cov: np.ndarray, cov_inv: np.ndarray
) -> float:
    """Calculates the value of the objective function in equation (8) of the CellMCD paper without
    the penalty for a certain X, W, location, covariance and the inverse covariance
    [Raymaekers and Rousseeuw, The Cellwise Minimum Covariance Determinant Estimator, 2023,
            Journal of the American Statistical Association]

    Arguments:
    - X (np.ndarray): the data
    - W (np.ndarray): the matrix W describing which cells are currently flagged
    - location (np.array): the current location estimate
    - cov (np.ndarray): the current covariance estimate
    - cov_inv (np.ndarray): the inverse of the current covariance estimate
    """

    objective = 0
    unique_rows = np.unique(W, axis=0)
    for w in unique_rows:
        row_idx_w = np.where(np.all(W == w, axis=1))[0]
        w_ones = np.where(w)[0]

        subset_location = np.array(location)[w_ones]
        subset_cov = cov[np.ix_(w_ones, w_ones)]
        subset_cov_inv = inverse_submatrix(cov, cov_inv, w_ones)

        subset_X = X[row_idx_w][:, w_ones] - subset_location
        partial_MD = np.sum(subset_X.dot(subset_cov_inv) * subset_X, axis=1)
        objective = objective + np.sum(
            partial_MD + np.linalg.slogdet(subset_cov)[1] + np.log(2 * np.pi) * len(w_ones)
        )

    return objective
