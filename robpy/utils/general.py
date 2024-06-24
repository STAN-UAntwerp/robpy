import numpy as np


def inverse_submatrix(A: np.ndarray, A_inv: np.ndarray, indices: np.array) -> np.ndarray:
    """Given a matrix A and its inverse A_inv, this function calculates the inverse
    of the submatrix of A consisting of the rows and columns in indices.

    Arguments:
    - A (np.ndarray): the matrix of interest
    - A_inv (np.ndarray): the inverse of the matrix of interest
    - indices (np.array): the indices corresponding to the submatrix of interest
    """

    p = A.shape[1]
    n_submatrix = len(indices)
    indices_neg = np.setdiff1d(np.arange(p), indices)
    result = np.zeros([n_submatrix, n_submatrix])

    if n_submatrix < p and n_submatrix > p / 2.0:  # in this scenario it useful to use the trick
        result = (
            A_inv[np.ix_(indices, indices)]
            - A_inv[np.ix_(indices, indices_neg)]
            @ np.linalg.inv(A_inv[np.ix_(indices_neg, indices_neg)])
            @ A_inv[np.ix_(indices_neg, indices)]
        )
    elif n_submatrix < p and n_submatrix <= p / 2.0:  # don't use the trick
        result = np.linalg.inv(A[np.ix_(indices, indices)])
    else:  # submatrix is the original matrix
        result = A_inv

    return result


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
