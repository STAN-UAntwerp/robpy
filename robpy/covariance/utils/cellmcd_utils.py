import numpy as np

from robpy.utils.general import inverse_submatrix


def group_rows_by_pattern(B: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group the rows of a 0/1 matrix by their pattern.

    The patterns are packed into bytes and sorted once (``np.unique`` with ``return_inverse``),
    and the rows are split by group with one stable sort, instead of comparing every row with
    every pattern. The groups come in the order of ``np.unique(B, axis=0)``.

    Args:
        B (np.ndarray): Matrix of zeros and ones (or booleans), shape (n, p).

    Returns:
        list of (pattern, rows): the boolean pattern of each group and the indices of its rows.
    """
    B = np.asarray(B, dtype=bool)
    if B.shape[0] == 0:
        return []
    keys = np.packbits(B, axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    bounds = np.flatnonzero(np.diff(inverse[order])) + 1
    return [(B[first[g]], rows) for g, rows in enumerate(np.split(order, bounds))]


def objective_function(
    X: np.ndarray, W: np.ndarray, location: np.ndarray, cov: np.ndarray, cov_inv: np.ndarray
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
    for w, row_idx_w in group_rows_by_pattern(W):
        w_ones = np.flatnonzero(w)

        subset_location = np.array(location)[w_ones]
        subset_cov = cov[np.ix_(w_ones, w_ones)]
        subset_cov_inv = inverse_submatrix(cov, cov_inv, w_ones)

        subset_X = X[row_idx_w][:, w_ones] - subset_location
        partial_MD = np.sum(subset_X.dot(subset_cov_inv) * subset_X, axis=1)
        objective = objective + np.sum(
            partial_MD + np.linalg.slogdet(subset_cov)[1] + np.log(2 * np.pi) * len(w_ones)
        )

    return objective
