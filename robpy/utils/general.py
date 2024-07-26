import numpy as np


def inverse_submatrix(A: np.ndarray, A_inv: np.ndarray, indices: np.array) -> np.ndarray:
    """Given a matrix A and its inverse A_inv, this function calculates the inverse
    of the submatrix of A consisting of the rows and columns in indices.

    Arguments:
        A (np.ndarray): the matrix of interest
        A_inv (np.ndarray): the inverse of the matrix of interest
        indices (np.array): the indices corresponding to the submatrix of interest
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
