import numpy as np


def inverse_submatrix(A: np.ndarray, A_inv: np.ndarray, indices: np.array) -> np.ndarray:
    """Given a matrix A and its inverse A_inv, this function calculates the inverse
    of the submatrix of A consisting of the rows and columns in indices.

    Arguments:
        A (np.ndarray): The matrix of interest.
        A_inv (np.ndarray): The inverse of the matrix of interest.
        indices (np.array): The indices corresponding to the submatrix of interest.
    """

    p = A.shape[1]
    indices = np.asarray(indices)
    n_submatrix = len(indices)

    if n_submatrix < p and n_submatrix > p / 2.0:  # in this scenario it useful to use the trick
        complement = np.ones(p, dtype=bool)
        complement[indices] = False
        indices_neg = np.flatnonzero(complement)
        return (
            A_inv[indices[:, None], indices]
            - A_inv[indices[:, None], indices_neg]
            @ np.linalg.inv(A_inv[indices_neg[:, None], indices_neg])
            @ A_inv[indices_neg[:, None], indices]
        )
    if n_submatrix < p:  # don't use the trick
        return np.linalg.inv(A[indices[:, None], indices])
    return A_inv  # submatrix is the original matrix
