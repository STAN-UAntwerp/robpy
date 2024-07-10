import numpy as np


def truncated_covariance(cov: np.ndarray, min_eigenvalue: float) -> np.ndarray:
    """Modifies the covariance such that all eigenvalues are at least as large as the
    given minimum value.

    Parameters:
    - cov: An np.ndarray representing the current covariance matrix.
    - min_eigenvalue: A float indicating what the minimal eigenvalue allowed eigenvalue.

    Returns:
    - a new covariance estimate.
    """
    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
    eigvals_idx = np.argsort(eigvals)[::-1]
    eigvals[eigvals < min_eigenvalue] = min_eigenvalue
    return eigvecs[:, eigvals_idx] @ np.diag(eigvals[eigvals_idx]) @ eigvecs[:, eigvals_idx].T


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Converts a covariance matrix to a correlation matrix

    Parameters:
    - cov: An np.ndarray representing the current covariance matrix.

    Returns:
    - the estimate for the corresponding correlation matrix.
    """
    stds = np.sqrt(np.diag(cov))
    return cov / np.outer(stds, stds)
