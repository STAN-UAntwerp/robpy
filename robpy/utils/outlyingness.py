import numpy as np
from scipy.stats import median_abs_deviation


def stahel_donoho(X: np.ndarray, n_points: int = 2, n_dir: int = 250) -> np.ndarray:
    """Calculate the degree of outlyingness for multivariate points.
    Based on the algorithm proposed by Stahel (1981) and Donoho (1982).

    Args:
        X (np.ndarray): data matric of shape (n_obs, n_features)
        n_points (int, optional): number of points to determine the hyperplane. Defaults to 2.
        n_dir (int, optional): number of random directions to consider. Defaults to 250.

    Returns:
        np.ndarray: single column of outlyingness values

    References:
        Stahel W.A. (1981). Robuste Schatzungen: infinitesimale Optimalitat und Schatzungen von
        Kovarianzmatrizen. PhD Thesis, ETH Zurich.

        Donoho D.L. (1982). Breakdown properties of multivariate location estimators.
        Ph.D. Qualifying paper, Dept. Statistics, Harvard University, Boston.
    """
    # step 1: get random directions
    D = np.hstack(
        [_get_random_direction(X, n_points=n_points).reshape(-1, 1) for _ in range(n_dir)]
    )  # (n_features, n_dir)
    # step 2: projections
    projections = X @ D  # (n_obs, n_dir)

    # step 3: outlyingness
    # to do: let scale and loc estimators be passed as arguments
    return np.max(
        np.abs(projections - np.median(projections, axis=0))
        / median_abs_deviation(projections, axis=0),
        axis=1,
    )


def _get_random_direction(X: np.ndarray, n_points: int = 2) -> np.ndarray:
    """Get direction orthogonal to the hyperplane spanned by random points in X

    Args:
        X (np.ndarray): data matrix (n x p)
        n_points (int, optional): number of points to determine the hyperplane. Defaults to 2.

    Returns:
        np.ndarray: vector of shape (p, ) indicating the direction
    """
    points = X[np.random.choice(X.shape[0], n_points, replace=False)]

    if n_points == 2:
        d = points[0] - points[1]
    else:
        # vectors spanning hyperplane
        vectors = points - points[0]
        # direction perpendicular to hyperplane
        # U of SVD gives orthogonal basis, last vector is perpendicular to hyperplane
        d = np.linalg.svd(vectors)[-1][-1]

    return d / np.linalg.norm(d)
