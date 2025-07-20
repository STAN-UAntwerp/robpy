import numpy as np
from scipy.stats import median_abs_deviation


def stahel_donoho(
    X: np.ndarray,
    n_points: int = 2,
    n_dir: int = 250,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Calculate a degree of outlyingness for multivariate points.
    Based on the principle proposed by Stahel, W. A. (1981) and Donoho, D. L. (1982).

    Args:
        X (np.ndarray): Data matrix of shape (n_obs, n_features).
        n_points (int, optional): Number of points to determine the direction to project on.
            Defaults to 2. For n_points = 2, each projection is on a line passing through 2 data
            points, as in Hubert et al. (2005). If not, each projection is on the direction
            orthogonal to a hyperplane passing through n_points data points.
        n_dir (int, optional): Number of random directions to consider. Defaults to 250.
        random_seed (int | None, optional): Can be used to provide a random seed. Defaults to None.

    Returns:
        np.ndarray: Single column of outlyingness values.

    References:
        - Donoho, D. L. (1982). Breakdown properties of multivariate location estimators.
          Technical report, Harvard University, Boston.
        - Hubert, M., Rousseeuw, P. J., & Vanden Branden, K. (2005). ROBPCA: a new approach to
          robust principal component analysis. Technometrics, 47(1), 64-79.
        - Stahel, W. A. (1981). Robuste schätzungen: infinitesimale optimalität und schätzungen von
          kovarianzmatrizen (Doctoral dissertation, ETH Zurich).
    """
    # step 1: get random directions
    D = np.hstack(
        [
            _get_random_direction(X, n_points=n_points, random_seed=random_seed).reshape(-1, 1)
            for _ in range(n_dir)
        ]
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


def _get_random_direction(
    X: np.ndarray,
    n_points: int = 2,
    random_seed: int | None = None,
) -> np.ndarray:
    """Get direction orthogonal to the hyperplane spanned by random points in X.

    Args:
        X (np.ndarray): data matrix (n x p).
        n_points (int, optional): number of points to determine the hyperplane. Defaults to 2.
        random_seed (int | None, optional): can be used to provide a random seed. Defaults to None.

    Returns:
        np.ndarray: vector of shape (p, ) indicating the direction.
    """
    rng = np.random.default_rng(random_seed)
    points = X[rng.choice(X.shape[0], n_points, replace=False)]

    if n_points == 2:
        d = points[0] - points[1]
    else:
        # vectors spanning hyperplane
        vectors = points - points[0]
        # direction perpendicular to hyperplane
        # U of SVD gives orthogonal basis, last vector is perpendicular to hyperplane
        d = np.linalg.svd(vectors)[-1][-1]

    return d / np.linalg.norm(d)
