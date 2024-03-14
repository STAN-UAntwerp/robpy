import numpy as np
from scipy.stats import median_abs_deviation, chi2

from robpy.utils import mahalanobis_distance
from robpy.covariance.base import RobustCovarianceEstimator
from robpy.preprocessing import LocationOrScaleEstimator


class OGKEstimator(RobustCovarianceEstimator):
    def __init__(
        self,
        *,
        store_precision=True,
        assume_centered=False,
        location_estimator: LocationOrScaleEstimator = np.median,
        scale_estimator: LocationOrScaleEstimator = median_abs_deviation,
        n_iterations: int = 2,
        reweighting: bool = False,
        reweighting_beta: float = 0.9
    ):
        """Implementation of the Orthogonalized Gnanadesikan-Kettenring estimator for location
        dispersion proposed in
            Maronna, R. A., & Zamar, R. H. (2002).
            Robust Estimates of Location and Dispersion for High-Dimensional Datasets.
            Technometrics, 44(4), 307–317. http://www.jstor.org/stable/1271538

        Args:
            - store_precision: whether to store the precision matrix
            - assume_centered: whether the data is already centered
            - location_estimator: function to estimate the location of the data, should accept
                an array like input as first value and a named argument axis
            - scale_estimator: function to estimate the scale of the data, should accept
                an array like input as first value and a named argument axis
            - n_iterations: number of iteration for orthogonalization step
            - reweighting: whether to apply reweighting at the end
                (i.e. calculating regular location and covariance after filtering outliers based on
                Mahalanobis distance using OGK estimates)
            - reweighting_beta: quantile of chi2 distribution to use as cutoff for reweighting

        """
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.location_estimator = location_estimator
        self.scale_estimator = scale_estimator
        self.n_iterations = n_iterations
        self.reweighting = reweighting
        self.reweighting_beta = reweighting_beta

    def calculate_covariance(self, X) -> np.ndarray:
        """Calculate location and covariance with the algorithm described in Maronna & Zamar (2002).
        Covariance is returned, location is overwritten.
        """
        p = X.shape[1]
        Z = X
        for _ in range(self.n_iterations):
            D = np.diag(self.scale_estimator(Z, axis=0))  # (p x p)
            Y = Z @ np.linalg.inv(D).T  # (n x p)
            U = np.ones(shape=(p, p))
            # Loop over pairs of variables, lower triangle suffises as U is symmetric
            for i in range(p):
                for j in range(i):
                    scale_sum = self.scale_estimator(Y[:, i] + Y[:, j])
                    scale_diff = self.scale_estimator(Y[:, i] - Y[:, j])
                    cor = (scale_sum**2 - scale_diff**2) / (scale_sum**2 + scale_diff**2)
                    U[i, j] = U[j, i] = cor
            _, E = np.linalg.eig(U)  # (p x p)
            Z = Y @ E  # (n x p)
        var = np.diag(np.power(self.scale_estimator(Z, axis=0), 2))  # (p x p)
        m = self.location_estimator(Z, axis=0)  # (p, )
        mu_Y = E @ m  # (p, )
        cov_Y = E @ var @ E.T  # (p x p)

        mu_X = D @ mu_Y  # (p, )
        cov_X = D @ cov_Y @ D.T  # (p x p)

        if self.reweighting:
            mahalanobis = mahalanobis_distance(X, location=mu_X, covariance=cov_X)
            cutoff = np.sqrt(chi2.ppf(self.reweighting_beta, p) / chi2.ppf(0.5, p)) * np.median(
                mahalanobis
            )  # mahalanobis is the sqrt distance, so we need to take the sqrt of the chi2 quantiles
            mask = mahalanobis < cutoff
            cov_X = np.cov(X[mask], rowvar=False)
            mu_X = np.mean(X[mask])
        self.location_ = mu_X
        return cov_X
