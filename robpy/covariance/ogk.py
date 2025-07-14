import numpy as np
from scipy.stats import median_abs_deviation, chi2

from robpy.utils.distance import mahalanobis_distance
from robpy.covariance.base import RobustCovariance
from robpy.univariate import LocationOrScaleEstimator


class OGK(RobustCovariance):
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
        """
        Implementation of the Orthogonalized Gnanadesikan-Kettenring estimator for location and
        dispersion proposed in Maronna, R. A., & Zamar, R. H. (2002).

        Args:
            store_precision (boolean, optional):
                Whether to store the precision matrix. Defaults to True.
            assume_centered (boolean, optional):
                Whether the data is already centered. Defaults to False.
            location_estimator (LocationOrScaleEstimator, optional):
                Function to estimate the location of the data, should accept an array like input as
                first value and a named argument axis. Defaults to np.median.
            scale_estimator (LocationOrScaleEstimator, optional):
                Function to estimate the scale of the data, should accept an array like input as
                first value and a named argument axis. Defaults to median_abs_deviation.
            n_iterations (int, optional):
                Number of iterations for the orthogonalization step. Defaults to 2.
            reweighting (boolean, optional):
                Whether to apply reweighting at the end (i.e. calculating regular location and
                covariance after filtering outliers based on Mahalanobis distance using OGK
                estimates). Defaults to False.
            reweighting_beta (float, optional):
                Quantile of chi-squared distribution to use as cutoff for the reweighting. Defaults
                to 0.9.

        References:
            - Maronna, R. A., & Zamar, R. H. (2002). Robust estimates of location and dispersion for
              high-dimensional datasets. Technometrics, 44(4), 307-317.

        """
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.location_estimator = location_estimator
        self.scale_estimator = scale_estimator
        self.n_iterations = n_iterations
        self.reweighting = reweighting
        self.reweighting_beta = reweighting_beta

    def calculate_covariance(self, X) -> np.ndarray:
        p = X.shape[1]
        Z = np.copy(X)
        DE = []
        for _ in range(self.n_iterations):
            s = np.array(self.scale_estimator(Z, axis=0))
            D = np.diag(s)
            Dinv = np.diag(1.0 / s)
            Y = Z @ Dinv  # (n x p)
            U = np.ones(shape=(p, p))
            # Loop over pairs of variables, lower triangle suffises as U is symmetric
            for i in range(p):
                for j in range(i):
                    scale_sum = self.scale_estimator(Y[:, i] + Y[:, j])
                    scale_diff = self.scale_estimator(Y[:, i] - Y[:, j])
                    cor = (scale_sum**2 - scale_diff**2) / (scale_sum**2 + scale_diff**2)
                    U[i, j] = U[j, i] = cor
            _, E = np.linalg.eigh(U)  # (p x p)
            E = E[:, ::-1]
            Z = Y @ E  # (n x p)
            DE.append(D @ E)
        cov_X = np.diag(np.power(self.scale_estimator(Z, axis=0), 2))  # (p x p)
        mu_X = self.location_estimator(Z, axis=0)  # (p, )

        for mat in reversed(DE):
            mu_X = mat @ mu_X.reshape(-1, 1)
            cov_X = mat @ cov_X @ mat.T

        mu_X = mu_X.flatten()
        if self.reweighting:
            mahalanobis = mahalanobis_distance(X, location=mu_X, covariance=cov_X)
            cutoff = np.sqrt(chi2.ppf(self.reweighting_beta, p) / chi2.ppf(0.5, p)) * np.median(
                mahalanobis
            )  # mahalanobis is the sqrt distance, so we need to take the sqrt of the chi2 quantiles
            mask = mahalanobis < cutoff
            cov_X = np.cov(X[mask], rowvar=False)
            mu_X = np.mean(X[mask], axis=0)
        self.location_ = mu_X.flatten()

        return cov_X
