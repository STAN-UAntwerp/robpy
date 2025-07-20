import logging
import numpy as np
import pandas as pd

from robpy.covariance.base import RobustCovariance
from robpy.utils.distance import mahalanobis_distance
from robpy.utils.logging import get_logger
from robpy.covariance.utils.alter_covariance import truncated_covariance, covariance_to_correlation
from robpy.preprocessing.scaling import RobustScaler
from robpy.univariate.onestep_m import OneStepWrapping
from robpy.outliers.ddc import DDC
from robpy.preprocessing.utils import wrapping_transformation
from scipy.stats import chi2


class InitialDDCW(RobustCovariance):
    def __init__(
        self,
        *,
        alpha: float = 0.75,
        min_eigenvalue: float = 1e-4,
        verbosity: int = logging.WARNING,
    ):
        """
        Calculates the initial robust scatter and location estimates for the CellMCD, described
        in the Supplementary Material to Raymaekers, J., & Rousseeuw, P. J. (2024). The code is
        based on the function cellWise:::DDCWcov in R.

        Parameters:
            alpha (float, optional):
                Percentage indicating how much cells must remain unflagged in each column.
                Must lie within 0.5 to 1.0. Defaults to 0.75.
            min_eigenvalue (float, optional):
                Lower bound on the minimum eigenvalue of the covariance estimator
                on the standardized data. Should be at least 1e-6.
                Defaults to 1e-4.

        References:
            - Raymaekers, J., & Rousseeuw, P. J. (2024). The cellwise minimum covariance determinant
              estimator. Journal of the American Statistical Association, 119(548), 2610-2621.
        """
        super().__init__(store_precision=True, assume_centered=False, nans_allowed=True)
        self.alpha = alpha
        self.min_eigenvalue = min_eigenvalue
        self.logger = get_logger("InitialDDCW", level=verbosity)
        self.verbosity = verbosity

    def calculate_covariance(self, X: np.ndarray):
        n, p = X.shape

        # check that alpha creates a h-subset larger than [n/2]+1
        if 0.5 <= self.alpha <= 1:
            if self.alpha < (int(n / 2) + 1.0) / n:
                self.logger.warning(
                    f"h = alpha*n is too small and therefore set to [n/2] + 1"
                    f" ({(int(n / 2) + 1.0) / n})."
                )
            self.alpha = np.max([self.alpha, (int(n / 2) + 1.0) / n])
        else:
            raise ValueError(f"alpha must a float between 0.5 and 1, but received {self.alpha}.")

        # DDC with constraint -> imputed and rescaled Zimp:
        ddc = DDC(chi2_quantile=0.9, scale_estimator=OneStepWrapping()).fit(pd.DataFrame(X))
        W = np.copy(ddc.cellwise_outliers_)
        flagged_too_many = np.where(np.sum(W, axis=0) / X.shape[0] > 1 - self.alpha)[0]
        for i in flagged_too_many:
            ordering = np.argsort(np.abs(ddc.standardized_residuals_[:, i]))[::-1]
            W[:, i] = [False for _ in range(n)]
            W[ordering[0 : int(n * 0.25)], i] = True
        Zimp = np.copy(X)
        Zimp[np.logical_or(W, np.isnan(X))] = ddc.impute(
            pd.DataFrame(X), impute_outliers=True
        ).to_numpy()[np.logical_or(W, np.isnan(X))]
        Z = (X - ddc.location_) / ddc.scale_
        Zimp = (Zimp - ddc.location_) / ddc.scale_
        Zimp_original = np.copy(Zimp)
        Zimp = np.delete(Zimp, np.where(ddc.row_outliers_)[0], axis=0)

        # project data on eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(Zimp, rowvar=False))
        eigenvectors = eigvecs[:, np.where(eigvals > self.min_eigenvalue)[0][::-1]]
        Zimp_proj = Zimp @ eigenvectors

        # wrapped location and covariance
        Zimp_proj_scaler = RobustScaler(scale_estimator=OneStepWrapping()).fit(
            Zimp_proj, ignore_nan=True
        )
        Zimp_proj_scaler.scales_[
            Zimp_proj_scaler.scales_ < self.min_eigenvalue
        ] = self.min_eigenvalue
        Zimp_proj_wrapped_cov = np.cov(
            wrapping_transformation(
                Zimp_proj,
                location_estimator=lambda *args, **kwargs: Zimp_proj_scaler.locations_,
                scale_estimator=lambda *args, **kwargs: Zimp_proj_scaler.scales_,
                rescale=True,
            ),
            rowvar=False,
        )
        cov = eigenvectors @ Zimp_proj_wrapped_cov @ eigenvectors.T  # back to original axis system
        cov = covariance_to_correlation(cov)
        cov = (truncated_covariance(cov, self.min_eigenvalue) * ddc.scale_).T * ddc.scale_

        # temporary points: delete casewise outliers
        U = np.minimum(np.maximum(Z, -2), 2)
        RD = mahalanobis_distance(U, np.zeros(p), cov) ** 2
        U_outlying_cases = np.where(RD / np.median(RD) * chi2.ppf(0.5, p) > chi2.ppf(0.99, p))[0]
        Z = np.delete(Z, U_outlying_cases, axis=0)
        Zimp = np.delete(Zimp_original, U_outlying_cases, axis=0)

        # project data on eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(Zimp, rowvar=False))
        eigenvectors = eigvecs[:, np.where(eigvals > self.min_eigenvalue)[0][::-1]]
        Zimp_proj = Zimp @ eigenvectors

        # wrapped location and covariance
        Zimp_proj_scaler = RobustScaler(scale_estimator=OneStepWrapping()).fit(
            Zimp_proj, ignore_nan=True
        )
        Zimp_proj_scaler.scales_[
            Zimp_proj_scaler.scales_ < self.min_eigenvalue
        ] = self.min_eigenvalue
        Zimp_proj_wrapped_cov = np.cov(
            wrapping_transformation(
                Zimp_proj,
                location_estimator=lambda *args, **kwargs: Zimp_proj_scaler.locations_,
                scale_estimator=lambda *args, **kwargs: Zimp_proj_scaler.scales_,
                rescale=True,
            ),
            rowvar=False,
        )
        cov = eigenvectors @ Zimp_proj_wrapped_cov @ eigenvectors.T  # back to original axis system
        cov = covariance_to_correlation(cov)
        cov = (truncated_covariance(cov, self.min_eigenvalue) * ddc.scale_).T * ddc.scale_

        self.location_ = np.array(ddc.location_)

        return cov
