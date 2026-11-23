from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2

from robpy.covariance.base import RobustCovariance
from robpy.covariance.utils.alter_covariance import covariance_to_correlation
from robpy.outliers.ddc import DDC
from robpy.univariate import OneStepWrapping
from robpy.univariate._cellwise import est_loc_scale, wrap
from robpy.utils.logging import get_logger


def truncate_eigenvalues(cov: np.ndarray, min_eigenvalue: float) -> np.ndarray:
    """Raise the eigenvalues of a symmetric matrix to at least ``min_eigenvalue``, from its
    eigendecomposition (the R function ``truncEig`` of cellWise)."""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
    return eigvecs @ np.diag(np.maximum(eigvals, min_eigenvalue)) @ eigvecs.T


def _wrapped_covariance(Zimp: np.ndarray, min_eigenvalue: float, center: bool) -> np.ndarray:
    """Covariance of the wrapped principal component scores of Zimp, rotated back."""
    eigvals, eigvecs = np.linalg.eigh(np.cov(Zimp, rowvar=False))
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    eigvecs = eigvecs[:, eigvals >= min_eigenvalue]
    scores = Zimp @ eigvecs
    loc, scale = est_loc_scale(scores, "wrap", center=center)
    scale = np.maximum(scale, min_eigenvalue)
    wrapped = wrap(scores, loc, scale)
    return eigvecs @ np.atleast_2d(np.cov(wrapped, rowvar=False)) @ eigvecs.T


class InitialDDCW(RobustCovariance):
    def __init__(
        self,
        *,
        alpha: float = 0.75,
        min_eigenvalue: float = 1e-4,
        fixed_center: bool = False,
        verbosity: int = logging.WARNING,
    ):
        """
        Calculates the initial robust scatter and location estimates for the CellMCD, described
        in the Supplementary Material to Raymaekers, J., & Rousseeuw, P. J. (2024). This follows
        the function ``DDCWcov`` of the R package cellWise (version 2.5.7):

        1. DDC (with ``tolProbCell = 0.9`` and the wrapping standardization) flags cells; in
           columns with more than a fraction 1 - alpha of flagged cells, only the cells with
           the largest absolute standardized residuals are kept flagged. The flagged and missing
           cells are imputed by DDC, and the rows flagged by DDC are left out.
        2. The covariance of the imputed, standardized data is the back-rotated covariance of
           its wrapped principal component scores (scores standardized around 0).
        3. Rows that are outlying with respect to that covariance (after clipping the
           standardized data to [-2, 2]) are left out, and step 2 is repeated on the remaining
           rows (now with centered scores). The result is converted to a correlation matrix,
           its eigenvalues are raised to at least ``min_eigenvalue``, and it is rescaled with
           the DDC scales.

        As in cellWise, step 3 removes no rows when the data contain missing values (their
        Mahalanobis distances are then not available).

        Parameters:
            alpha (float, optional):
                Percentage indicating how much cells must remain unflagged in each column.
                Must lie within 0.5 to 1.0. Defaults to 0.75.
            min_eigenvalue (float, optional):
                Lower bound on the minimum eigenvalue of the covariance estimator
                on the standardized data. Should be at least 1e-6.
                Defaults to 1e-4.
            fixed_center (bool, optional):
                If True, the center is fixed at 0 (``fixedCenter`` in cellWise) and the DDC
                scales are computed around 0. Defaults to False.

        References:
            - Raymaekers, J., & Rousseeuw, P. J. (2024). The cellwise minimum covariance determinant
              estimator. Journal of the American Statistical Association, 119(548), 2610-2621.
        """
        super().__init__(store_precision=True, assume_centered=False, nans_allowed=True)
        self.alpha = alpha
        self.min_eigenvalue = min_eigenvalue
        self.fixed_center = fixed_center
        self.logger = get_logger("InitialDDCW", level=verbosity)
        self.verbosity = verbosity

    def calculate_covariance(self, X: np.ndarray):
        n, p = X.shape
        if not 0.5 <= self.alpha <= 1:
            raise ValueError(f"alpha must a float between 0.5 and 1, but received {self.alpha}.")
        if p > n:
            raise ValueError(
                f"You need at least {p + 1} cases to estimate a nonsingular covariance matrix of "
                f"size {p}, but there are only {n} cases."
            )
        max_col = 1 - self.alpha

        # DDC with at most a fraction max_col of flagged cells per column
        frame = pd.DataFrame(X)
        ddc = DDC(
            cell_quantile=0.9,
            scale_estimator=OneStepWrapping(),
            center=np.zeros(p) if self.fixed_center else None,
            min_unique_values=0,
        ).fit(frame)
        flags = ddc.cellwise_outliers_.copy()
        for j in np.flatnonzero(flags.sum(axis=0) > max_col * n):
            residuals = np.abs(ddc.standardized_residuals_[:, j])
            order = np.argsort(-np.nan_to_num(residuals, nan=-np.inf), kind="stable")
            flags[:, j] = False
            flags[order[: int(np.floor(max_col * n))], j] = True
        estimates = ddc.rescaled_predictions_
        Ximp = np.where(flags | np.isnan(X), estimates, X)
        loc, scale = ddc.location_, ddc.scale_
        Z = (X - loc) / scale
        Zimp = (Ximp - loc) / scale

        # first covariance: wrapped scores around 0, without the rows flagged by DDC
        keep = ~ddc.row_outliers_
        cov = _wrapped_covariance(Zimp[keep], self.min_eigenvalue, center=False)
        cov = truncate_eigenvalues(cov, self.min_eigenvalue)
        cov = covariance_to_correlation(cov) * np.outer(scale, scale)

        # row rule: leave out the rows that are outlying with respect to this covariance
        clipped = np.clip(Z, -2, 2)
        inv = np.linalg.inv(cov)
        distances = np.einsum("ij,jk,ik->i", clipped, inv, clipped)
        if np.isnan(distances).any():
            outlying = np.zeros(n, dtype=bool)  # cellWise: median(MDs) is NA
        else:
            outlying = distances / np.median(distances) * chi2.ppf(0.5, p) > chi2.ppf(0.99, p)

        # final covariance: wrapped centered scores of the remaining rows
        cov = _wrapped_covariance(Zimp[~outlying], self.min_eigenvalue, center=True)
        cov = truncate_eigenvalues(covariance_to_correlation(cov), self.min_eigenvalue)
        cov = cov * np.outer(scale, scale)

        self.location_ = np.asarray(loc, dtype=float)
        return cov
