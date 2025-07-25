from __future__ import annotations
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation, chi2
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from robpy.utils.distance import mahalanobis_distance
from robpy.covariance import FastMCD

logger = logging.getLogger(__name__)


class RobustRegression(RegressorMixin, BaseEstimator):
    def __init__(
        self,
    ):
        """
        Base class for robust regression estimators.
        """
        super().__init__()

    def fit(self, X, y) -> RobustRegression:
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @property
    def scale(self) -> float:
        scale = getattr(self, "_scale", None)
        if scale is None:
            raise NotFittedError("Model has not been fitted yet.")
        return scale

    def outlier_map(
        self,
        X,
        y,
        robust_scaling: bool = True,
        robust_distance: bool = True,
        vertical_outlier_threshold: float = 2.5,
        leverage_threshold_percentile: float = 0.975,
        figsize: tuple[int, int] = (4, 4),
        return_data: bool = False,
    ) -> None | tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        Creates a diagnostic plot where the robust residuals of the target are plotted against the
        robust Mahalanobis distances of the features.

        Args:
            X (array like of shape (n_samples, n_features)):
                Training features.
            y (array like of shape (n_samples, )):
                Training targets.
            robust_scaling (bool, optional):
                Whether to scale the residuals using MAD instead of std. Defaults to True.
            robust_distance (bool, optional):
                Whether to use the MCD as loc/scale estimator instead of mean/cov for calculating
                the Mahalanobis distances. Defaults to True.
            vertical_outlier_threshold (float, optional):
                Where to draw the upper (and lower) limit for the standardized residuals to indicate
                outliers. Defaults to 2.5.
            leverage_threshold_percentile (float, optional):
                Which percentile from the chi-squared distribution to use to set as threshold for
                leverage points. Defaults to 0.975.
            figsize (tuple[int, int], optional):
                Size of the plot. Defaults to (4, 4).
            return_data (bool, optional):
                Whether to return the residuals, the standardized residuals and the distances.
                Defaults to False.
        References:
            - Rousseeuw P.J., Hubert M. (2018). Anomaly detection by robust statistics.
              Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery,
              8(2), 1–14.
            - Rousseeuw P.J., van Zomeren B.C. (1990). Unmasking multivariate outliers and leverage
              points. Journal of the American Statistical Association,  85(411), 633–651.
        """

        residuals = y.reshape(-1, 1) - self.predict(X).reshape(-1, 1)
        standardized_residuals = (
            residuals
            / (
                median_abs_deviation(residuals, scale="normal")
                if robust_scaling
                else np.std(residuals)
            )
        ).flatten()

        if robust_distance:
            mcd = FastMCD().fit(X)
            covariance = mcd.covariance_
            location = mcd.location_
        else:
            covariance = np.cov(X, rowvar=False)
            location = np.mean(X, axis=0)
        distances = mahalanobis_distance(X, location=location, covariance=covariance)

        _, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(x=distances, y=standardized_residuals)
        ax.set_xlabel(f"{'Robust' if robust_distance else 'Non-robust'} distance of X")
        ax.set_ylabel(f"{'Robust' if robust_scaling else 'Non-robust'} standardized residuals of y")

        ax.axhline(vertical_outlier_threshold, ls="--", c="grey")
        ax.axhline(-vertical_outlier_threshold, ls="--", c="grey")

        df = X.shape[-1]
        leverage_threshold = float(np.sqrt(chi2.ppf(leverage_threshold_percentile, df)))
        ax.axvline(leverage_threshold, ls="--", c="grey")

        if return_data:
            return (
                residuals,
                standardized_residuals,
                distances,
                vertical_outlier_threshold,
                leverage_threshold,
            )


def _convert_input_to_array(X, y=None) -> tuple[np.ndarray, np.ndarray | None]:
    if isinstance(X, pd.DataFrame):
        X = X.values
    if (len(X.shape) == 1) or (X.shape[1] == 1):
        X = X.reshape(-1, 1)
    if isinstance(y, pd.Series):
        y = y.values

    return X, y
