from __future__ import annotations
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import median_abs_deviation, chi2
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from robpy.utils.distance import mahalanobis_distance
from robpy.covariance import FastMCDEstimator

logger = logging.getLogger(__name__)


class RobustRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
    ):
        super().__init__()

    def fit(self, X, y) -> RobustRegressor:
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    @property
    def scale(self) -> float:
        scale = getattr(self, "_scale", None)
        if scale is None:
            raise NotFittedError("Model has not been fitted yet.")
        return scale

    def diagnostic_plot(
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
        """Create a diagnostic plot where robust residuals are plotted against the robust
        mahalabobis distances of the training data.

        Args:
            X (array like of shape (n_samples, n_features)): training features
            y (array like of shape (n_samples, )): training targets
            robust_scaling (bool): whether to scale residuals using MAD instead of std
            robust_distance (bool): whether to use MCD as loc/scale estimator instead of mean/cov
                for calculating the Mahalanobis distances
            vertical_outlier_threshold: where to draw the upper (and lower) limit for
                the standardized residuals to indicate outliers
            leverage_threshold_percentile: which percentile from the chisquare distribution
                to use to set as threshold for leverage points
            figsize (tuple[int, int], optional): Size of the plot. Defaults to (10, 4).
            return_data (bool, optional):
                Whether to return the residuals, the standardized residuals and the distances.
                Defaults to False.
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
            mcd = FastMCDEstimator().fit(X)
            covariance = mcd.covariance_
            location = mcd.location_
        else:
            covariance = np.cov(X, rowvar=False)
            location = np.mean(X, axis=0)
        distances = mahalanobis_distance(X, location=location, covariance=covariance)

        _, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(x=distances, y=standardized_residuals)
        ax.set_xlabel(f"Mahalanobis distance {'(robust)' if robust_distance else ''}")
        ax.set_ylabel(f"Standardized residuals  {'(robust)' if robust_scaling else ''}")

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
