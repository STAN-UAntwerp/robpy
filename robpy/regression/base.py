import logging
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import median_abs_deviation, chi2
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.covariance import MinCovDet

from robpy.utils import mahalanobis_distance

logger = logging.getLogger(__name__)


class RobustRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
    ):
        super().__init__()

    def predict(self, X):
        raise NotImplementedError

    def diagnostic_plot(
        self,
        X,
        y,
        robust_scaling: bool = True,
        robust_distance: bool = True,
        vertical_outlier_threshold: float = 2.5,
        leverage_threshold_percentile: float = 0.975,
        figsize: tuple[int, int] = (10, 4),
    ) -> plt.Figure:
        """Create a diagnostic plot where robust residuals are plotted against the robust
        mahalabobis distances of the training data.
        Horizontal thresholds (i.e. the thresholds for vertical outliers )

        Args:
            X (array like of shape (n_samples, n_features)): training features
            y (array like of shape (n_samples, )): training targets
            robust_scaling (bool): whether to scale residuals using MAD instead of std
            robust_distance (bool): whether to use MCD as loc/scale estimator instead of mean/cov
                for calculating the Mahalanobis distances
            vertical_outlier_threshold: where to draw the upper (and lower) limit for
                the standardized residuals to indicate outliers
            horizontal_threshold_percentile: which percentile from the chisquare distribution
                to use to set as threshold for leverage points
        """

        residuals = self.predict(X).reshape(-1, 1) - y.reshape(-1, 1)
        standardized_residuals = (
            residuals / (median_abs_deviation(residuals) if robust_scaling else np.std(residuals))
        ).flatten()

        if robust_distance:
            mcd = MinCovDet().fit(X)
            covariance = mcd.covariance_
            location = mcd.location_
        else:
            covariance = np.cov(X, rowvar=False)
            location = np.mean(X, axis=0)
        distances = mahalanobis_distance(X, location=location, covariance=covariance)

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.scatter(x=distances, y=standardized_residuals)
        ax.set_xlabel(f"Mahalanobis distance {'(robust)' if robust_distance else ''}")
        ax.set_ylabel(f"Standardized residuals  {'(robust)' if robust_scaling else ''}")

        ax.axhline(vertical_outlier_threshold, ls="--", c="grey")
        ax.axhline(-vertical_outlier_threshold, ls="--", c="grey")

        df = X.shape[-1]
        leverage_threshold = np.sqrt(chi2.ppf(leverage_threshold_percentile, df))
        ax.axvline(leverage_threshold, ls="--", c="grey")

        return fig
