from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import chi2
from scipy.linalg import pinvh
from sklearn.covariance import EmpiricalCovariance
from sklearn.exceptions import NotFittedError
from robpy.utils.distance import mahalanobis_distance


class RobustCovarianceEstimator(EmpiricalCovariance):
    def __init__(self, *, store_precision=True, assume_centered=False, nans_allowed=False):
        super().__init__(
            store_precision=store_precision,
            assume_centered=assume_centered,
        )
        self.nans_allowed = nans_allowed

    def fit(self, X: np.ndarray | pd.DataFrame) -> RobustCovarianceEstimator:
        """Fit the covariance estimator."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.nans_allowed:
            self.n_features_in_ = X.shape[1]
        else:
            X = self._validate_data(X)  # this sets n_features_in_ also

        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance = self.calculate_covariance(X)
        self._set_covariance(covariance)
        if self.store_precision:
            self.precision_ = pinvh(covariance)
        self.default_covariance_ = np.cov(X, rowvar=False)
        self._robust_distances = mahalanobis_distance(
            X, location=self.location_, covariance=self.covariance_
        )
        self._mahalanobis_distances = mahalanobis_distance(
            X, location=self.location_, covariance=self.default_covariance_
        )

        return self

    def calculate_covariance(self, X) -> np.ndarray:
        raise NotImplementedError

    @property
    def covariance(self) -> np.ndarray:
        if not hasattr(self, "covariance_") or self.covariance_ is None:
            raise NotFittedError()
        return self.covariance_

    @property
    def correlation(self) -> np.ndarray:
        if not hasattr(self, "correlation_"):
            raise AttributeError("This estimator does not calculate correlation.")
        return self.correlation_

    def distance_distance_plot(
        self, chi2_percentile: float = 0.975, figsize: tuple[int, int] = (4, 4)
    ):
        if not hasattr(self, "covariance_"):
            raise NotFittedError()

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        threshold = np.sqrt(chi2.ppf(chi2_percentile, self.n_features_in_))

        ax.scatter(x=self._mahalanobis_distances, y=self._robust_distances)

        ax.axhline(threshold, color="grey", linestyle="--")
        ax.axvline(threshold, color="grey", linestyle="--")
        ax.axline((0, 0), slope=1, color="grey", linestyle="-.")

        ax.set_xlabel("Mahalanobis distance (default covariance)")
        ax.set_ylabel("Robust distance")

        return fig
