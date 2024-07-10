import numpy as np
import pandas as pd
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)

from robpy.univariate.base import RobustScaleEstimator
from robpy.univariate.mcd import UnivariateMCDEstimator


class RobustScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Scaling features using a RobustScaleEstimator"""

    def __init__(
        self,
        scale_estimator: RobustScaleEstimator = UnivariateMCDEstimator(),
        with_centering: bool = True,
        with_scaling: bool = True,
    ):
        self.scale_estimator = scale_estimator
        self.with_centering = with_centering
        self.with_scaling = with_scaling

    def fit(self, X: np.ndarray | pd.DataFrame, ignore_nan: bool = False):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.locations_ = np.zeros(X.shape[1])
        self.scales_ = np.ones(X.shape[1])

        for feature_idx in range(X.shape[1]):
            fitted_estimator = self.scale_estimator.fit(X[:, feature_idx], ignore_nan)
            if self.with_centering:
                self.locations_[feature_idx] = fitted_estimator.location
            if self.with_scaling:
                self.scales_[feature_idx] = fitted_estimator.scale

        return self

    def transform(self, X: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return (X - self.locations_) / self.scales_

    def inverse_transform(self, X: np.ndarray | pd.DataFrame):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        return X * self.scales_ + self.locations_
