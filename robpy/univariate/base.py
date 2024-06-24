from __future__ import annotations

import numpy as np
from sklearn.exceptions import NotFittedError
from abc import abstractmethod, ABC
from typing import Protocol


class LocationOrScaleEstimator(Protocol):
    def __call__(self, x: np.ndarray, axis: int = 0) -> np.ndarray | float:
        ...


class RobustScaleEstimator(ABC):
    def fit(self, X: np.ndarray, omit_nans: bool = False) -> RobustScaleEstimator:
        if len(X.shape) != 1:
            raise ValueError(
                f"X must be univariate, but received a matrix with dimensions {X.shape}"
            )
        if omit_nans:
            X = X[~np.isnan(X)]
        self._calculate(X)
        return self

    @abstractmethod
    def _calculate(self, X: np.ndarray):
        """Must set self.scale_ and self.location_"""
        pass

    @property
    def scale(self):
        if not hasattr(self, "scale_"):
            raise NotFittedError("Scale not available. First run .fit()")
        return self.scale_

    @property
    def location(self):
        if not hasattr(self, "location_"):
            raise NotFittedError("Location not available. First run .fit()")
        return self.location_
