from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from robpy.regression.base import RobustRegressor, _convert_input_to_array
from robpy.regression.s_estimator import SEstimator
from robpy.utils.rho import BaseRho, TukeyBisquare


class MMEstimator(RobustRegressor):
    """
    Implementation of MM-regression estimator

    References:
        https://www.jstor.org/stable/2241331
    """

    def __init__(
        self,
        initial_estimator: RobustRegressor = SEstimator(),
        rho: BaseRho = TukeyBisquare(c=3.44),
        max_iterations: int = 500,
        epsilon: float = 1e-7,
    ):
        self.initial_estimator = initial_estimator
        self.rho = rho
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.model = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> MMEstimator:
        X, y = _convert_input_to_array(X, y)
        self.model = self.initial_estimator.fit(X, y)
        self._scale = self.model.scale
        self.n_iter = 0
        for i in range(self.max_iterations):
            residuals = y - self.model.predict(X)
            scaled_residuals = residuals / self._scale
            weights = self.rho.psi(scaled_residuals) / scaled_residuals
            self.model = LinearRegression().fit(
                np.sqrt(weights).reshape(-1, 1) * X, np.sqrt(weights) * y
            )
            new_residuals = y - self.model.predict(X)
            self.n_iter = i + 1
            if np.max(np.abs(new_residuals - residuals)) < self.epsilon:
                break
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise NotFittedError
        X, _ = _convert_input_to_array(X)
        return self.model.predict(X)
