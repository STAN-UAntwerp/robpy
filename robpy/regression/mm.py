from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from robpy.regression.base import RobustRegression, _convert_input_to_array
from robpy.regression.s import SRegression
from robpy.utils.rho import BaseRho, TukeyBisquare


class MMRegression(RobustRegression):
    def __init__(
        self,
        initial_estimator: RobustRegression = SRegression(),
        rho: BaseRho = TukeyBisquare(c=3.44),
        max_iterations: int = 500,
        epsilon: float = 1e-7,
    ):
        """
        Implementation of MM-regression estimator of Yohai, V. J. (1987).

        Args:
            initial_estimator (RobustRegression, optional):
                Initial regression estimator. Defaults to SRegression.
            rho (BaseRho, optional):
                The rho-function used for the MM-estimate. Defaults to TukeyBisquare(c=3.44).
            max_iterations (int, optional):
                Maximum number of iterations. Defaults to 500.
            epsilon (float, optional):
                If the absolute difference between all the new and old residuals in an iteration
                is below epsilon, we stop the computation. Defautls to 1e-7.

        References:
            - Yohai, V. J. (1987). High breakdown-point and high efficiency robust estimates for
              regression. The Annals of statistics, 15(2), 642-656.
        """
        self.initial_estimator = initial_estimator
        self.rho = rho
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.model = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> MMRegression:
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
