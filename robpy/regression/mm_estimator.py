import numpy as np
from robpy.regression.base import RobustRegressor
from robpy.regression.s_estimator import SEstimator
from robpy.utils.rho import BaseRho, TukeyBisquare


class MMEstimator(RobustRegressor):
    def __init__(
        self,
        initial_estimator: RobustRegressor = SEstimator(),
        rho: BaseRho = TukeyBisquare(c=3.44),
    ):
        self.initial_estimator = initial_estimator
        self.rho = rho

    def fit(self, X, y):
        pass

    def predict(self, X) -> np.ndarray:
        return np.array([])
