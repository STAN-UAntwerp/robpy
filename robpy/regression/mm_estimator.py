import numpy as np
from robpy.regression.base import RobustRegressor
from robpy.regression.s_estimator import SEstimator


class MMEstimator(RobustRegressor):
    def __init__(
        self,
        score_function: ScoreFunction = ScoreFunction.BISQUARE,
        initial_estimator: RobustRegressor = SEstimator(),
    ):
        self.score_function = score_function

    def fit(self, X, y):
        pass

    def predict(self, X) -> np.ndarray:
        return np.array([])
