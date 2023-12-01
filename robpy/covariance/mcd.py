import numpy as np

from sklearn.covariance import MinCovDet

from robpy.covariance.base import RobustCovarianceEstimator


class FastMCDEstimator(RobustCovarianceEstimator):
    def calculate_covariance(self, X) -> np.ndarray:
        return MinCovDet().fit(X).covariance_
