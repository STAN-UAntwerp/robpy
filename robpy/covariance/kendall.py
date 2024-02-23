import numpy as np
from scipy.stats import kendalltau
from robpy.covariance.base import RobustCovarianceEstimator


class KendallTauEstimator(RobustCovarianceEstimator):
    def calculate_covariance(self, X) -> np.ndarray:
        p = X.shape[1]
        cov = np.ones(shape=(p, p))
        for i in range(p):
            for j in range(i):
                tau = kendalltau(X[:, i], X[:, j]).statistic
                cov[i, j] = cov[j, i] = tau * np.std(X[:, i]) * np.std(X[:, j])
        return cov
