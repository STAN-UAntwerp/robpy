import numpy as np
from scipy.stats import kendalltau
from robpy.covariance.base import RobustCovarianceEstimator
from robpy.univariate import QnEstimator


class KendallTauEstimator(RobustCovarianceEstimator):
    """Estimate the covariance matrix using Kendall's tau correlation."""

    def calculate_covariance(self, X) -> np.ndarray:
        p = X.shape[1]
        scales = [QnEstimator().fit(col).scale for col in X.T]
        cor = np.ones(shape=(p, p))
        cov = np.ones(shape=(p, p))
        for i in range(p):
            for j in range(i):
                cor[i, j] = cor[j, i] = kendalltau(X[:, i], X[:, j]).statistic
                cov[i, j] = cov[j, i] = cor[i, j] * scales[i] * scales[j]
            cov[i, i] = scales[i] ** 2
        self.correlation_ = cor
        return cov
