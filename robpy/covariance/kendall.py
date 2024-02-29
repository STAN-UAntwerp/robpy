import numpy as np
from scipy.stats import kendalltau
from statsmodels.robust.scale import qn_scale
from robpy.covariance.base import RobustCovarianceEstimator


class KendallTauEstimator(RobustCovarianceEstimator):
    def calculate_covariance(self, X) -> np.ndarray:
        p = X.shape[1]
        scales = qn_scale(X, axis=0)
        cor = np.ones(shape=(p, p))
        cov = np.ones(shape=(p, p))
        for i in range(p):
            for j in range(i):
                cor[i, j] = cor[j, i] = kendalltau(X[:, i], X[:, j]).statistic
                cov[i, j] = cov[j, i] = cor[i, j] * scales[i] * scales[j]
        self.correlation_ = cor
        return cov
