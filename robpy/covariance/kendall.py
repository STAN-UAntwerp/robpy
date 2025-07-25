import numpy as np
from scipy.stats import kendalltau
from robpy.covariance.base import RobustCovariance
from robpy.univariate import Qn


class KendallTau(RobustCovariance):
    def __init__(
        self,
    ):
        """Estimate a covariance matrix using Kendall's tau pairwise correlation."""
        super().__init__()

    def calculate_covariance(self, X) -> np.ndarray:
        p = X.shape[1]
        scales = [Qn().fit(col).scale for col in X.T]
        cor = np.ones(shape=(p, p))
        cov = np.ones(shape=(p, p))
        for i in range(p):
            for j in range(i):
                cor[i, j] = cor[j, i] = kendalltau(X[:, i], X[:, j]).statistic
                cov[i, j] = cov[j, i] = cor[i, j] * scales[i] * scales[j]
            cov[i, i] = scales[i] ** 2
        self.correlation_ = cor
        return cov
