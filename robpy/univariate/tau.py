import numpy as np

from robpy.univariate.base import RobustScaleEstimator
from scipy.stats import median_abs_deviation, norm


class TauEstimator(RobustScaleEstimator):
    def __init__(
        self,
        c1: float = 4.5,
        c2: float = 3.0,
        consistency_correction: bool = True,
    ):
        """
        Implementation of tau estimator of scale

        [Robust Estimates of Location and Dispersion for High-Dimensional Datasets,
        Ricarco A Maronna and Ruben H Zamar (2002)]

        Args:
            c1 (float, optional):
                constant for the weight function, defaults to 4.5
            c2 (float, optional):
                constant for the rho function, defaults to 3.0
            consistency_correction (bool, optional):
                boolean indicating if consistency for normality should be applied.
                Defaults to True.
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.consistency_correction = consistency_correction

    def _calculate(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray):
                univariate data
        """
        n = len(X)
        sigma0 = median_abs_deviation(X)
        weights = self._weight_function((X - np.median(X)) / sigma0)
        self.location_ = np.sum(X * weights) / np.sum(weights)
        self.scale_ = sigma0 * np.sqrt(
            1 / n * np.sum(self._rho_function((X - self.location_) / sigma0))
        )
        if self.consistency_correction:
            """
            expectation of rho(X/qnorm(3/4)) for X standard normal
            """
            b = self.c2 * norm.ppf(3 / 4)
            corr = 2 * ((1 - b**2) * norm.cdf(b) - b * norm.pdf(b) + b**2) - 1
            self.scale_ = self.scale_ / np.sqrt(corr)

    def _weight_function(self, X):
        return np.where(
            np.abs(X) <= self.c1,
            (1 - (X / self.c1) ** 2) ** 2,
            0.0,
        )

    def _rho_function(self, X):
        return np.where(X**2 <= self.c2**2, X**2, self.c2**2)
