import numpy as np

from scipy.stats import median_abs_deviation

from robpy.univariate.base import RobustScaleEstimator
from robpy.utils.rho import Huber


class UnivariateHuberMEstimator1step(RobustScaleEstimator):
    def __init__(
        self,
        b: float = 1.5,
        delta: float = 0.7784655,
        min_abs_scale: float = 1e-12,
    ):
        """
        Implementation of Huber M-estimator with 1 step: location and scale

        [analoguous to estLocScale {cellWise}: type ="hubhub"
        https://github.com/cran/cellWise/blob/master/src/LocScaleEstimators.cpp]

        Args:
            b (float, optional): Cutoff value for Huber's rho/psi.
                        Defaults to 1.5.
            delta (float, optional): Consistency factor at normal model depending on b:
                    quad(np.minimum(np.abs(x**2), b**2) * norm.pdf(x), -np.inf, np.inf, args=(b))
                        Defaults to 0.7784655.
            min_abs_scale (float, optional): Only if mad is larger than min_abs_scale
                        the M estimator will be calculated.
                        Defaults to 1e-12.
        """
        self.b = b
        self.delta = delta
        self.min_abs_scale = min_abs_scale

    def _calculate(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray):
                univariate data
        """
        n = len(X)

        # initial estimates:
        med = np.median(X)
        mad = median_abs_deviation(X, scale="normal")

        # first location:
        residuals = (X - med) / np.where(mad < self.min_abs_scale, 1, mad)
        mask = np.abs(residuals) > 1e-5
        weights = np.ones_like(residuals)
        weights[mask] = Huber(b=self.b).psi(residuals[mask]) / residuals[mask]
        mu_new = np.sum(weights * X) / np.sum(weights)

        # second scale:
        if mad < self.min_abs_scale:
            s_new = 0
        else:
            Xcentered = X - med
            residuals = Xcentered / mad
            weights = np.where(np.abs(residuals) <= self.b, residuals**2, self.b**2)
            s_new = mad * np.sqrt(1 / (n * self.delta) * np.sum(weights))

        self.location_ = mu_new
        self.scale_ = s_new

        return self
