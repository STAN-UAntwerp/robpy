import numpy as np

from scipy.stats import median_abs_deviation

from robpy.univariate.base import RobustScaleEstimator
from robpy.utils.rho import Huber


class UnivariateHuberMEstimator1step(RobustScaleEstimator):
    def __init__(
        self,
        min_abs_scale: float = 1e-12,
    ):
        """
        Implementation of Huber M-estimator with 1 step: location and scale

        [analoguous to estLocScale {cellWise}: type ="hubhub"
        https://github.com/cran/cellWise/blob/master/src/LocScaleEstimators.cpp]

        Args:
            min_abs_scale (float, optional): Only if mad is larger than min_abs_scale
                        the M estimator will be calculated
                        Defaults to 1e-12.
        """
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
        if mad < self.min_abs_scale:
            raise ValueError("The sample has a scale that is (approximately) zero.")

        # first location:
        residuals = (X - med) / mad
        mask = np.abs(residuals) > 1e-5
        wi = np.ones_like(residuals)
        wi[mask] = Huber(b=1.5).psi(residuals[mask]) / residuals[mask]
        mu_new = np.sum(wi * X) / np.sum(wi)

        # second scale:
        Xcentered = X - med
        residuals = Xcentered / mad
        wi = np.where(np.abs(residuals) <= 1.5, residuals**2, 1.5**2)
        s_new = mad * np.sqrt(1 / (n * 0.7784655) * np.sum(wi))

        self.location_ = mu_new
        self.scale_ = s_new

        return self
