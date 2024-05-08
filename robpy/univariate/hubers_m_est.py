import numpy as np

from scipy.stats import median_abs_deviation, norm

from robpy.univariate.base import RobustScaleEstimator
from robpy.utils.rho import BaseRho, Huber


class UnivariateHuberMEstimator(RobustScaleEstimator):
    def __init__(
        self,
        rho: BaseRho = Huber(b=1.5),
        delta: float = norm.cdf(1.5) * (1 + 1.5**2) + 1.5 * norm.pdf(1.5) - 0.5 - 1.5**2,
        tol: float = 1e-6,
        min_abs_scale: float = 1e-12,
    ):
        """
        Implementation of univariate M-estimator: location and scale

        [Robust statistics, Ricardo A. Maronna, R. Douglas Martin & Victor J. Yohai (2006)]

        Args:
            rho (BaseRho, optional):
                Rho function to use on the residuals.
                Defaults to Huber's rho with b = 1.5.
            delta (float, optional):
                consistency factor corresponding to the rho function
                Defaults to 0.477153 (the delta corresponding to Huber's rho with b = 1.5)
            tol (float, optional):
                tolerance, determines the stopping criteria for convergence
                Defaults to 1e-6
            min_abs_scale (float, optional): Only if mad is larger than min_abs_scale
                        the M estimator will be calculated
                        Defaults to 1e-12.
        """
        self.rho = rho
        self.tol = tol
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
        if mad < self.min_abs_scale:
            raise ValueError("The sample has a scale that is (approximately) zero.")

        # first location:
        mu_old = med
        while True:
            residuals = (X - mu_old) / mad
            mask = np.abs(residuals) > 1e-5
            wi = np.ones_like(residuals)
            wi[mask] = self.rho.psi(residuals[mask]) / residuals[mask]
            mu_new = np.sum(wi * X) / np.sum(wi)
            if np.abs(mu_new - mu_old) < self.tol * mad:
                break
            else:
                mu_old = mu_new

        # second scale:
        s_old = mad
        Xcentered = X - med
        while True:
            residuals = Xcentered / s_old
            mask = np.abs(residuals) > 1e-5
            wi = np.ones_like(residuals)
            wi[mask] = self.rho.rho(residuals[mask]) / (residuals[mask] ** 2)
            s_new = np.sqrt(1 / (n * self.delta) * np.sum(wi * Xcentered**2))
            if np.abs(s_new / s_old - 1) < self.tol:
                break
            else:
                s_old = s_new

        self.location_ = mu_new
        self.scale_ = s_new

        return self
