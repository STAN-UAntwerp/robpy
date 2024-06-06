import numpy as np

from scipy.stats import median_abs_deviation

from robpy.univariate.base import RobustScaleEstimator
from robpy.utils.rho import BaseRho, Huber, TukeyBisquare


class OneStepMEstimator(RobustScaleEstimator):
    def __init__(
        self,
        loc_rho: BaseRho,
        scale_rho: BaseRho,
        delta: float,
        min_abs_scale: float = 1e-12,
    ):
        """
        Implementation of the single-step M-estimator for location and scale

        Args:
            loc_rho: rho function for scale estimation (e.g. Huber(b=1.5))
            scale_rho: rho function for scale estimation (e.g. Huber(b=2.5))
            delta (float, optional): Consistency factor at normal model depending on b:
                    quad(np.minimum(np.abs(x**2), b**2) * norm.pdf(x), -np.inf, np.inf, args=(b))
            min_abs_scale (float, optional): Only if mad is larger than min_abs_scale
                        the M estimator will be calculated.
                        Defaults to 1e-12.

        References:
            Rousseeuw, P. J., & Bossche, W. V. D. (2018).
            Detecting deviating data cells. Technometrics, 60(2), 135-145.
            --> loc_rho = TukeyBiWeight(c=3) and scale_rho = Huber(b=2.5)

            See also r code: https://rdrr.io/cran/cellWise/man/estLocScale.html

        """
        self.loc_rho = loc_rho
        self.scale_rho = scale_rho
        self.delta = delta
        self.min_abs_scale = min_abs_scale

    def _calculate(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray):
                univariate data
        """
        if np.isnan(X).any():
            raise ValueError("Input data contains NaN values")
        # initial estimates:
        med = np.median(X)
        mad = median_abs_deviation(X, scale="normal")
        safe_mad = 1 if mad < self.min_abs_scale else mad

        # first location:
        residuals = (X - med) / safe_mad
        mask = np.abs(residuals) > 1e-5
        weights = np.ones_like(residuals)
        weights[mask] = self.loc_rho.psi(residuals[mask]) / residuals[mask]
        mu_new = np.sum(weights * X) / np.sum(weights)

        # second scale:
        if mad < self.min_abs_scale:
            s_new = 0
        else:
            Xcentered = X - med
            residuals = Xcentered / mad
            s_new = mad * np.sqrt(1 / self.delta * np.mean(self.scale_rho.psi(residuals) ** 2))
            # TODO: confirm this is correct

        self.location_ = mu_new
        self.scale_ = s_new

        return self


class HuberOneStepMEstimator(OneStepMEstimator):
    def __init__(self):
        """
        Implementation of Huber M-estimator with 1 step: location and scale

        [analoguous to estLocScale {cellWise}: type ="hubhub"
        https://github.com/cran/cellWise/blob/master/src/LocScaleEstimators.cpp]
        """
        super().__init__(loc_rho=Huber(b=1.5), scale_rho=Huber(b=1.5), delta=0.7784655)


class CellwiseOneStepMEstimator(OneStepMEstimator):
    def __init__(self):
        """
        Implementation of the single step M estimator (robLoc and robScale) proposed in
        Rousseeuw, P. J., & Bossche, W. V. D. (2018).
            Detecting deviating data cells. Technometrics, 60(2), 135-145.
            --> loc_rho = TukeyBiWeight(c=3) and scale_rho = Huber(b=2.5)

        """
        super().__init__(loc_rho=TukeyBisquare(c=3), scale_rho=Huber(b=2.5), delta=0.845)
