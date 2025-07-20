import numpy as np

from scipy.stats import median_abs_deviation

from robpy.univariate.base import RobustScale
from robpy.utils.rho import BaseRho, Huber, TukeyBisquare
from robpy.univariate.mcd import UnivariateMCD


class OneStepM(RobustScale):
    def __init__(
        self,
        loc_rho: BaseRho,
        scale_rho: BaseRho,
        delta: float,
        min_abs_scale: float = 1e-12,
    ):
        """
        Implementation of the single-step M-estimator for location and scale.

        Args:
            loc_rho (BaseRho): Rho function for scale estimation (e.g. Huber(b=1.5)).
            scale_rho (BaseRho): Rho function for scale estimation (e.g. Huber(b=2.5)).
            delta (float): Consistency factor at normal model depending on b:
                    quad(np.minimum(np.abs(x**2), b**2) * norm.pdf(x), -np.inf, np.inf, args=(b)).
            min_abs_scale (float, optional): Only if MAD is larger than min_abs_scale
                        the M-estimator will be calculated. Defaults to 1e-12.

        References:
            Rousseeuw, P. J., & Van Den Bossche, W. (2018).
            Detecting deviating data cells. Technometrics, 60(2), 135-145.
        """
        super().__init__()
        self.loc_rho = loc_rho
        self.scale_rho = scale_rho
        self.delta = delta
        self.min_abs_scale = min_abs_scale

    def _calculate(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray):
                Univariate data.
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

        self.location_ = mu_new
        self.scale_ = s_new

        return self


class HuberOneStepM(OneStepM):
    def __init__(self):
        """
        Implementation of Huber M-estimator with 1 step: location and scale. Analogous to the
        R function estLocScale in the package cellWise using type = `hubhub`.
        (cfr. https://github.com/cran/cellWise/blob/master/src/LocScaleEstimators.cpp)
        """
        super().__init__(loc_rho=Huber(b=1.5), scale_rho=Huber(b=1.5), delta=0.7784655)


class CellwiseOneStepM(OneStepM):
    def __init__(self):
        """
        Implementation of the single step M estimator (robLoc and robScale) proposed in
        Rousseeuw, P. J., & Van Den Bossche, W. (2018). In this paper, the location rho function is
        set to `TukeyBiWeight(c=3)` and the scale rho function to `Huber(b=2.5)`.

        References:
            - Rousseeuw, P. J., & Van Den Bossche, W.  (2018). Detecting deviating data cells.
              Technometrics, 60(2), 135-145.
        """
        super().__init__(loc_rho=TukeyBisquare(c=3), scale_rho=Huber(b=2.5), delta=0.845)


class OneStepWrapping(RobustScale):
    def __init__(self):
        """
        Analogous to the R function estLocScale in the package cellWise using type = `wrap`.
        (cfr. https://github.com/cran/cellWise/blob/master/src/LocScaleEstimators.cpp)
        """
        super().__init__()

    def _calculate(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray):
                univariate data
        """

        # initial estimates: univariate MCD
        initial_estimates = UnivariateMCD().fit(X)

        # one step M estimator for location using hyperbolic tangent weight function:
        X_standardized = (X - initial_estimates.location) / initial_estimates.scale
        weights = np.array([self._tanh_weights(val) for val in X_standardized])
        self.location_ = np.sum(weights * X) / np.sum(weights)

        self.scale_ = initial_estimates.scale

    def _tanh_weights(self, val: float) -> float:
        b = 1.5
        c = 4
        if np.abs(val) < b:
            return 1.0
        elif np.abs(val) > c:
            return 0.0
        else:
            A = 0.7532528
            B = 0.8430849
            k = 4.1517212
            return (
                np.sqrt(A * (k - 1))
                * np.tanh(0.5 * np.sqrt((k - 1) * B**2 / A) * (c - np.abs(val)))
                / (np.abs(val))
            )
