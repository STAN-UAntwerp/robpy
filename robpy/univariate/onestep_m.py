import numpy as np

from scipy.stats import median_abs_deviation

from robpy.univariate.base import RobustScale
from robpy.utils.rho import BaseRho, Huber, TukeyBisquare
from robpy.univariate._cellwise import loc_scale_1d


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


class _CellwiseLocScale(OneStepM):
    """Shared implementation of the estimators that reproduce a type of ``cellWise::estLocScale``
    exactly (see :mod:`robpy.univariate._cellwise`)."""

    _type: str

    def __init__(self, loc_rho: BaseRho, scale_rho: BaseRho, delta: float, center: bool = True):
        super().__init__(loc_rho=loc_rho, scale_rho=scale_rho, delta=delta)
        self.center = center

    def _calculate(self, X: np.ndarray):
        self.location_, self.scale_ = loc_scale_1d(X, self._type, center=self.center)
        return self


class HuberOneStepM(_CellwiseLocScale):
    _type = "hubhub"

    def __init__(self, center: bool = True):
        """
        Huber M-estimator of location and scale with one step each, as the R function
        ``estLocScale`` of the package cellWise with ``type = "hubhub"``.

        The location takes one Huber step (b = 1.5) from the median, with the MAD as scale. The
        scale takes one step of the Huber scale M-estimator (b = 1.5) around that location,
        starting from the MAD around the location.

        Args:
            center (bool, optional): If False, the location is fixed at 0 and the scale is
                computed around 0 (``center = FALSE`` in cellWise). Defaults to True.

        References:
            - Raymaekers, J., & Rousseeuw, P. J. (2023). cellWise: Analyzing Data with Cellwise
              Outliers. R package, ``src/LocScaleEstimators.cpp``.
        """
        super().__init__(
            loc_rho=Huber(b=1.5), scale_rho=Huber(b=1.5), delta=0.7784655, center=center
        )


class CellwiseOneStepM(_CellwiseLocScale):
    _type = "1stepM"

    def __init__(self, center: bool = True):
        """
        The single step M-estimators of location and scale (robLoc and robScale) of
        Rousseeuw & Van den Bossche (2018), as the R function ``estLocScale`` of the package
        cellWise with ``type = "1stepM"``, which DDC uses by default.

        The location takes one step with the biweight weight function from the median (with
        c = 3 relative to the unnormalized MAD). The scale takes one step of the Huber scale
        M-estimator with b = 2.5 * qnorm(0.75) around that location, starting from the MAD
        around the location.

        Args:
            center (bool, optional): If False, the location is fixed at 0 and the scale is
                computed around 0 (``center = FALSE`` in cellWise). Defaults to True.

        References:
            - Rousseeuw, P. J., & Van Den Bossche, W.  (2018). Detecting deviating data cells.
              Technometrics, 60(2), 135-145.
        """
        super().__init__(
            loc_rho=TukeyBisquare(c=3), scale_rho=Huber(b=2.5), delta=0.845, center=center
        )


class OneStepWrapping(RobustScale):
    def __init__(self, center: bool = True):
        """
        Location and scale as the R function ``estLocScale`` of the package cellWise with
        ``type = "wrap"`` (its default): the reweighted univariate MCD of cellWise (alpha = 0.5,
        with its finite-sample corrections), followed by one step of the location M-estimator
        with the hyperbolic tangent (wrapping) weight function. The scale is the MCD scale.

        Args:
            center (bool, optional): If False, the location is fixed at 0 and the scale is
                computed around 0 (``center = FALSE`` in cellWise). Defaults to True.

        References:
            - Raymaekers, J., & Rousseeuw, P. J. (2023). cellWise: Analyzing Data with Cellwise
              Outliers. R package, ``src/LocScaleEstimators.cpp``.
        """
        super().__init__()
        self.center = center

    def _calculate(self, X: np.ndarray):
        """
        Args:
            X (np.ndarray):
                univariate data
        """
        self.location_, self.scale_ = loc_scale_1d(X, "wrap", center=self.center)
        return self
