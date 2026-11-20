import numpy as np
import logging
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)
from typing import Literal
from robpy.utils.rho import TukeyBisquare
from scipy.stats import median_abs_deviation, norm, chi2
from scipy.optimize import minimize_scalar
from robpy.univariate import HuberOneStepM


class RobustPowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        method: Literal["boxcox", "yeojohnson", "auto"] = "yeojohnson",
        standardize: bool = True,
        lambda_range: tuple[float, float] = (-4.0, 6.0),
        quantile: float = 0.99,
        nsteps: int = 2,
        rectified: bool = False,
    ):
        """
        Apply a robust power transformation using reweighted maximum likelihood to transform the
        features closer to normality. Uses the Box-Cox or the Yeo-Johnson transformation.

        The transformation parameter lambda is estimated as in Raymaekers & Rousseeuw (2024): an
        initial estimate from the rectified transformation, followed by ``nsteps`` steps of
        reweighted maximum likelihood, in which observations whose robustly standardized
        transformed value exceeds ``sqrt(chi2_1(quantile))`` get weight 0. This follows the
        reference implementation ``transfo`` in the R package cellWise.

        Args:
            method (Literal str, optional): Method used for the power transformation.
                Can be "boxcox" for Box-Cox, "yeojohnson" for Yeo-Johnson, or "auto"
                for the solution with the lowest objective function. Box-Cox can only be used for
                strictly positive features. Defaults to "yeojohnson".
            standardize (boolean, optional): Whether to standardize the features before and after
                the power transformation. Defaults to True.
            lambda_range (tuple[float, float], optional): Initial search range for lambda; must
                contain 1. If the estimate lies close to a bound, that bound is moved away from 1
                (its distance to 1 is doubled) and the fit is repeated. Defaults to (-4.0, 6.0).
            quantile (float, optional): Quantile used to calculate the weights. Defaults to 0.99.
            nsteps (int, optional): Number of steps used in the reweighted maximum likelihood.
                Defaults to 2.
            rectified (bool, optional): Whether ``transform`` and ``inverse_transform`` use the
                rectified transformation with the fitted lambda instead of the classical one.
                The rectified transformation equals the classical one up to a change point
                (``changepoint_``, determined from the quartiles of the training data) and
                continues linearly beyond it. For lambda < 1 the classical transformation
                compresses the upper tail (and for lambda > 1 the lower tail), which can move
                outliers into the bulk of the transformed data; the rectified version keeps
                them outlying, which helps when outliers are detected after the
                transformation (e.g. with DDC). The estimate of lambda does not depend on this
                option. Note that the rectified transformation of the clean data is not
                exactly the maximum likelihood transformation. Defaults to False, which gives
                the transformation of the paper and of cellWise.

        Attributes:
            lambda_rew (float): The reweighted estimate of lambda.
            method (str): The transformation used ("boxcox" or "yeojohnson"; "auto" is resolved
                during fit).
            changepoint_ (float | None): Change point of the rectified transformation for the
                fitted lambda, on the scale of the (pre-standardized) data, or None when
                lambda equals 1. Only set when ``rectified`` is True.
            location_pre (float): Location (median) subtracted before the Yeo-Johnson
                transformation (only when ``standardize`` is True).
            scale_pre (float): Scale (MAD for Yeo-Johnson, median for Box-Cox) divided by
                before the transformation (only when ``standardize`` is True).
            location_post (float): Mean of the transformed observations with weight 1,
                subtracted after the transformation (only when ``standardize`` is True).
            scale_post (float): Standard deviation of the transformed observations with weight
                1, divided by after the transformation (only when ``standardize`` is True).

        References:
            - Raymaekers, J., & Rousseeuw, P. J. (2024). Transforming variables to central
              normality. Machine Learning, 113(8), 4953-4975.
        """
        self.method = method
        self.standardize = standardize
        self.lambda_range = lambda_range
        self.quantile = quantile
        self.nsteps = nsteps
        self.rectified = rectified
        self.logger = logging.getLogger("RobustPowerTransformer")

    def fit(self, x: np.ndarray):
        """Calculates lambda, the transformation parameter depending on the method.

        Args:
            x (np.ndarray): The data.
        """
        x = np.asarray(x, dtype=float)
        self._get_method(x)
        if not (len(self.lambda_range) == 2 and self.lambda_range[0] < 1 < self.lambda_range[1]):
            raise ValueError("lambda_range must be a pair (low, high) with low < 1 < high.")
        x_sorted = np.sort(x)

        if self.method in ["boxcox", "auto"]:
            fit_boxcox = self._fit_boxcox(x_sorted)
        if self.method in ["yeojohnson", "auto"]:
            fit_yeojohnson = self._fit_yeojohnson(x_sorted)

        if self.method == "auto":
            crit_val_boxcox = self._robnormality(x, "boxcox", fit_boxcox["lambda"])
            crit_val_yeojohnson = self._robnormality(x, "yeojohnson", fit_yeojohnson["lambda"])
            self.method = "boxcox" if crit_val_boxcox < crit_val_yeojohnson else "yeojohnson"

        result = fit_boxcox if self.method == "boxcox" else fit_yeojohnson
        self.lambda_rew = result["lambda"]
        if self.standardize:
            if self.method == "yeojohnson":
                self.location_pre = result["location_pre"]
            self.scale_pre = result["scale_pre"]
            self.location_post = result["location_post"]
            self.scale_post = result["scale_post"]
        if self.rectified:
            self.changepoint_ = result["changepoint"]

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the data using the calculated lambda estimate and the corresponding method.

        Uses the rectified transformation if ``rectified`` is True.

        Args:
            x (np.ndarray): The data.
        """
        x = np.asarray(x, dtype=float)
        if self.method == "boxcox" and np.min(x) <= 0:
            raise ValueError(
                "The data is not strictly positive. Box-Cox transformation cannot be applied."
            )
        x = self._standardize_pre(x)
        x = self._transform_standardized(
            x, self.method, self.lambda_rew, self._fitted_changepoint()
        )
        if self.standardize:
            x = (x - self.location_post) / self.scale_post
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the data back using inverse Yeo-Johnson/Box-Cox. The previously fitted lambda
        estimate and the corresponding method are used (rectified if ``rectified`` is True).

        Args:
            x (np.ndarray): The data.
        """
        x = np.asarray(x, dtype=float)
        if self.standardize:
            x = x * self.scale_post + self.location_post

        changepoint = self._fitted_changepoint()
        if changepoint is None:
            classical = np.ones(x.shape, dtype=bool)
        else:
            changepoint_transf, slope, upper = self._rectified_linear_part(
                changepoint, self.method, self.lambda_rew
            )
            classical = x <= changepoint_transf if upper else x >= changepoint_transf
        xt = np.empty_like(x)
        x_classical = self._handle_bounds(x[classical])
        if self.method == "boxcox":
            xt[classical] = self._inv_transf_boxcox(x_classical, self.lambda_rew)
        else:
            xt[classical] = self._inv_transf_yeojohnson(x_classical, self.lambda_rew)
        if changepoint is not None:
            xt[~classical] = changepoint + (x[~classical] - changepoint_transf) / slope

        if self.standardize:
            if self.method == "boxcox":
                xt = xt * self.scale_pre
            else:
                xt = xt * self.scale_pre + self.location_pre

        return xt

    def _standardize_pre(self, x: np.ndarray) -> np.ndarray:
        """Standardization applied before the power transformation."""
        if not self.standardize:
            return x
        if self.method == "boxcox":
            return x / self.scale_pre
        return (x - self.location_pre) / self.scale_pre

    def _fitted_changepoint(self) -> float | None:
        """Change point used by transform / inverse_transform (None: classical)."""
        return getattr(self, "changepoint_", None) if self.rectified else None

    def _transform_standardized(
        self,
        x: np.ndarray,
        transf: Literal["boxcox", "yeojohnson"],
        my_lambda: float,
        changepoint: float | None = None,
    ) -> np.ndarray:
        """Classical transformation, or the rectified one with the given change point."""
        if transf == "boxcox":
            if changepoint is None:
                return self._transf_boxcox(x, my_lambda)[0]
            return self._transf_boxcox_rectified(x, my_lambda, changepoint=changepoint)[0]
        if changepoint is None:
            return self._transf_yeojohnson(x, my_lambda)[0]
        return self._transf_yeojohnson_rectified(x, my_lambda, changepoint=changepoint)[0]

    def _rectified_linear_part(
        self, changepoint: float, transf: Literal["boxcox", "yeojohnson"], my_lambda: float
    ) -> tuple[float, float, bool]:
        """The linear part of the rectified transformation beyond the change point.

        Returns:
            tuple: the transformed change point, the slope of the linear part, and whether the
            linear part lies above the change point (lambda < 1) or below it (lambda > 1).
        """
        c = np.array([changepoint], dtype=float)
        if transf == "boxcox":
            changepoint_transf = self._transf_boxcox(c, my_lambda)[0][0]
            slope = changepoint ** (my_lambda - 1)
        else:
            changepoint_transf = self._transf_yeojohnson(c, my_lambda)[0][0]
            exponent = my_lambda - 1 if my_lambda < 1 else 1 - my_lambda
            slope = (1 + np.abs(changepoint)) ** exponent
        return changepoint_transf, slope, my_lambda < 1

    def _fit_transformation(
        self,
        x: np.ndarray,
        transf: Literal["boxcox", "yeojohnson"],
    ) -> dict:
        """Fits the Box-Cox or Yeo-Johnson power transformation.

        Args:
            x (np.ndarray): The data, sorted.
            transf: The transformation.

        Returns:
            dict: lambda, standardization constants and (if rectified) the change point.
        """
        location_pre, scale_pre = None, None
        if self.standardize:
            if transf == "boxcox":
                scale_pre = np.median(x)
                x = x / scale_pre
            else:
                location_pre = np.median(x)
                scale_pre = median_abs_deviation(x, scale="normal")
                x = (x - location_pre) / scale_pre

        # Widen the range of lambda when the estimate lies close to one of its bounds, as in
        # cellWise::transfo.
        lambdarange = np.asarray(self.lambda_range, dtype=float)
        while True:
            lambda_raw = self._calculate_lambda_0(x, f"{transf}_rect", lambdarange)
            lambda_rew, weights = self._reweighted_max_likelihood_robust(
                x, lambda_raw, transf, lambdarange, self.quantile, self.nsteps
            )
            distance = np.abs(lambda_rew - lambdarange)
            if np.min(distance) > np.diff(lambdarange)[0] * 0.05:
                break
            lambdarange = 1 + (lambdarange - 1) * (1 + (distance == np.min(distance)))

        changepoint = None
        if self.rectified and lambda_rew != 1.0:
            if transf == "boxcox":
                changepoint = self._get_rectified_boxcox_changepoint(x, lambda_rew)
            else:
                changepoint = self._get_rectified_yeojohnson_changepoint(x, lambda_rew)
        xt = self._transform_standardized(x, transf, lambda_rew, changepoint)

        return {
            "lambda": lambda_rew,
            "location_pre": location_pre,
            "scale_pre": scale_pre,
            "location_post": np.mean(xt[weights]),
            "scale_post": np.std(xt[weights], ddof=1),
            "changepoint": changepoint,
        }

    def _fit_boxcox(self, x: np.ndarray) -> dict:
        """Fits the Box-Cox power transformation (x sorted)."""
        return self._fit_transformation(x, "boxcox")

    def _fit_yeojohnson(self, x: np.ndarray) -> dict:
        """Fits the Yeo-Johnson power transformation (x sorted)."""
        return self._fit_transformation(x, "yeojohnson")

    def _calculate_bounds(self):
        """Calculates the bounds of the range of the power transformation."""

        lower_bound, upper_bound = None, None

        if self.method == "boxcox" and self.lambda_rew > 0.0:
            lower_bound = -1.0 / np.abs(self.lambda_rew)
        elif self.method == "boxcox" and self.lambda_rew < 0.0:
            upper_bound = 1.0 / np.abs(self.lambda_rew)
        elif self.method == "yeojohnson" and self.lambda_rew > 2.0:
            lower_bound = -1.0 / np.abs(self.lambda_rew - 2.0)
        elif self.method == "yeojohnson" and self.lambda_rew < 0.0:
            upper_bound = 1.0 / np.abs(self.lambda_rew)

        return lower_bound, upper_bound

    def _handle_bounds(self, x: np.ndarray):
        if np.size(x) == 0:
            return x
        lower_bound, upper_bound = self._calculate_bounds()

        if lower_bound is not None and np.min(x) < lower_bound:
            new_bound = 0.95 * lower_bound
            self.logger.info(
                f"Some (standardized) values in x are below the range of the {self.method} "
                f"transformation. These values were replaced by {new_bound} so they "
                "can be transformed back."
            )
            x = np.where(x < lower_bound, new_bound, x)

        if upper_bound is not None and np.max(x) > upper_bound:
            new_bound = 0.95 * upper_bound
            self.logger.info(
                f"Some (standardized) values in x are above the range of the {self.method} "
                f"transformation. These values were replaced by {new_bound} so they "
                "can be transformed back."
            )
            x = np.where(x > upper_bound, new_bound, x)

        return x

    def _transf_boxcox_rectified(
        self,
        x: np.ndarray,
        my_lambda: float,
        standardize_too: bool = False,
        changepoint: float | None = None,
    ):
        """Rectified Box-Cox transformation. Without a change point, it is computed from x,
        which must then be sorted."""

        if np.min(x) <= 0:
            raise ValueError(
                "Data values should be strictly positive for a Box-Cox transformation."
            )
        xt = np.zeros_like(x)

        if my_lambda == 1.0:
            xt = x - 1.0
        else:
            if changepoint is None:
                changepoint = self._get_rectified_boxcox_changepoint(x, my_lambda)

            if my_lambda > 1.0:
                mask = x < changepoint
            else:
                mask = x > changepoint
            if my_lambda == 0:
                xt[~mask] = np.log(x[~mask])
                changepoint_transf = np.log(changepoint)
                xt[mask] = changepoint_transf + (x[mask] - changepoint) / changepoint
            else:
                xt[~mask] = (x[~mask] ** my_lambda - 1) / my_lambda
                changepoint_transf = (changepoint**my_lambda - 1) / my_lambda
                xt[mask] = changepoint_transf + (x[mask] - changepoint) * changepoint ** (
                    my_lambda - 1
                )

        if standardize_too:
            loc_scale = HuberOneStepM().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _get_rectified_boxcox_changepoint(
        self, x: np.ndarray, my_lambda: float, factor: float = 1.5, eps: float = 1e-5
    ) -> float:
        """Get C_u or C_l for the rectified Box-Cox transform."""
        n = len(x)
        Q1 = x[int(np.ceil(n / 4.0) - 1.0)]
        Q3 = x[int(n - np.ceil(n / 4.0))]

        if my_lambda < 1:
            changepoint = self._transf_boxcox(Q3, my_lambda)[0] * factor
        elif my_lambda > 1:
            changepoint = self._transf_boxcox(Q1, my_lambda)[0] * factor
        else:
            raise ValueError("There is no changepoint for lambda equals 1.")

        if my_lambda < 0.0:
            changepoint = min(changepoint, np.abs(1.0 / my_lambda) - eps)
        elif my_lambda > 0.0:
            changepoint = max(changepoint, -1.0 / my_lambda + eps)

        changepoint = self._inv_transf_boxcox(changepoint, my_lambda)
        changepoint = min(max(changepoint, x[0]), x[n - 1])

        return changepoint

    def _transf_boxcox(self, x: np.ndarray, my_lambda: float, standardize_too: bool = False):
        """Classical Box-Cox transformation."""
        if np.min(x) <= 0:
            raise ValueError(
                "Data values should be strictly positive for a Box-Cox transformation."
            )
        if my_lambda == 0:
            xt = np.log(x)
        else:
            xt = (x**my_lambda - 1) / my_lambda

        if standardize_too:
            loc_scale = HuberOneStepM().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _transf_yeojohnson(self, x: np.ndarray, my_lambda: float, standardize_too: bool = False):
        """Classical Yeo-Johnson transformation."""

        positive_mask = x >= 0.0
        xt = np.zeros_like(x)

        if my_lambda == 0.0:
            xt[positive_mask] = np.log(1.0 + x[positive_mask])
            xt[~positive_mask] = -((1 - x[~positive_mask]) ** (2 - my_lambda) - 1) / (2 - my_lambda)
        elif my_lambda == 2.0:
            xt[positive_mask] = ((1 + x[positive_mask]) ** my_lambda - 1) / my_lambda
            xt[~positive_mask] = -np.log(1 - x[~positive_mask])
        else:
            xt[positive_mask] = ((1 + x[positive_mask]) ** my_lambda - 1) / my_lambda
            xt[~positive_mask] = -((1 - x[~positive_mask]) ** (2 - my_lambda) - 1) / (2 - my_lambda)

        if standardize_too:
            loc_scale = HuberOneStepM().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _transf_yeojohnson_rectified(
        self,
        x: np.ndarray,
        my_lambda: float,
        standardize_too: bool = False,
        changepoint: float | None = None,
    ):
        """Rectified Yeo-Johnson transformation. Without a change point, it is computed from x,
        which must then be sorted."""

        xt = np.zeros_like(x)

        if my_lambda == 1.0:
            xt = x

        else:
            if changepoint is None:
                changepoint = self._get_rectified_yeojohnson_changepoint(x, my_lambda)
            positive_mask = x >= 0.0
            negative_mask = x < 0.0

            if my_lambda > 1.0:
                mask_lower = x < changepoint
                mask_upper = (changepoint <= x) & (x < 0)
                xt[positive_mask] = ((1 + x[positive_mask]) ** my_lambda - 1.0) / my_lambda
                if my_lambda == 2:
                    xt[mask_upper] = -np.log(1 - x[mask_upper])
                    xt[mask_lower] = -np.log(1 - changepoint) + (
                        x[mask_lower] - changepoint
                    ) * 1 / (1 - changepoint)
                else:
                    xt[mask_upper] = -((1 - x[mask_upper]) ** (2 - my_lambda) - 1) / (2 - my_lambda)
                    changepoint_transf = self._transf_yeojohnson(changepoint, my_lambda)[0]
                    xt[mask_lower] = changepoint_transf + (x[mask_lower] - changepoint) * (
                        1 + np.abs(changepoint)
                    ) ** (1 - my_lambda)
            else:
                mask_upper = x > changepoint
                mask_lower = (x >= 0) & (x <= changepoint)
                xt[negative_mask] = -((1 - x[negative_mask]) ** (2 - my_lambda) - 1) / (
                    2 - my_lambda
                )
                if my_lambda == 0.0:
                    xt[mask_lower] = np.log(1 + x[mask_lower])
                    xt[mask_upper] = np.log(1 + changepoint) + (x[mask_upper] - changepoint) * 1 / (
                        1 + changepoint
                    )
                else:
                    xt[mask_lower] = ((1 + x[mask_lower]) ** my_lambda - 1) / my_lambda
                    changepoint_transf = self._transf_yeojohnson(changepoint, my_lambda)[0]
                    xt[mask_upper] = changepoint_transf + (x[mask_upper] - changepoint) * (
                        1 + np.abs(changepoint)
                    ) ** (my_lambda - 1)

        if standardize_too:
            loc_scale = HuberOneStepM().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _get_rectified_yeojohnson_changepoint(
        self, x: np.ndarray, my_lambda: float, factor: float = 1.5, eps: float = 1e-5
    ) -> float:
        """Get C_u or C_l for the rectified Box-Cox transformation."""
        n = len(x)
        Q1 = x[int(np.ceil(n / 4.0) - 1.0)]
        Q3 = x[int(n - np.ceil(n / 4.0))]

        if my_lambda < 1:
            changepoint = self._transf_yeojohnson(Q3, my_lambda)[0] * factor
        elif my_lambda > 1:
            changepoint = self._transf_yeojohnson(Q1, my_lambda)[0] * factor
        else:
            raise ValueError("There is no changepoint for lambda equals 1.")

        if my_lambda < 0.0:
            changepoint = min(changepoint, np.abs(1.0 / my_lambda) - eps)
        elif my_lambda > 2.0:
            changepoint = max(changepoint, (1.0 / (2 - my_lambda)) + eps)

        changepoint = self._inv_transf_yeojohnson(changepoint, my_lambda)
        changepoint = min(max(changepoint, x[0]), x[n - 1])

        return changepoint

    def _inv_transf_boxcox(self, x: np.ndarray, my_lambda: float) -> np.ndarray:
        """Classical Box-Cox transformation inversed."""
        if my_lambda == 0:
            xt = np.exp(x)
        else:
            xt = (x * my_lambda + 1) ** (1.0 / my_lambda)
        return xt

    def _inv_transf_yeojohnson(self, x: np.ndarray, my_lambda: float) -> np.ndarray:
        """Classical Yeo-Johnson transformation inversed."""

        positive_mask = x >= 0.0
        xt = np.zeros_like(x)

        if my_lambda == 0.0:
            xt[positive_mask] = np.exp(x[positive_mask]) - 1.0
            xt[~positive_mask] = 1.0 - (1 + (my_lambda - 2) * x[~positive_mask]) ** (
                1 / (2 - my_lambda)
            )
        elif my_lambda == 2.0:
            xt[positive_mask] = (my_lambda * x[positive_mask] + 1) ** (1 / my_lambda) - 1
            xt[~positive_mask] = 1 - np.exp(-x[~positive_mask])
        else:
            xt[positive_mask] = (my_lambda * x[positive_mask] + 1) ** (1 / my_lambda) - 1
            xt[~positive_mask] = 1.0 - (1 + (my_lambda - 2) * x[~positive_mask]) ** (
                1 / (2 - my_lambda)
            )

        return xt

    def _robnormality(
        self,
        x: np.ndarray,
        transf: Literal["boxcox_rect", "yeojohnson_rect", "boxcox", "yeojohnson"],
        my_lambda: float,
    ) -> float:
        """Objective function of Equation (5) in Raymaekers, J., & Rousseeuw, P. J. (2024)."""

        if not np.all(np.diff(x) >= 0):
            x = np.sort(x)

        if transf == "boxcox_rect":
            x = self._transf_boxcox_rectified(x, my_lambda)[0]
        elif transf == "yeojohnson_rect":
            x = self._transf_yeojohnson_rectified(x, my_lambda)[0]
        elif transf == "boxcox":
            x = self._transf_boxcox(x, my_lambda)[0]
        elif transf == "yeojohnson":
            x = self._transf_yeojohnson(x, my_lambda)[0]
        else:
            raise NotImplementedError("Other transformations not yet implemented.")

        x = x[~np.isnan(x)]
        n = len(x)

        loc_scale = HuberOneStepM().fit(x)
        x = (x - loc_scale.location) / np.where(loc_scale.scale == 0, 1, loc_scale.scale)

        theo_quantile = norm.ppf((np.arange(1, n + 1) - 1 / 3) / (n + 1 / 3))
        obj = TukeyBisquare(c=0.5).rho(x - theo_quantile)
        crit = np.sum(obj)

        return crit

    def _calculate_lambda_0(
        self,
        x: np.ndarray,
        transf: Literal["boxcox_rect", "yeojohnson_rect"] = "boxcox_rect",
        lambdarange: list[float] = [-4.0, 6.0],
    ):
        """Computes the initial estimate for lambda by optimizing the objective function."""
        lambda_0 = minimize_scalar(
            lambda lambdatemp: self._robnormality(x, transf, lambdatemp),
            bounds=lambdarange,
            method="bounded",
        )

        return lambda_0.x

    def _reweighted_max_likelihood_robust(
        self,
        x: np.ndarray,
        lambda_raw: float,
        transf: Literal["boxcox", "yeojohnson"],
        lambdarange: list[float] = [-4.0, 6.0],
        quantile: float = 0.99,
        nsteps: int = 2,
    ):
        """Computes the reweighted estimate for lambda by performing reweighted ML.

        Starting from the rectified transformation with the initial lambda, each step estimates
        lambda by maximum likelihood on the observations with weight 1, transforms all
        observations with the new lambda and standardizes them robustly (Huber M-estimator of
        location and scale, as ``YJ`` / ``BC`` in cellWise). Observations with an absolute
        standardized value above ``sqrt(chi2_1(quantile))`` get weight 0.

        Args:
            x (np.ndarray): The (pre-standardized) data, sorted.

        Returns:
            tuple[float, np.ndarray]: the reweighted lambda and the final weights (boolean
            array), computed after the last step.
        """
        if nsteps < 1:
            raise ValueError("nsteps must be at least 1.")
        cutoff = np.sqrt(chi2.ppf(quantile, 1))
        if transf == "boxcox":
            zt = self._transf_boxcox_rectified(x, lambda_raw, standardize_too=True)[1]
        elif transf == "yeojohnson":
            zt = self._transf_yeojohnson_rectified(x, lambda_raw, standardize_too=True)[1]
        else:
            raise NotImplementedError("Wrong transformation.")

        for _ in range(nsteps):
            w = np.abs(zt) <= cutoff
            lambda_rew = self._estimate_max_likelihood(x[w], transf, lambdarange)
            if transf == "boxcox":
                zt = self._transf_boxcox(x, lambda_rew, standardize_too=True)[1]
            else:
                zt = self._transf_yeojohnson(x, lambda_rew, standardize_too=True)[1]

        return lambda_rew, np.abs(zt) <= cutoff

    def _estimate_max_likelihood(
        self,
        x: np.ndarray,
        transf: Literal["boxcox", "yeojohnson"],
        lambdarange: list[float] = [-4.0, 6.0],
    ):
        """Computes an estimate for lambda by using ML."""
        n = len(x)

        if transf == "boxcox":

            def obj_func(lambdatemp):
                x_bc = self._transf_boxcox(x, lambdatemp)[0]
                mu = np.mean(x_bc)
                sigma2 = np.mean((x_bc - mu) ** 2)
                return (n / 2) * np.log(sigma2) - (lambdatemp - 1) * np.sum(np.log(x))

        elif transf == "yeojohnson":

            def obj_func(lambdatemp):
                x_yj = self._transf_yeojohnson(x, lambdatemp)[0]
                mu = np.mean(x_yj)
                sigma2 = np.mean((x_yj - mu) ** 2)
                return (n / 2) * np.log(sigma2) - (lambdatemp - 1) * np.sum(
                    np.sign(x) * np.log(1 + np.abs(x))
                )

        else:
            raise NotImplementedError("Wrong transformation.")

        lambda_est = minimize_scalar(obj_func, bounds=lambdarange, method="bounded").x

        return lambda_est

    def _get_method(self, x: np.ndarray):
        if self.method == "boxcox":
            if np.min(x) <= 0:
                raise ValueError("The data is not strictly positive. Box-Cox cannot be applied.")
        elif self.method == "auto":
            if np.min(x) <= 0:
                self.method = "yeojohnson"
        elif self.method != "yeojohnson":
            raise ValueError(
                'The only supported methods of transformation are "boxcox", "yeojohnson" or "auto".'
            )
        return self
