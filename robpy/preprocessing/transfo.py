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
from robpy.univariate import HuberOneStepMEstimator


class RobustPowerTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        method: Literal["boxcox", "yeojohnson", "auto"] = "yeojohnson",
        standardize: bool = True,
        lambda_range: tuple[float, float] = (-4.0, 6.0),
        quantile: float = 0.99,
        nsteps: int = 2,
    ):
        """
        Apply a robust power transformation using reweighted maximum likelihood to transform the
        features closer to normality. Uses the Box-Cox or the Yeo-Johnson transformation.

        Args:
            method (Literal str, optional): method used for the power transformation.
                Can be "boxcox" for Box-Cox, "yeojohnson" for Yeo-Johnson, or "auto"
                for best objective solution. Box-Cox can only be used for strictly
                positive features. Defaults to "auto".
            standardize (boolean, optional): whether to standardize the features before and after
                the power transformation. Defaults to True.
            quantile (float, optional): quantile used to calculate the weights. Defaults to 0.99.
            nsteps (int, optional): number of steps used in the reweighted maximum likelihood.
                Defaults to 2.
        """
        self.method = method
        self.standardize = standardize
        self.lambda_range = list(lambda_range)
        self.quantile = quantile
        self.nsteps = nsteps
        self.logger = logging.getLogger("RobustPowerTransformer")

    def fit(self, x: np.ndarray):
        """Calculates lambda, the transformation parameter depending on the method.

        Args:
            x (np.array): data.
        """

        self._get_method(x)
        x_sorted = np.sort(x)

        if self.method in ["boxcox", "auto"]:
            lambda_boxcox, mu_rew_boxcox, sd_rew_boxcox, scale_boxcox = self._fit_boxcox(x_sorted)
            if self.method == "auto":
                crit_val_boxcox = self._robnormality(x, "boxcox", lambda_boxcox)

        if self.method in ["yeojohnson", "auto"]:
            (
                lambda_yeojohnson,
                mu_rew_yeojohnson,
                sd_rew_yeojohnson,
                loc_yeojohnson,
                scale_yeojohnson,
            ) = self._fit_yeojohnson(x_sorted)
            if self.method == "auto":
                crit_val_yeojohnson = self._robnormality(x, "yeojohnson", lambda_yeojohnson)

        if self.method == "auto":
            if crit_val_boxcox < crit_val_yeojohnson:
                self.method = "boxcox"
            else:
                self.method = "yeojohnson"

        if self.method == "boxcox":
            if self.standardize:
                self.scale_pre = scale_boxcox
                self.location_post = mu_rew_boxcox
                self.scale_post = sd_rew_boxcox
            self.lambda_rew = lambda_boxcox
        else:  # yeojohnson
            if self.standardize:
                self.location_pre = loc_yeojohnson
                self.scale_pre = scale_yeojohnson
                self.location_post = mu_rew_yeojohnson
                self.scale_post = sd_rew_yeojohnson
            self.lambda_rew = lambda_yeojohnson

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the data using the calculated lambda estimate and the corresponding method.

        Args:
            x (np.array): data.
        """

        if self.method == "boxcox":
            if np.min(x) <= 0:
                raise ValueError(
                    "The data is not strictly positive. Box-Cox transformation "
                    "cannot be applied."
                )
            if self.standardize:
                x = x / self.scale_pre
            x = self._transf_boxcox(x, self.lambda_rew)[0]
            if self.standardize:
                x = (x - self.location_post) / self.scale_post
        else:  # yeojohnson
            if self.standardize:
                x = (x - self.location_pre) / self.scale_pre
            x = self._transf_yeojohnson(x, self.lambda_rew)[0]
            if self.standardize:
                x = (x - self.location_post) / self.scale_post

        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Transforms the data back using inverse Yeo-Johnson/Box-cox, the previously fitted lambda
        estimate and the corresponding method are used.

        Args:
            x (np.array): data.
        """

        if self.method == "boxcox":
            if self.standardize:
                x = x * self.scale_post + self.location_post
            x = self._handle_bounds(x)
            x = self._inv_transf_boxcox(x, self.lambda_rew)
            if self.standardize:
                x = x * self.scale_pre

        else:  # yeojohnson
            if self.standardize:
                x = x * self.scale_post + self.location_post
            x = self._handle_bounds(x)
            x = self._inv_transf_yeojohnson(x, self.lambda_rew)
            if self.standardize:
                x = x * self.scale_pre + self.location_pre

        return x

    def _fit_boxcox(self, x: np.ndarray):
        """fits the Box-Cox power transformation

        Args:
            x (np.array): data"""
        lambdarange = self.lambda_range
        converged = False
        while not converged:
            if self.standardize:
                scale_boxcox = np.median(x)
                x = x / scale_boxcox
            else:
                scale_boxcox = None
            lambda_raw = self._calculate_lambda_0(x, "boxcox_rect", lambdarange)
            lambda_rew, mu_rew_boxcox, sd_rew_boxcox = self._reweighted_max_likelihood_robust(
                x,
                lambda_raw,
                "boxcox",
                lambdarange,
                self.quantile,
                self.nsteps,
            )
            converged = np.min(np.abs(lambda_rew - lambdarange)) > np.diff(lambdarange) * 0.05
            if not converged:
                lambdarange = 1 + (lambdarange - 1) * (
                    1 + (abs(lambda_rew - lambdarange) == min(abs(lambda_rew - lambdarange)))
                )

        return lambda_rew, mu_rew_boxcox, sd_rew_boxcox, scale_boxcox

    def _fit_yeojohnson(self, x: np.ndarray):
        """fits the Yeo-Johnson power transformation

        Args:
            x (np.array): data"""
        lambdarange = self.lambda_range
        converged = False
        while not converged:
            if self.standardize:
                loc_yeojohnson = np.median(x)
                scale_yeojohnson = median_abs_deviation(x, scale="normal")
                x = (x - loc_yeojohnson) / scale_yeojohnson
            else:
                loc_yeojohnson, scale_yeojohnson = None, None
            lambda_raw = self._calculate_lambda_0(x, "yeojohnson_rect", lambdarange)
            (
                lambda_rew,
                mu_rew_yeojohnson,
                sd_rew_yeojohnson,
            ) = self._reweighted_max_likelihood_robust(
                x,
                lambda_raw,
                "yeojohnson",
                lambdarange,
                self.quantile,
                self.nsteps,
            )
            converged = np.min(np.abs(lambda_rew - lambdarange)) > np.diff(lambdarange) * 0.05
            if not converged:
                lambdarange = 1 + (lambdarange - 1) * (
                    1 + (abs(lambda_rew - lambdarange) == min(abs(lambda_rew - lambdarange)))
                )

        return lambda_rew, mu_rew_yeojohnson, sd_rew_yeojohnson, loc_yeojohnson, scale_yeojohnson

    def _calculate_bounds(self):
        """Calculates the bounds of the range of the power transformation"""

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
        self, x: np.ndarray, my_lambda: float, standardize_too: bool = False
    ):
        """Rectified BoxCox transformation"""

        if np.min(x) <= 0:
            raise ValueError("Data values should be strictly positive for a BoxCox transformation.")
        xt = np.zeros_like(x)

        if my_lambda == 1.0:
            xt = x - 1.0
        else:
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
            loc_scale = HuberOneStepMEstimator().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _get_rectified_boxcox_changepoint(
        self, x: np.ndarray, my_lambda: float, factor: float = 1.5, eps: float = 1e-5
    ) -> float:
        """Get C_u or C_l for the rectified BoxCox transform"""
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
        """Classical BoxCox transformation"""
        if np.min(x) <= 0:
            raise ValueError("Data values should be strictly positive for a BoxCox transformation.")
        if my_lambda == 0:
            xt = np.log(x)
        else:
            xt = (x**my_lambda - 1) / my_lambda

        if standardize_too:
            loc_scale = HuberOneStepMEstimator().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _transf_yeojohnson(self, x: np.ndarray, my_lambda: float, standardize_too: bool = False):
        """Classical Yeo-Johnson transformation"""

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
            loc_scale = HuberOneStepMEstimator().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _transf_yeojohnson_rectified(
        self, x: np.ndarray, my_lambda: float, standardize_too: bool = False
    ):
        """Rectified Yeo-Johnson transformation"""

        xt = np.zeros_like(x)

        if my_lambda == 1.0:
            xt = x

        else:
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
            loc_scale = HuberOneStepMEstimator().fit(xt)
            zt = (xt - loc_scale.location) / loc_scale.scale
        else:
            zt = None

        return xt, zt

    def _get_rectified_yeojohnson_changepoint(
        self, x: np.ndarray, my_lambda: float, factor: float = 1.5, eps: float = 1e-5
    ) -> float:
        """Get C_u or C_l for the rectified BoxCox transform"""
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
        """Classical BoxCox transformation inversed"""
        if my_lambda == 0:
            xt = np.exp(x)
        else:
            xt = (x * my_lambda + 1) ** (1.0 / my_lambda)
        return xt

    def _inv_transf_yeojohnson(self, x: np.ndarray, my_lambda: float) -> np.ndarray:
        """Classical Yeo-Johnson transformation inversed"""

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
        """Objective function of Equation (5) in Raymaekers & Rousseeuw (2021)
        Transforming variables to central normality
        """

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

        loc_scale = HuberOneStepMEstimator().fit(x)
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
        """Computes the reweighted estimate for lambda by performing reweighted ML."""

        if transf == "boxcox":
            zt = self._transf_boxcox_rectified(x, lambda_raw, standardize_too=True)[1]
        elif transf == "yeojohnson":
            zt = self._transf_yeojohnson_rectified(x, lambda_raw, standardize_too=True)[1]

        for _ in range(nsteps):
            w = np.abs(zt) <= np.sqrt(chi2.ppf(quantile, 1))
            lambda_rew = self._estimate_max_likelihood(x[w], transf, lambdarange)
            if transf == "boxcox":
                xt = self._transf_boxcox(x, lambda_rew, standardize_too=False)[0]
            elif transf == "yeojohnson":
                xt = self._transf_yeojohnson(x, lambda_rew, standardize_too=False)[0]
            zt = (xt - np.mean(xt)) / np.std(xt)

        mu_rew = np.mean(xt[w])
        sd_rew = np.std(xt[w])

        return lambda_rew, mu_rew, sd_rew

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
