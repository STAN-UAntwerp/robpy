import numpy as np
from robpy.utils.rho import TukeyBisquare
from scipy.stats import median_abs_deviation, norm, chi2
from scipy.optimize import minimize_scalar
from robpy.univariate.hubers_m_est import UnivariateHuberMEstimator1step

# TODO : first apply DataCleaner (wants a pd.DataFrame and returns a pd.Dataframe for now)

# per column/variable (univariate):
# take only the non NAs
# find type: check whether min(x) positive (for BC)
# robust or not? two versions
# if robust: rewML
# if not robust: estML

# params:
# X: np or pd
# standardize: True or False
# type: bc, yj, best_obj
# robust: True or False
# quantile: for weights in rew (default = 0.99)
# n_rew_steps: int


# First needed: Huber M-estimator of location and scale (univariate)
# -> new univariate RobustScaleEstimator


def transfo(
    x: np.array,
    type: str = "BC",
    robust: bool = True,
    standardize: bool = True,
    quant: float = 0.99,
    nsteps: int = 2,
):
    lambdarange = np.array([-4.0, 6.0])

    type = get_type(x, type)

    if robust:
        x = np.sort(x)
        if type in ["BC", "best_objective"]:
            converged = False
            while not converged:
                if standardize:
                    x = x / np.median(x)
                lambda_raw = calculate_lambda_0(x, "BCr", lambdarange)
                lambda_rew = rew_ML_rob(
                    x, lambda_raw, "BC", lambdarange, standardize, quant, nsteps
                )
                converged = np.min(np.abs(lambda_rew - lambdarange)) > np.diff(lambdarange) * 0.05
                if not converged:
                    lambdarange = 1 + (lambdarange - 1) * (
                        1 + (abs(lambda_rew - lambdarange) == min(abs(lambda_rew - lambdarange)))
                    )
            if type == "best_objective":
                lambda_BC = lambda_rew
        if type in ["YJ", "best_objective"]:
            converged = False
            while not converged:
                if standardize:
                    x = (x - np.median(x)) / median_abs_deviation(x, scale="normal")
                lambda_raw = calculate_lambda_0(x, "YJr", lambdarange)
                lambda_rew = rew_ML_rob(
                    x, lambda_raw, "YJ", lambdarange, standardize, quant, nsteps
                )
                converged = np.min(np.abs(lambda_rew - lambdarange)) > np.diff(lambdarange) * 0.05
                if not converged:
                    lambdarange = 1 + (lambdarange - 1) * (
                        1 + (abs(lambda_rew - lambdarange) == min(abs(lambda_rew - lambdarange)))
                    )
            if type == "best_objective":
                lambda_YJ = lambda_rew
        if type == "best_objective":
            # critical values needed.
            0

    else:
        raise NotImplementedError("Non-robust version not yet implemented.")

    return lambda_rew


def transf_BoxCox_rectified(x: np.array, my_lambda: float, standardize: bool = False):
    """Rectified BoxCox transformation"""

    if np.min(x) <= 0:
        raise ValueError("Data values should be strictly positive for a BoxCox transformation.")
    xt = np.zeros_like(x)

    if my_lambda == 1.0:
        xt = x - 1.0
    else:
        chg = get_changepoint_rect_BC(x, my_lambda)

        if my_lambda > 1.0:
            mask = x < chg
        else:
            mask = x > chg
        if my_lambda == 0:
            xt[~mask] = np.log(x[~mask])
            chgt = np.log(chg)
            xt[mask] = chgt + (x[mask] - chg) / chg
        else:
            xt[~mask] = (x[~mask] ** my_lambda - 1) / my_lambda
            chgt = (chg**my_lambda - 1) / my_lambda
            xt[mask] = chgt + (x[mask] - chg) * chg ** (my_lambda - 1)

    if standardize:
        loc_scale = UnivariateHuberMEstimator1step().fit(xt)
        zt = (xt - loc_scale.location) / loc_scale.scale
    else:
        zt = None

    return xt, zt


def get_changepoint_rect_BC(
    x: np.array, my_lambda: float, fac: float = 1.5, eps: float = 1e-5
) -> float:
    """Get C_u or C_l for the rectified BoxCox transform"""
    n = len(x)
    Q1 = x[int(np.ceil(n / 4.0) - 1.0)]
    Q3 = x[int(n - np.ceil(n / 4.0))]

    if my_lambda < 1:
        chg = transf_BoxCox(Q3, my_lambda)[0] * fac
    elif my_lambda > 1:
        chg = transf_BoxCox(Q1, my_lambda)[0] * fac
    else:
        raise ValueError("There is no changepoint for lambda equals 1.")

    if my_lambda < 0.0:
        chg = min(chg, np.abs(1.0 / my_lambda) - eps)
    elif my_lambda > 0.0:
        chg = max(chg, -1.0 / my_lambda + eps)

    chg = inv_transf_BoxCox(chg, my_lambda)
    chg = min(max(chg, x[0]), x[n - 1])

    return chg


def transf_BoxCox(x: np.array, my_lambda: float, standardize: bool = False):
    """Classical BoxCox transformation"""
    if np.min(x) <= 0:
        raise ValueError("Data values should be strictly positive for a BoxCox transformation.")
    if my_lambda == 0:
        xt = np.log(x)
    else:
        xt = (x**my_lambda - 1) / my_lambda

    if standardize:
        loc_scale = UnivariateHuberMEstimator1step().fit(xt)
        zt = (xt - loc_scale.location) / loc_scale.scale
    else:
        zt = None

    return xt, zt


def transf_YeoJohnson(x: np.array, my_lambda: float, standardize: bool = False):
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

    if standardize:
        loc_scale = UnivariateHuberMEstimator1step().fit(xt)
        zt = (xt - loc_scale.location) / loc_scale.scale
    else:
        zt = None

    return xt, zt


def transf_YeoJohnson_rectified(x: np.array, my_lambda: float, standardize: bool = False):
    """Rectified Yeo-Johnson transformation"""

    xt = np.zeros_like(x)

    if my_lambda == 1.0:
        xt = x

    else:
        chg = get_changepoint_rect_YJ(x, my_lambda)
        positive_mask = x >= 0.0
        negative_mask = x < 0.0

        if my_lambda > 1.0:
            mask_lower = x < chg
            mask_upper = (chg <= x) & (x < 0)
            xt[positive_mask] = ((1 + x[positive_mask]) ** my_lambda - 1.0) / my_lambda
            if my_lambda == 2:
                xt[mask_upper] = -np.log(1 - x[mask_upper])
                xt[mask_lower] = -np.log(1 - chg) + (x[mask_lower] - chg) * 1 / (1 - chg)
            else:
                xt[mask_upper] = -((1 - x[mask_upper]) ** (2 - my_lambda) - 1) / (2 - my_lambda)
                chgt = transf_YeoJohnson(chg, my_lambda)[0]
                xt[mask_lower] = chgt + (x[mask_lower] - chg) * (1 + np.abs(chg)) ** (1 - my_lambda)
        else:
            mask_upper = x > chg
            mask_lower = (x >= 0) & (x <= chg)
            xt[negative_mask] = -((1 - x[negative_mask]) ** (2 - my_lambda) - 1) / (2 - my_lambda)
            if my_lambda == 0.0:
                xt[mask_lower] = np.log(1 + x[mask_lower])
                xt[mask_upper] = np.log(1 + chg) + (x[mask_upper] - chg) * 1 / (1 + chg)
            else:
                xt[mask_lower] = ((1 + x[mask_lower]) ** my_lambda - 1) / my_lambda
                chgt = transf_YeoJohnson(chg, my_lambda)[0]
                xt[mask_upper] = chgt + (x[mask_upper] - chg) * (1 + np.abs(chg)) ** (my_lambda - 1)

    if standardize:
        loc_scale = UnivariateHuberMEstimator1step().fit(xt)
        zt = (xt - loc_scale.location) / loc_scale.scale
    else:
        zt = None

    return xt, zt


def get_changepoint_rect_YJ(
    x: np.array, my_lambda: float, fac: float = 1.5, eps: float = 1e-5
) -> float:
    """Get C_u or C_l for the rectified BoxCox transform"""
    n = len(x)
    Q1 = x[int(np.ceil(n / 4.0) - 1.0)]
    Q3 = x[int(n - np.ceil(n / 4.0))]

    if my_lambda < 1:
        chg = transf_YeoJohnson(Q3, my_lambda)[0] * fac
    elif my_lambda > 1:
        chg = transf_YeoJohnson(Q1, my_lambda)[0] * fac
    else:
        raise ValueError("There is no changepoint for lambda equals 1.")

    if my_lambda < 0.0:
        chg = min(chg, np.abs(1.0 / my_lambda) - eps)
    elif my_lambda > 2.0:
        chg = max(chg, (1.0 / (2 - my_lambda)) + eps)

    chg = inv_transf_YeoJohnson(chg, my_lambda)
    chg = min(max(chg, x[0]), x[n - 1])

    return chg


def inv_transf_BoxCox(x: np.array, my_lambda: float) -> np.array:
    """Classical BoxCox transformation inversed"""
    if my_lambda == 0:
        xt = np.exp(x)
    else:
        xt = (x * my_lambda + 1) ** (1.0 / my_lambda)
    # TODO: in R kan dit ook nog extra de xt schalen...
    return xt


def inv_transf_YeoJohnson(x: np.array, my_lambda: float) -> np.array:
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

    # TODO: in R kan dit ook nog extra de xt schalen...
    return xt


def robnormality(x: np.array, transf: str, my_lambda: float) -> float:
    """Objective function of Equation (5) in Raymaekers & Rousseeuw (2021)
    Transforming variables to central normality

    Args:
        X (np.array): ordered np.array
    """

    if not np.all(np.diff(x) >= 0):
        x = np.sort(x)

    if transf == "BCr":
        x = transf_BoxCox_rectified(x, my_lambda)[0]
    elif transf == "YJr":
        x = transf_YeoJohnson_rectified(x, my_lambda)[0]
    else:
        raise NotImplementedError("Other transformations not yet implemented.")

    x = x[~np.isnan(x)]
    n = len(x)

    loc_scale = UnivariateHuberMEstimator1step().fit(x)
    x = (x - loc_scale.location) / loc_scale.scale

    theo_quant = norm.ppf((np.arange(1, n + 1) - 1 / 3) / (n + 1 / 3))
    obj = TukeyBisquare(c=0.5).rho(x - theo_quant)
    crit = np.sum(obj)

    return crit


def calculate_lambda_0(
    x: np.array, transf: str = "BCr", lambdarange: tuple[float, float] = [-4.0, 6.0]
):

    lambda_0 = minimize_scalar(
        lambda lambdatemp: robnormality(x, transf, lambdatemp), bounds=lambdarange, method="bounded"
    )

    return lambda_0.x


def rew_ML_rob(
    x: np.array,
    lambda_0: float,
    type: str,
    lambdarange: tuple[float, float] = [-4.0, 6.0],
    standardize: bool = True,
    quant: float = 0.99,
    nsteps: int = 2,
):

    lambda_rew = lambda_0

    for _ in range(nsteps):
        if type == "BC":
            zt = transf_BoxCox(x, lambda_rew, standardize=True)[1]
        elif type == "YJ":
            zt = transf_YeoJohnson(x, lambda_rew, standardize=True)[1]
        else:
            raise ValueError("Invalid type!")
        w = np.abs(zt) <= np.sqrt(chi2.ppf(quant, 1))
        lambda_rew = estML(x[w], type, lambdarange)

    return lambda_rew


def estML(x: np.array, transf: str, lambdarange: tuple[float, float] = [-4.0, 6.0]):

    n = len(x)

    if transf == "BC":

        def obj_func(lambdatemp):
            x_bc = transf_BoxCox(x, lambdatemp)[0]
            mu = np.mean(x_bc)
            sigma2 = np.mean((x_bc - mu) ** 2)
            return (n / 2) * np.log(sigma2) - (lambdatemp - 1) * np.sum(np.log(x))

    elif transf == "YJ":

        def obj_func(lambdatemp):
            x_yj = transf_YeoJohnson(x, lambdatemp)[0]
            mu = np.mean(x_yj)
            sigma2 = np.mean((x_yj - mu) ** 2)
            return (n / 2) * np.log(sigma2) - (lambdatemp - 1) * np.sum(
                np.sign(x) * np.log(1 + np.abs(x))
            )

    else:
        raise NotImplementedError("Wrong transformation.")

    lambda_est = minimize_scalar(obj_func, bounds=lambdarange, method="bounded").x

    return lambda_est


def get_type(x: np.array, type: str) -> str:

    if type == "BC":
        if np.min(x) <= 0:
            raise ValueError("The data is not strictly positive. Box-Cox cannot be applied.")
        return type
    elif type == "best_objective":
        if np.min(x) <= 0:
            return "YJ"
        return type
    elif type == "YJ":
        return type
    else:
        raise ValueError("The only supported types of transformation are BC, YJ or best_objective.")


# TODO: als je weights wil returnen, dan moet je de originele orde onthouden
