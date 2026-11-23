from __future__ import annotations

import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import chi2, norm
from sklearn.exceptions import NotFittedError

from robpy.covariance.cellmcd import CellMCD
from robpy.covariance.utils.cellmcd_utils import objective_function
from robpy.outliers.ddc import get_custom_cmap
from robpy.regression.base import RobustRegression
from robpy.regression.lts import get_correction_factor, get_correction_factor_reweighting
from robpy.univariate._cellwise import est_loc_scale, uni_mcd

logger = logging.getLogger(__name__)

SymmetrizationMethod = Literal["rDNsym", "DNsym", "qsym", "sym", "none"]


# ------------------------------------------------------------------- symmetrization


def _random_orders(n: int, k: int, seed: int) -> list[np.ndarray]:
    """The k successive random orderings of the rows used by the "rDNsym" symmetrization: each
    ordering is a random permutation of the previous one. Kept as a separate function so that
    tests can replace it with the orderings drawn by R."""
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    orders = []
    for _ in range(k):
        order = order[rng.permutation(n)]
        orders.append(order)
    return orders


def _random_order(n: int, seed: int) -> np.ndarray:
    """The random ordering of the rows used by the "DNsym" symmetrization."""
    return _random_orders(n, 1, seed)[0]


def _symmetrize(
    X: np.ndarray, y: np.ndarray, method: str, k: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Differences of pairs of observations (X_i - X_j, y_i - y_j), as the functions
    ``symmetrize``, ``qsymmetrize``, ``DNsymmetrize`` and ``randomDNsymmetrize`` of the
    cellLTS code of Raymaekers & Rousseeuw (2026). The differences have center 0 and twice the
    covariance of the observations, and are free of the intercept."""
    Xy = np.column_stack([X, y])
    n = Xy.shape[0]
    if n < 2:
        raise ValueError("At least 2 observations are needed to symmetrize.")
    if method == "rDNsym":
        # each observation minus its successor in k random orderings
        blocks = [
            Xy[np.r_[order[1:], order[0]]] - Xy[order] for order in _random_orders(n, k, seed)
        ]
        D = np.vstack(blocks)
    elif method == "DNsym":
        # each observation subtracted from its k successors in one random ordering
        # (Dumbgen & Nordhausen)
        Xy = Xy[_random_order(n, seed)]
        kk = min(n - 1, k)
        steps = np.arange(1, kk + 1)
        rows = np.arange(n)
        D = (Xy[(rows[:, None] + steps[None, :]) % n] - Xy[rows][:, None, :]).reshape(
            -1, Xy.shape[1]
        )
    elif method in ("sym", "qsym"):
        # all pairs i < j: X_i - X_j
        i, j = np.triu_indices(n, k=1)
        D = Xy[i] - Xy[j]
        if method == "qsym":
            # n * min(n - 1, k) pairs at equispaced quantiles of the differences of y
            kk = min(n - 1, k)
            order = np.argsort(D[:, -1], kind="stable")
            positions = np.round(np.linspace(1, order.size, n * kk)).astype(int) - 1
            D = D[order[positions]]
    else:
        raise ValueError(f"Unknown symmetrization method '{method}'.")
    return D[:, :-1], D[:, -1]


# ------------------------------------------------------------------- flags and imputation


def _get_W(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, h: int, quantile: float) -> np.ndarray:
    """Flag matrix W (1 = kept, 0 = flagged or missing) for fixed location and covariance
    (``getW``): start from the marginal flags and repeat the W-update of cellMCD (with at least
    h kept cells per column) until no flag changes, at most 100 times."""
    cutoff = chi2.ppf(quantile, 1)
    if X.shape[1] == 1:
        with np.errstate(invalid="ignore"):
            return (((X - mu) / np.sqrt(sigma.ravel())) ** 2 <= cutoff).astype(float)
    with np.errstate(invalid="ignore"):
        W = (np.abs((X - mu) / np.sqrt(np.diag(sigma))) <= np.sqrt(cutoff)).astype(float)
    W[np.isnan(X)] = 0.0
    sigma_inv = np.linalg.inv(sigma)
    q = cutoff - np.log(np.diag(sigma_inv)) + np.log(2 * np.pi)
    for _ in range(100):
        W_new = CellMCD._update_W(X, W.copy(), mu, sigma, sigma_inv, q, h)
        converged = np.array_equal(W_new, W)
        W = W_new
        if converged:
            break
    return W


def _impute(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Replace the flagged and missing cells by their conditional expectation given the kept
    cells of the same row (``allpreds_cpp``)."""
    predictions, _ = CellMCD._make_predictions(X, sigma, np.linalg.inv(sigma), mu, W)
    return np.where(W == 0, predictions, X)


def _location_fixed_sigma(
    X: np.ndarray, sigma: np.ndarray, alpha: float, quantile: float, iterate: bool
) -> np.ndarray:
    """Location of X given its covariance (``getmu``): the coordinatewise wrapping location
    of cellWise, followed, if ``iterate``, by cellMCD iterations with the covariance fixed
    (``iterMCD_fixedSigma``)."""
    mu = est_loc_scale(X, "wrap")[0]
    if not iterate or X.shape[1] == 1:
        return mu
    n, d = X.shape
    sigma_inv = np.linalg.inv(sigma)
    cutoff = chi2.ppf(quantile, 1)
    q = cutoff - np.log(np.diag(sigma_inv)) + np.log(2 * np.pi)
    with np.errstate(invalid="ignore"):
        W = (np.abs((X - mu) / np.sqrt(np.diag(sigma))) <= np.sqrt(cutoff)).astype(float)
    W[np.isnan(X)] = 0.0
    h = int(np.ceil(0.75 * n)) if alpha is None else int(np.ceil(alpha * n))
    objective = objective_function(X, W, mu, sigma, sigma_inv) + np.sum(q * (1 - W).sum(0))
    for _ in range(100):
        W_new = CellMCD._update_W(X, W.copy(), mu, sigma, sigma_inv, q, h)
        # one EM step for the location only
        X_imputed = X.copy()
        for i in range(n):
            mis, obs = np.flatnonzero(W_new[i] == 0), np.flatnonzero(W_new[i] == 1)
            if mis.size == d:
                X_imputed[i] = mu
            elif mis.size > 0:
                X_imputed[i, mis] = mu[mis] - np.linalg.solve(
                    sigma_inv[np.ix_(mis, mis)], sigma_inv[np.ix_(mis, obs)] @ (X[i, obs] - mu[obs])
                )
        mu_new = X_imputed.mean(axis=0)
        new_objective = objective_function(X, W_new, mu_new, sigma, sigma_inv) + np.sum(
            q * (1 - W_new).sum(0)
        )
        if new_objective > objective:
            break
        change = np.max(np.abs(mu_new - mu))
        W, mu, objective = W_new, mu_new, new_objective
        if change <= 1e-4:
            break
    return mu


# ------------------------------------------------------------------- LTS with ridge penalty


def _ridge_coefficients(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray | None:
    """Slopes of least squares with intercept and ridge penalty ``ridge * ||beta||^2``
    (``getCoef``); None if the system is singular."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    H = Xc.T @ Xc
    if ridge > 0:
        H[np.diag_indices_from(H)] += ridge
    try:
        beta = np.linalg.solve(H, Xc.T @ yc)
    except np.linalg.LinAlgError:
        return None
    return beta if np.all(np.isfinite(beta)) else None


def _best_window(residuals: np.ndarray, h: int) -> tuple[np.ndarray, float, float]:
    """The h residuals with the smallest variance around their mean (``updateIdx``): returns
    their indices, their mean (the intercept) and their sum of squared deviations (the LTS
    objective)."""
    order = np.argsort(residuals, kind="stable")
    r = residuals[order]
    cs = np.concatenate([[0.0], np.cumsum(r)])
    cs2 = np.concatenate([[0.0], np.cumsum(r**2)])
    sums = cs[h:] - cs[:-h]
    ss = (cs2[h:] - cs2[:-h]) - sums**2 / h
    best = int(np.argmin(ss))
    return order[best : best + h], sums[best] / h, ss[best]


def _lts_ridge(
    X: np.ndarray,
    y: np.ndarray,
    h: int,
    ridge: float,
    rng: np.random.Generator,
    n_starts: int = 500,
    n_final: int = 10,
    max_iter: int = 100,
    precision: float = 1e-5,
) -> tuple[float, np.ndarray]:
    """LTS regression with intercept and a ridge penalty on the slopes (``lts_cpp``).

    ``n_starts`` random elemental subsets of d + 1 observations give starting fits; the
    ``n_final`` starts with the lowest objective are refined by C-steps until the objective
    decreases by less than ``precision``, and the best result is returned."""
    n, d = X.shape
    candidates: list[tuple[float, np.ndarray]] = []
    for _ in range(n_starts):
        order = rng.permutation(n)
        beta = None
        size = d + 1
        while beta is None and size <= n:
            beta = _ridge_coefficients(X[order[:size]], y[order[:size]], ridge)
            size += 1
        if beta is None:
            continue
        idx, _, objective = _best_window(y - X @ beta, h)
        candidates.append((objective, idx))
    if not candidates:
        raise ValueError("Could not find a non-singular subset to start LTS.")
    candidates.sort(key=lambda c: c[0])

    best = (np.inf, 0.0, np.zeros(d))
    for _, idx in candidates[:n_final]:
        objective = np.inf
        intercept, beta = 0.0, np.zeros(d)
        for _ in range(max_iter):
            new_beta = _ridge_coefficients(X[idx], y[idx], ridge)
            if new_beta is None:
                break
            beta = new_beta
            idx, intercept, new_objective = _best_window(y - X @ beta, h)
            if objective - new_objective < precision:
                objective = min(objective, new_objective)
                break
            objective = new_objective
        if objective < best[0]:
            best = (objective, intercept, beta)
    return best[1], best[2]


def _raw_lts_ridge(
    X: np.ndarray,
    y: np.ndarray,
    complete: np.ndarray,
    h: int,
    ridge: float,
    seed: int,
    n_starts: int,
) -> tuple[float, np.ndarray]:
    """Raw LTS fit with ridge penalty on the covariates divided by their robust (wrapping)
    scales, transformed back to the original scale: returns the intercept and the slopes."""
    scales = est_loc_scale(X, "wrap")[1]
    rng = np.random.default_rng(seed)
    intercept, beta = _lts_ridge(X[complete] / scales, y[complete], h, ridge, rng, n_starts)
    return intercept, beta / scales


def _lts_ridge_reweighted(
    y: np.ndarray, X: np.ndarray, alpha: float, ridge: float, seed: int, n_starts: int
) -> tuple[np.ndarray, float]:
    """Reweighted LTS regression with a ridge penalty (``ltsRidgeReg``), on covariates scaled
    by their robust (wrapping) scales. Returns the coefficients (intercept first) and the
    consistency-corrected scale of the residuals."""
    d = X.shape[1]
    complete = np.isfinite(X).all(axis=1) & np.isfinite(y)
    n = int(complete.sum())
    n2 = (n + d + 1) // 2
    h = int(np.floor(2 * n2 - n + 2 * (n - n2) * alpha))
    raw_intercept, raw_beta = _raw_lts_ridge(X, y, complete, h, ridge, seed, n_starts)
    residuals = y - (X @ raw_beta + raw_intercept)

    # consistency and small-sample correction of the raw scale (as in robustbase::ltsReg; the
    # reference code passes the number of slopes as p together with an intercept)
    correction = get_correction_factor(d - 1, n, alpha)
    s0 = np.sqrt(np.mean(np.sort(residuals[complete] ** 2)[:h]))
    qn_q = norm.ppf((h + n) / (2 * n))
    s0 = s0 / np.sqrt(1 - (2 * n) / (h / qn_q) * norm.pdf(qn_q)) * correction
    if abs(s0) < 1e-7:
        return np.concatenate([[raw_intercept], raw_beta]), 0.0

    # reweighted least squares with the ridge penalty as d extra rows ridge * I with response 0
    # (lm.wfit in the reference code; this penalizes ridge^2 * ||beta||^2)
    with np.errstate(invalid="ignore"):
        weights = (np.abs(residuals / s0) <= norm.ppf(0.9875)).astype(float)
    weights[~complete] = 0.0
    sum_w = weights.sum()
    design = np.column_stack([np.ones(n), X[complete]])
    response = y[complete]
    sw = np.sqrt(weights[complete])
    if ridge > 0:
        design_w = np.vstack(
            [design * sw[:, None], np.column_stack([np.zeros(d), np.eye(d) * ridge])]
        )
        response_w = np.concatenate([response * sw, np.zeros(d)])
    else:
        design_w, response_w = design * sw[:, None], response * sw
    coefficients = np.linalg.lstsq(design_w, response_w, rcond=None)[0]
    residuals = response - design @ coefficients
    scale = np.sqrt(np.sum(weights[complete] * residuals**2) / (sum_w - 1))
    if sum_w != n:
        qn_w = norm.ppf((sum_w + n) / (2 * n))
        cdelta = 1 / np.sqrt(1 - (2 * n) / (sum_w / qn_w) * norm.pdf(qn_w))
        scale = scale * cdelta * get_correction_factor_reweighting(d - 1, n, alpha)
    return coefficients, float(scale)


# ------------------------------------------------------------------- estimator


class CellLTSRegression(RobustRegression):
    def __init__(
        self,
        symmetrize: SymmetrizationMethod = "rDNsym",
        n_pairs: int = 20,
        quantile: float = 0.99,
        quantile_cellmcd: float = 0.99,
        alpha_cellmcd: float = 0.5,
        alpha_lts: float = 0.5,
        iterate_location: bool = False,
        ridge: float = 1e-4,
        n_starts: int = 500,
        random_state: int = 123,
    ):
        """Cellwise least trimmed squares regression (cellLTS) of Raymaekers & Rousseeuw (2026),
        for data with cellwise outliers and missing values in the predictors and casewise
        outliers in the response.

        The algorithm follows the authors' R code (``cellLTS``):

        1. The predictors are symmetrized: the regression is fitted on differences of pairs of
           observations (``symmetrize``), which have center 0 and no intercept. cellMCD with a
           fixed center at 0 on the differences of the predictors gives their covariance
           matrix (half that of the differences); the location of the predictors is their
           coordinatewise robust (wrapping) location.
        2. With location and covariance fixed, the cellMCD W-update flags the outlying cells
           of the predictors (at least ``alpha_cellmcd * n`` kept cells per column). Flagged
           and missing cells are imputed by their conditional expectation given the kept cells
           of the same row.
        3. Reweighted LTS regression with a small ridge penalty on the differences of the
           imputed predictors and the response gives the slopes and the residual scale; the
           intercept is the univariate MCD location of y minus the fitted part.
        4. The residuals are standardized by their conditional standard deviation given the
           kept cells of the predictors (joint normal model of the response and the
           predictors); cases with an absolute standardized residual above
           ``sqrt(chi2_1(quantile))`` are flagged.

        With ``symmetrize="none"``, cellMCD (with estimated center) and LTS are applied to the
        data themselves.

        Differences with the reference code: the random pairings and the random LTS subsets
        are drawn by numpy, so results differ slightly from R (with the same pairings the
        results agree closely). The LTS keeps the best of the refined candidates (the
        reference code keeps the last one, and for n > 1500 uses a subsampling scheme);
        location and scale estimates use all observations (cellWise subsamples 25000 values
        for larger data).

        Args:
            symmetrize ("rDNsym", "DNsym", "qsym", "sym" or "none", optional): How to form the
                differences of observations. "rDNsym": each observation minus its successor in
                ``n_pairs`` random orderings (n * n_pairs differences). "DNsym": each
                observation subtracted from its ``n_pairs`` successors in one random ordering
                (Dumbgen & Nordhausen). "sym": all n (n - 1) / 2 pairs. "qsym": n * n_pairs
                pairs at equispaced quantiles of the differences of y. "none": no
                symmetrization. Defaults to "rDNsym".
            n_pairs (int, optional): Number of differences per observation (``k``). Defaults
                to 20.
            quantile (float, optional): Probability of the chi-squared distribution with one
                degree of freedom that sets the cutoff for flagging cases by their
                standardized residuals. Defaults to 0.99.
            quantile_cellmcd (float, optional): The same for flagging cells of the predictors
                (cellMCD). Defaults to 0.99.
            alpha_cellmcd (float, optional): Fraction of cells per predictor that cellMCD must
                keep. Defaults to 0.5.
            alpha_lts (float, optional): Fraction of cases on which LTS fits (h = about
                alpha_lts * n). Defaults to 0.5.
            iterate_location (bool, optional): If True, the location of the predictors is
                refined by cellMCD iterations with fixed covariance (``iterateMu``). Defaults
                to False.
            ridge (float, optional): Size of the ridge penalty in the LTS fit (``lambda``).
                Defaults to 1e-4.
            n_starts (int, optional): Number of random starting subsets of LTS. Defaults to
                500.
            random_state (int, optional): Seed for the random pairings and LTS subsets.
                Defaults to 123.

        Attributes:
            intercept_ (float): Intercept.
            coef_ (np.ndarray): Slopes, one per predictor.
            scale_ (float): Scale of the regression errors.
            predictor_location_ (np.ndarray): Robust location of the predictors.
            predictor_covariance_ (np.ndarray): Robust covariance matrix of the predictors.
            W_ (np.ndarray): Flags of the predictor cells, 1 = kept and 0 = flagged or missing.
            cellwise_outliers_ (np.ndarray): Boolean array, True for flagged (non-missing)
                cells of the predictors.
            X_imputed_ (np.ndarray): Predictors with flagged and missing cells imputed.
            fitted_values_ (np.ndarray): Fitted values, from the imputed predictors.
            residuals_ (np.ndarray): Residuals y minus fitted values.
            standardized_residuals_ (np.ndarray): Residuals divided by their conditional
                standard deviation.
            case_weights_ (np.ndarray): 1 for a case kept, 0 for a case flagged by its
                standardized residual (``v``).

        References:
            - Raymaekers, J., & Rousseeuw, P. J. (2026). Least trimmed squares regression with
              missing values and cellwise outliers. arXiv:2603.04632.
            - Raymaekers, J., & Rousseeuw, P. J. (2024). The cellwise minimum covariance
              determinant estimator. Journal of the American Statistical Association,
              119(548), 2610-2621.
        """
        self.symmetrize = symmetrize
        self.n_pairs = n_pairs
        self.quantile = quantile
        self.quantile_cellmcd = quantile_cellmcd
        self.alpha_cellmcd = alpha_cellmcd
        self.alpha_lts = alpha_lts
        self.iterate_location = iterate_location
        self.ridge = ridge
        self.n_starts = n_starts
        self.random_state = random_state

    # ------------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> CellLTSRegression:
        X, y = self._check_X_y(X, y)
        if self.symmetrize not in ("rDNsym", "DNsym", "qsym", "sym", "none"):
            raise ValueError(f"Unknown symmetrization method '{self.symmetrize}'.")
        symmetrized = self.symmetrize != "none"
        n, d = X.shape
        seed = self.random_state

        if d == 1:
            mu_X, sigma_X, W, X_imputed = self._fit_univariate_predictor(X, y, symmetrized)
        else:
            mu_X, sigma_X = self._predictor_location_covariance(X, y, symmetrized)
            W = _get_W(
                X, mu_X, sigma_X, int(np.round(self.alpha_cellmcd * n)), self.quantile_cellmcd
            )
            X_imputed = _impute(X, mu_X, sigma_X, W)

        observed_y = np.isfinite(y)
        if symmetrized:
            X_fit, y_fit = _symmetrize(
                X_imputed[observed_y], y[observed_y], self.symmetrize, self.n_pairs, 321
            )
        else:
            X_fit, y_fit = X_imputed[observed_y], y[observed_y]
        coefficients, scale = _lts_ridge_reweighted(
            y_fit, X_fit, self.alpha_lts, self.ridge, seed, self.n_starts
        )
        beta = coefficients[1:]
        if symmetrized:
            # the differences have twice the error variance and no intercept
            scale = scale / np.sqrt(2)
            pseudo_residuals = y - X_imputed @ beta
            intercept = uni_mcd(pseudo_residuals[observed_y], self.alpha_lts)["loc"]
        else:
            intercept = coefficients[0]

        fitted = intercept + X_imputed @ beta
        residuals = y - fitted
        # conditional variance of y given the kept predictor cells, joint normal model
        mu_y = intercept + mu_X @ beta
        sigma_xb = sigma_X @ beta
        joint = np.block(
            [
                [np.array([[scale**2 + beta @ sigma_xb]]), sigma_xb[None, :]],
                [sigma_xb[:, None], sigma_X],
            ]
        )
        joint_W = np.column_stack([np.ones(n), W])
        _, conditional_variances = CellMCD._make_predictions(
            np.column_stack([y, X]), joint, np.linalg.inv(joint), np.r_[mu_y, mu_X], joint_W
        )
        standardized = residuals / np.sqrt(conditional_variances[:, 0])
        with np.errstate(invalid="ignore"):
            flagged_cases = np.abs(standardized) > np.sqrt(chi2.ppf(self.quantile, 1))

        self.intercept_ = float(intercept)
        self.coef_ = beta
        self.scale_ = self._scale = float(scale)
        self.predictor_location_ = mu_X
        self.predictor_covariance_ = sigma_X
        self.W_ = W
        self.cellwise_outliers_ = (W == 0) & np.isfinite(X)
        self.X_imputed_ = X_imputed
        self.fitted_values_ = fitted
        self.residuals_ = residuals
        self.standardized_residuals_ = standardized
        self.case_weights_ = (~flagged_cases).astype(float)
        self.n_features_in_ = d
        return self

    def _predictor_location_covariance(
        self, X: np.ndarray, y: np.ndarray, symmetrized: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Location and covariance of the predictors: cellMCD with fixed center 0 on the
        differences of the predictors (half their covariance), and the coordinatewise robust
        location; or cellMCD on the predictors themselves without symmetrization."""
        if not symmetrized:
            cellmcd = CellMCD(alpha=self.alpha_cellmcd, quantile=self.quantile_cellmcd).fit(X)
            return cellmcd.location_, cellmcd.covariance_
        differences, _ = _symmetrize(X, y, self.symmetrize, self.n_pairs, self.random_state)
        cellmcd = CellMCD(
            alpha=self.alpha_cellmcd, quantile=self.quantile_cellmcd, fixed_center=True
        ).fit(differences)
        sigma = cellmcd.covariance_ / 2
        mu = _location_fixed_sigma(
            X, sigma, self.alpha_cellmcd, self.quantile_cellmcd, self.iterate_location
        )
        return mu, sigma

    def _fit_univariate_predictor(self, X: np.ndarray, y: np.ndarray, symmetrized: bool):
        """One predictor: univariate MCD of (the differences of) x; outlying values are
        winsorized at the cutoff instead of imputed (as in the reference code)."""
        observed = np.flatnonzero(np.isfinite(X[:, 0]))
        if symmetrized:
            differences, _ = _symmetrize(
                X[observed], y[observed], self.symmetrize, self.n_pairs, self.random_state
            )
            sigma = uni_mcd(differences[:, 0], 0.5)["scale"] ** 2 / 2
            mu = _location_fixed_sigma(X, np.array([[sigma]]), self.alpha_cellmcd, 0.99, False)
        else:
            fitted = uni_mcd(X[:, 0], 0.5)
            mu, sigma = np.array([fitted["loc"]]), fitted["scale"] ** 2
        cutoff = np.sqrt(chi2.ppf(self.quantile_cellmcd, 1)) * np.sqrt(sigma)
        x = X[:, 0]
        W = np.ones((X.shape[0], 1))
        X_imputed = X.copy()
        with np.errstate(invalid="ignore"):
            high, low = x > mu[0] + cutoff, x < mu[0] - cutoff
        W[high | low, 0] = 0.0
        X_imputed[high, 0] = mu[0] + cutoff
        X_imputed[low, 0] = mu[0] - cutoff
        W[~np.isfinite(x), 0] = 0.0
        return mu, np.array([[sigma]]), W, X_imputed

    # ------------------------------------------------------------------------ new data

    def impute(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Flag the outlying cells of (new) predictors with the fitted location and covariance
        (cellMCD W-update without the per-column constraint) and replace flagged and missing
        cells by their conditional expectation given the kept cells of the same row (by the
        location if there is only one predictor).

        Args:
            X (np.ndarray | pd.DataFrame): Predictors of shape (n_samples, n_features).

        Returns:
            np.ndarray: The imputed predictors.
        """
        X = self._check_new_X(X)
        mu, sigma = self.predictor_location_, self.predictor_covariance_
        W = _get_W(X, mu, sigma, 0, self.quantile_cellmcd)
        if X.shape[1] == 1:
            # with one predictor there is nothing to condition on: the location (as in the
            # reference code; fit winsorizes instead)
            return np.where(W == 0, mu, X)
        return _impute(X, mu, sigma, W) if np.any(W == 0) else X

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict the response of (new) predictors, after flagging and imputing their outlying
        and missing cells as in `impute` (``cellLTS_predict``).

        Args:
            X (np.ndarray | pd.DataFrame): Predictors of shape (n_samples, n_features).

        Returns:
            np.ndarray: The predictions.
        """
        X_imputed = self.impute(X)
        return self.intercept_ + X_imputed @ self.coef_

    def cellmap(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        rows: list[int] | np.ndarray | None = None,
        annotate: bool = False,
        fmt: str = ".3g",
        figsize: tuple[int, int] = (6, 10),
        vmax_clip: float = float(np.sqrt(chi2.ppf(0.999, df=1))),
    ) -> Axes:
        """Cell map of the fitted model, as in the examples of the authors: for each case the
        standardized residual of the response (residual / scale_), and for each predictor cell
        its standardized deviation from the imputed value, (x - x_imputed) / robust standard
        deviation of the predictor. Flagged cells are those with large deviations.

        Args:
            X (np.ndarray | pd.DataFrame): The training predictors. Column names and index of a
                DataFrame are used as labels.
            y (np.ndarray | pd.Series): The training response (its name is used as label).
            rows (list[int] | np.ndarray | None, optional): Positions of the rows to show, e.g.
                the most outlying cases. Defaults to None (all rows).
            annotate (bool, optional): Whether to print the observed values in the cells.
                Defaults to False.
            fmt (str, optional): Format of the annotations. Defaults to ".3g".
            figsize (tuple[int, int], optional): Figure size. Defaults to (6, 10).
            vmax_clip (float, optional): Absolute standardized residuals above this value get
                the darkest color.

        Returns:
            Axes: The matplotlib axes with the heatmap.
        """
        if not hasattr(self, "coef_"):
            raise NotFittedError("Model has not been fitted yet.")
        columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        index = X.index if isinstance(X, pd.DataFrame) else None
        response = y.name if isinstance(y, pd.Series) and y.name is not None else "y"
        X, y = self._check_X_y(X, y)
        columns = [response] + (columns or [f"x{j + 1}" for j in range(X.shape[1])])
        predictor_residuals = (X - self.X_imputed_) / np.sqrt(np.diag(self.predictor_covariance_))
        data = pd.DataFrame(
            np.column_stack([self.residuals_ / self.scale_, predictor_residuals]),
            columns=columns,
            index=index,
        )
        values = pd.DataFrame(np.column_stack([y, X]), columns=columns, index=index)
        if rows is not None:
            data, values = data.iloc[rows], values.iloc[rows]
        _, ax = plt.subplots(1, 1, figsize=figsize)
        sns.heatmap(
            data,
            cmap=get_custom_cmap(vmax_clip),
            vmin=-vmax_clip,
            vmax=vmax_clip,
            annot=values if annotate else False,
            fmt=fmt,
            cbar_kws={"label": "Standardized residuals"},
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        return ax

    # ------------------------------------------------------------------------ checks

    @staticmethod
    def _check_X_y(X, y) -> tuple[np.ndarray, np.ndarray]:
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.array(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have a different number of rows ({X.shape[0]} and {y.shape[0]})."
            )
        X[np.isinf(X)] = np.nan
        y[np.isinf(y)] = np.nan
        return X, y

    def _check_new_X(self, X) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise NotFittedError("Model has not been fitted yet.")
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1) if self.n_features_in_ == 1 else X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.n_features_in_} features."
            )
        X[np.isinf(X)] = np.nan
        return X
