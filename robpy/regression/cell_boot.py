from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2

from robpy.covariance.cellrcov import CellRCov
from robpy.outliers.ddc import _slope_med_wls
from robpy.pca.cellpca import calculate_q1q2, rho2_parameters, scale_tanh, weight_tanh
from robpy.univariate._cellwise import rho_huber25, scale_1step_m

logger = logging.getLogger(__name__)


@dataclass
class _WeightingModel:
    """Fixed parts of a cellRCov fit used to weight (resampled or simulated) data: the cellwise
    predictions of DDC (with soft weights) and the subspace and scales of cellPCA."""

    loc: np.ndarray  # location of the DDC standardization (cellPCA center)
    scale: np.ndarray  # scale of the DDC standardization (diagonal of the covariance)
    connected: np.ndarray
    neighbors: list
    slopes: list
    weights: list
    deshrinkage: np.ndarray
    residual_scales: np.ndarray
    mu: np.ndarray  # cellPCA center
    V: np.ndarray  # cellPCA loadings
    rho2_par: tuple
    par2: tuple
    mcd_center: np.ndarray
    mcd_chol: np.ndarray
    mcd_cutoff: float
    scale_1: np.ndarray | None = None
    scale_2: float | None = None


def _weighted_predictions(Z, model: _WeightingModel, W_filt):
    """Weighted-mean predictions of the connected columns from their neighbors, with weights
    (absolute correlation) x (cell weight of the neighbor) (``predict_cols_wmean_vec``)."""
    estimates = np.zeros_like(Z)
    for j, ngb, slopes, weights in zip(
        model.connected, model.neighbors, model.slopes, model.weights
    ):
        values = Z[:, ngb] * slopes
        w = W_filt[:, ngb] * weights
        bad = ~np.isfinite(values) | ~np.isfinite(w)
        values, w = np.where(bad, 0.0, values), np.where(bad, 0.0, w)
        with np.errstate(divide="ignore", invalid="ignore"):
            est = (values * w).sum(axis=1) / w.sum(axis=1)
        estimates[:, j] = np.where(np.isfinite(est), est, 0.0)
    return estimates


def _ddc_weighted_impute(X: np.ndarray, model: _WeightingModel) -> np.ndarray:
    """Soft DDC imputation (``DDCpredict2weight_cpp``): each cell is moved towards its
    prediction by the product of its tanh weight and the tanh weight of its residual."""
    Z = (X - model.loc) / model.scale
    W_filt = weight_tanh(Z)
    estimates = _weighted_predictions(Z, model, W_filt)
    estimates[:, model.connected] *= model.deshrinkage
    residuals = Z - estimates
    residuals[:, model.connected] /= model.residual_scales
    W = W_filt * weight_tanh(residuals)
    W[np.isnan(Z)] = 0.0
    X_est = estimates * model.scale + model.loc
    return W * (X - X_est) + X_est


def _subspace_weights(X, X_imputed, model: _WeightingModel, M=None, scales=None) -> dict:
    """Cellwise and rowwise weights from the residuals of the cellPCA subspace
    (``get_weights_sub``). ``scales``: None uses the scales of the model, "estimate" computes
    them, otherwise (scale_1, scale_2)."""
    n, p = X.shape
    centered = X_imputed - model.mu
    centered[np.isnan(centered)] = 0.0
    U = centered @ model.V
    fit = U @ model.V.T
    residuals = X - fit - model.mu
    if scales is None:
        scale_1, scale_2 = model.scale_1, model.scale_2
    elif isinstance(scales, str):
        scale_1 = np.array([scale_tanh(residuals[:, j]) for j in range(p)])
        scale_2 = None
    else:
        scale_1, scale_2 = scales
    weights_c = weight_tanh(residuals / scale_1, 1.5, *model.par2)
    weights_c[np.isnan(weights_c)] = 0.0
    squared = (X - fit - model.mu) ** 2
    if M is not None:
        squared[M == 0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rowti = np.nansum(weights_c * squared, axis=1) / np.nansum(weights_c, axis=1)
    if scale_2 is None:
        scale_2 = scale_tanh(rowti)
    weights_r = weight_tanh(rowti / scale_2, 1.5, *model.rho2_par)
    weights_r[np.isnan(weights_r)] = 0.0
    if M is not None:
        weights_c[M == 0] = 0.0
    return {
        "W": weights_c * np.sqrt(weights_r)[:, None],
        "weights_c": weights_c,
        "weights_r": weights_r,
        "U": U,
        "scale_1": scale_1,
        "scale_2": scale_2,
    }


def _mcd_weights(U, model: _WeightingModel, diagonal_only: bool = False) -> np.ndarray:
    """Weights of the scores from their squared Mahalanobis distance to the raw MCD of the
    cellPCA scores (``get_weights_mcd_tanh``). The reference code passes the tuning constants
    of the tanh weight function in shifted positions, which gives a hard rejection weight at
    the constant q1 of the (0.99, 0.9999) chi-squared quantiles; this is reproduced. In the
    bias correction (``S_unb_cov``) the reference code passes an upper triangular Cholesky
    factor where a lower triangular one is expected, so that only its diagonal is used
    (``diagonal_only``); this is reproduced as well."""
    if diagonal_only:
        distances = np.sum(((U - model.mcd_center) / np.diag(model.mcd_chol)) ** 2, axis=1)
    else:
        Y = np.linalg.solve(model.mcd_chol, (U - model.mcd_center).T)
        distances = np.sum(Y**2, axis=0)
    return (distances <= model.mcd_cutoff).astype(float)


def _weighted_covariance(
    X, model: _WeightingModel, M, scales, scaleX, shrinkage, diagonal_mcd: bool = False
):
    """One weighted location and covariance estimate (the core of cellBoot and of its bias
    correction), on the original scale."""
    X_imputed = _ddc_weighted_impute(X, model)
    out = _subspace_weights(X, X_imputed, model, M=M, scales=scales)
    weights_r = out["weights_r"]
    weights_sub = np.zeros(X.shape[0])
    rows = weights_r != 0
    weights_sub[rows] = _mcd_weights(out["U"][rows], model, diagonal_mcd)
    W = out["W"] * np.sqrt(weights_sub)[:, None]
    R = np.where(np.isfinite(X - model.mu), X - model.mu, 0.0)
    W = np.where(np.isfinite(W), W, 0.0)
    WR = W * R
    with np.errstate(divide="ignore", invalid="ignore"):
        S = (WR.T @ WR) / (W.T @ W)
    S = np.where(np.isfinite(S), S, 0.0)
    S = (1 - shrinkage) * S + shrinkage * np.diag(np.diag(S))
    Wmu = out["weights_c"] * (weights_r * weights_sub)[:, None]
    mu = np.nansum(Wmu * X, axis=0) / Wmu.sum(axis=0)
    return np.outer(scaleX, scaleX) * S, scaleX * mu, out


def _psd(S: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    values, vectors = np.linalg.eigh(S)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def _debias(mu0, S0, model, n, scaleX, shrinkage, n_sim, max_iter, tol, seed):
    """Bias correction of the weighted covariance by simulation (``S_unb_cov``): find the
    normal model whose simulated weighted estimates average to the observed estimate."""
    S_init = _psd(S0)
    seeds = np.random.default_rng(seed).integers(1, 10**8, size=n_sim)
    b, b_mu = np.zeros_like(S_init), np.zeros_like(mu0)
    objectives, S_iter, mu_iter = [], [], []
    above = 0
    for iteration in range(max_iter):
        S = S_init + b
        mu = mu0 + b_mu
        values, vectors = np.linalg.eigh(0.5 * (S + S.T))
        L = vectors * np.sqrt(np.maximum(values, 1e-14))
        estimates = []
        for s in seeds:
            Xs = np.random.default_rng(s).normal(size=(n, len(mu))) @ L.T + mu
            estimates.append(
                _weighted_covariance(Xs / scaleX, model, None, None, scaleX, shrinkage, True)[:2]
            )
        g = np.mean([e[0] for e in estimates], axis=0)
        g_mu = np.mean([e[1] for e in estimates], axis=0)
        mu_iter.append(mu)
        S_iter.append(S)
        b = _psd(S + (S_init - g)) - S_init
        b_mu = mu + (mu0 - g_mu) - mu0
        objective = float(np.sum((g - S_init) ** 2) + np.sum((g_mu - mu0) ** 2))
        objectives.append(objective)
        if objective <= tol:
            break
        if objective > min(objectives) and iteration > 3:
            above += 1
            if above > 6:
                break
        else:
            above = 0
    best = int(np.argmin(objectives))
    return S_iter[best], mu_iter[best], objectives


def _regression(S, mu, p, ridge):
    B = np.linalg.solve(S[:p, :p] + ridge * np.eye(p), S[:p, p:])
    return B, mu[p:] - B.T @ mu[:p]


@dataclass
class CellBootResult:
    """Result of cellBoot.

    Attributes:
        coef (np.ndarray): Bias-corrected slopes B of the observed data, (n_features, n_targets).
        intercept (np.ndarray): Bias-corrected intercepts b0.
        covariance (np.ndarray), location (np.ndarray): Bias-corrected joint estimates.
        theta (np.ndarray): Bootstrap replicates of the linear combinations, (B, m).
        confidence_intervals (np.ndarray): Percentile intervals, (m, 2).
        linear_combinations (np.ndarray): The linear combinations (rows) of (b0, vec(B)).
    """

    coef: np.ndarray
    intercept: np.ndarray
    covariance: np.ndarray
    location: np.ndarray
    theta: np.ndarray
    confidence_intervals: np.ndarray
    linear_combinations: np.ndarray


def cellboot(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    n_components: int = 1,
    ridge: float = 1e-8,
    linear_combinations: np.ndarray | None = None,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    n_simulations: int = 40,
    max_iter: int = 100,
    tol: float = 1e-8,
    n_jobs: int | None = None,
    random_state: int = 123,
) -> CellBootResult:
    """Bootstrap confidence intervals for cellMR (cellBoot) of Centofanti, Hubert & Rousseeuw
    (2026), following the authors' R code (``cellBoot``).

    cellMR's estimate is computed by a fast weighted version (FastCellRCov): the weights of the
    cells and rows come from the cellRCov fit of the data, so each bootstrap sample only needs a
    weighted covariance. Its bias is removed by simulation: the normal model is found whose
    simulated weighted estimates average to the observed one. The intervals are the percentile
    intervals of the bias-corrected bootstrap estimates of the requested linear combinations
    of (b0, vec(B)).

    The Monte Carlo draws and the bootstrap samples use numpy's random numbers, so they differ
    from those of the R code.

    Args:
        X, Y: Predictors (n, p) and responses (n, q); may contain missing values.
        n_components (int): Number of components k of cellPCA. Defaults to 1.
        ridge (float): Ridge penalty ``lambda``. Defaults to 1e-8.
        linear_combinations (np.ndarray | None): Rows of the linear combinations of
            (b0, vec(B)) (length q + p q, B stacked by columns); default all coefficients.
        alpha (float): 1 - confidence level. Defaults to 0.05.
        n_bootstrap (int): Number of bootstrap samples B. Defaults to 1000.
        n_simulations (int): Number of simulated data sets per bias-correction step (``H``).
            Defaults to 40.
        max_iter (int), tol (float): Maximum number of bias-correction steps and tolerance.
        n_jobs (int | None): Number of parallel jobs for the bootstrap (joblib).
        random_state (int): Seed.

    Returns:
        CellBootResult

    References:
        - Centofanti, F., Hubert, M., & Rousseeuw, P. J. (2026). Cellwise and casewise robust
          multivariate regression with inference. arXiv:2605.08002.
    """
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, p = X.shape
    q = Y.shape[1]
    Z = np.column_stack([X, Y])
    Z[np.isinf(Z)] = np.nan

    cov = CellRCov(
        n_components=n_components,
        max_components=min(30, p + q - 1),
        regularize=True,
        shrinkage=0.0,
        max_iter=1000,
    ).fit(Z)
    shrinkage = cov.shrinkage_
    scaleX = cov.scales_
    Zs = Z / scaleX
    model = _build_model(cov, Zs, p + q, n_components)
    M = (~np.isnan(Z)).astype(float)

    # scales of the residuals of the observed data, then fixed
    first = _subspace_weights(Zs, _ddc_weighted_impute(Zs, model), model, M=M, scales="estimate")
    scales = (first["scale_1"], first["scale_2"])
    model.scale_1, model.scale_2 = scales

    S_obs, mu_obs, _ = _weighted_covariance(Zs, model, M, scales, scaleX, shrinkage)
    S_unb, mu_unb, _ = _debias(
        mu_obs, S_obs, model, n, scaleX, shrinkage, n_simulations, max_iter, tol, random_state
    )
    B_obs, b_obs = _regression(S_unb, mu_unb, p, ridge)

    if linear_combinations is None:
        linear_combinations = np.eye(q + p * q)
    linear_combinations = np.atleast_2d(np.asarray(linear_combinations, dtype=float))

    def replicate(r):
        rng = np.random.default_rng(10 * (r + 1))
        index = np.repeat(np.arange(n), rng.multinomial(n, np.full(n, 1 / n)))
        S_b, mu_b, _ = _weighted_covariance(Zs[index], model, M[index], scales, scaleX, shrinkage)
        S_u, mu_u, _ = _debias(
            mu_b, S_b, model, n, scaleX, shrinkage, n_simulations, max_iter, tol, random_state
        )
        B_u, b_u = _regression(S_u, mu_u, p, ridge)
        return linear_combinations @ np.concatenate([b_u, B_u.ravel(order="F")])

    theta = np.array(Parallel(n_jobs=n_jobs)(delayed(replicate)(r) for r in range(n_bootstrap)))
    intervals = np.quantile(theta, [alpha / 2, 1 - alpha / 2], axis=0).T
    return CellBootResult(B_obs, b_obs, S_unb, mu_unb, theta, intervals, linear_combinations)


def _build_model(cov: CellRCov, Zs: np.ndarray, d: int, k: int) -> _WeightingModel:
    """The weighting model of cellBoot from the cellRCov fit of the standardized data."""
    pca = cov.cellpca_
    ddc = pca._macropca.ddc
    loc = pca.location_
    scale = np.diag(cov.standardized_covariance_)  # as in the reference code (variances)
    weights_full = ddc._neighbor_weights
    connected = np.flatnonzero(weights_full[:, 1:].sum(axis=1) > 0)
    neighbors, slopes, weights = [], [], []
    for j in connected:
        used = weights_full[j] > 0
        neighbors.append(ddc.neighbors_[j, used])
        slopes.append(ddc.neighbor_slopes_[j, used])
        weights.append(weights_full[j, used])
    model = _WeightingModel(
        loc=loc,
        scale=scale,
        connected=connected,
        neighbors=neighbors,
        slopes=slopes,
        weights=weights,
        deshrinkage=np.ones(len(connected)),
        residual_scales=np.ones(len(connected)),
        mu=pca.location_,
        V=pca.components_,
        rho2_par=rho2_parameters(d, fast=True),
        par2=(1.5, 4.0, *calculate_q1q2(1.5, 4.0)),
        mcd_center=pca.mcd_.raw_center,
        mcd_chol=np.linalg.cholesky(pca.mcd_.raw_cov),
        mcd_cutoff=calculate_q1q2(chi2.ppf(0.99, k), chi2.ppf(0.9999, k))[0],
    )
    # deshrinkage factors and residual scales of the weighted predictions on the data
    # (``DDCpredict2weight_compute_const``)
    Z = (Zs - loc) / scale
    estimates = _weighted_predictions(Z, model, weight_tanh(Z))
    q_regr = np.sqrt(chi2.ppf(0.99, 1))
    deshrinkage = np.array(
        [_slope_med_wls(estimates[:, j], Z[:, j], q_regr, 1e-12) for j in connected]
    )
    estimates[:, connected] *= deshrinkage
    estimates[~np.isfinite(estimates)] = 0.0
    residuals = Z - estimates
    model.deshrinkage = deshrinkage
    model.residual_scales = np.array(
        [scale_1step_m(residuals[:, j], rho_huber25) for j in connected]
    )
    return model
