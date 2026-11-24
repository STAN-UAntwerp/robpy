from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from robpy.covariance.base import RobustCovariance
from robpy.pca.cellpca import CellPCA, rho2_parameters, rho_tanh, scale_tanh

logger = logging.getLogger(__name__)


def error_covariance(X: np.ndarray, X_fit: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Weighted covariance of the residuals X - X_fit with cell weights W, normalized by the
    mean of W'W (``get_S_err_cpp``)."""
    R = np.where(np.isfinite(X - X_fit), X - X_fit, 0.0)
    W = np.where(np.isfinite(W), W, 0.0)
    WR = W * R
    S = WR.T @ WR / np.mean(W.T @ W)
    return np.where(np.isfinite(S), S, 0.0)


def _error_covariance_elementwise(X, X_fit, W):
    """As error_covariance, but normalized elementwise by W'W (``get_S_err``)."""
    WR = np.where(np.isfinite(W * (X - X_fit)), W * (X - X_fit), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        S = (WR.T @ WR) / (W.T @ W)
    return np.where(np.isfinite(S), S, 0.0)


def shrink_covariance(S: np.ndarray, delta: float) -> np.ndarray:
    """Shrink the off-diagonal elements of S by the factor 1 - delta (``cov_ridge``)."""
    return (1 - delta) * S + delta * np.diag(np.diag(S))


def cellpca_objective(X: np.ndarray, X_fit: np.ndarray, b_1: float = 1.5, par=None) -> float:
    """Objective of cellPCA for a fit X_fit of X, with scales computed from the residuals
    (``get_obj``)."""
    n, p = X.shape
    par = rho2_parameters(p) if par is None else par
    M = (~np.isnan(X)).astype(float)
    R = X - X_fit
    scale_1 = np.array([scale_tanh(R[:, j]) for j in range(p)])
    rho = rho_tanh(R / scale_1, b=b_1) * scale_1**2 * M
    m = M.sum(axis=1)
    rowti = np.sqrt(np.nansum(rho, axis=1) / m)
    scale_2 = scale_tanh(rowti)
    return float(scale_2**2 * np.sum(rho_tanh(rowti / scale_2, 1.5, *par) * m) / M.sum())


def random_scree(
    n: int, p: int, kmax: int, n_rep: int = 100, b_1: float = 1.5, random_state: int = 100
) -> np.ndarray:
    """0.99 quantiles of the decreases of the cellPCA objective with each extra component,
    for standard normal data of the same size (``get_ranscree_app``, with numpy's random
    numbers)."""
    par = rho2_parameters(p)
    objectives = np.zeros((n_rep, kmax + 1))
    for r in range(n_rep):
        rng = np.random.default_rng(random_state + r)
        X = rng.normal(size=(n, p))
        X = X / np.array([scale_tanh(x - np.median(x)) for x in X.T])
        center = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - center, full_matrices=False)
        scores = (X - center) @ Vt.T
        objectives[r, 0] = cellpca_objective(X, np.tile(center, (n, 1)), b_1, par)
        for k in range(1, kmax + 1):
            fit = center + scores[:, :k] @ Vt[:k]
            objectives[r, k] = cellpca_objective(X, fit, b_1, par)
    return np.quantile(np.abs(np.diff(objectives, axis=1)), 0.99, axis=0)


def shrinkage_cross_validation(
    X: np.ndarray,
    X_fit: np.ndarray,
    W: np.ndarray,
    deltas: np.ndarray,
    n_splits: int = 5,
    fraction: float = 1 / 3,
    random_state: int = 100,
) -> tuple[float, np.ndarray]:
    """Choose the shrinkage of the error covariance by repeated random splits: the shrunken
    covariance of one part should be close (Frobenius norm) to the unshrunken covariance of
    the other part (``multifold_cv``)."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    n1 = int(np.floor(n * fraction))
    errors = []
    for delta in deltas:
        e = []
        for _ in range(n_splits):
            order = rng.permutation(n)
            first, second = order[:n1], order[n1:]
            S1 = _error_covariance_elementwise(X[first], X_fit[first], W[first])
            S2 = _error_covariance_elementwise(X[second], X_fit[second], W[second])
            e.append(np.linalg.norm(shrink_covariance(S1, delta) - S2, "fro"))
        errors.append(np.mean(e))
    errors = np.array(errors)
    return float(deltas[int(np.argmin(errors))]), errors


class CellRCov(RobustCovariance):
    def __init__(
        self,
        *,
        n_components: int | None = None,
        max_components: int = 20,
        regularize: bool = True,
        shrinkage: float | None = None,
        shrinkage_grid: np.ndarray | None = None,
        scale: bool = True,
        b_1: float = 1.5,
        tol: float = 1e-6,
        max_iter: int = 100,
        quantile_diff: np.ndarray | None = None,
        n_random_scree: int = 100,
        random_state: int = 100,
    ):
        """Cellwise and casewise robust covariance estimator (cellRCov) of Centofanti, Hubert
        & Rousseeuw (2026), following the authors' R code (``cellRCov``).

        The (standardized) data are fitted by cellPCA with k components. The covariance is the
        sum of the covariance of the k-dimensional subspace, V diag(eigenvalues) V', and the
        covariance of the residuals with cellwise and rowwise weights (the error covariance).
        The off-diagonal elements of the error covariance can be shrunk (``regularize``), which
        keeps the estimate positive definite also when the dimension is large.

        Args:
            n_components (int | None, optional): Number of components k of cellPCA. If None,
                k is the first number of components after which the objective of cellPCA
                decreases by less than the 0.99 quantile of the decrease for standard normal
                data (``quantile_diff``). Defaults to None.
            max_components (int, optional): Largest k considered when k is chosen
                (``kmax``). Defaults to 20.
            regularize (bool, optional): Shrink the error covariance (``reg``). Defaults to
                True.
            shrinkage (float | None, optional): Shrinkage delta of the off-diagonal elements of
                the error covariance; if None it is chosen by cross-validation over
                ``shrinkage_grid`` (``delta_reg``). Defaults to None.
            shrinkage_grid (np.ndarray | None, optional): Values of delta for the
                cross-validation. Defaults to 10 equispaced values from 0.01 to 1.
            scale (bool, optional): Standardize the variables by a robust scale first.
                Defaults to True.
            b_1, tol, max_iter: Tuning constant, tolerance and maximum number of iterations
                of cellPCA. Defaults to 1.5, 1e-6 and 100.
            quantile_diff (np.ndarray | None, optional): Thresholds for choosing k; by default
                simulated with ``n_random_scree`` standard normal data sets.
            n_random_scree (int, optional): Number of simulated data sets. Defaults to 100.
            random_state (int, optional): Seed of the simulations and of the
                cross-validation. Defaults to 100.

        Attributes:
            covariance_ (np.ndarray): The covariance estimate.
            location_ (np.ndarray): The location estimate (center of cellPCA).
            n_components_ (int): The number of components k used.
            shrinkage_ (float): The shrinkage delta used.
            scales_ (np.ndarray): The robust scales used to standardize the variables.
            subspace_covariance_ (np.ndarray): Covariance of the cellPCA subspace,
                V diag(eigenvalues) V', on the standardized scale.
            error_covariance_ (np.ndarray): Weighted covariance of the cellPCA residuals (shrunk
                when regularized), on the standardized scale.
            cellpca_ (CellPCA): cellPCA of the standardized data.

        References:
            - Centofanti, F., Hubert, M., & Rousseeuw, P. J. (2026). Cellwise and casewise
              robust multivariate regression with inference. arXiv:2605.08002.
        """
        super().__init__(store_precision=True, assume_centered=False, nans_allowed=True)
        self.n_components = n_components
        self.max_components = max_components
        self.regularize = regularize
        self.shrinkage = shrinkage
        self.shrinkage_grid = shrinkage_grid
        self.scale = scale
        self.b_1 = b_1
        self.tol = tol
        self.max_iter = max_iter
        self.quantile_diff = quantile_diff
        self.n_random_scree = n_random_scree
        self.random_state = random_state

    def calculate_covariance(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        X[np.isinf(X)] = np.nan
        n, p = X.shape
        if self.scale:
            scales = np.array([scale_tanh(X[:, j] - np.nanmedian(X[:, j])) for j in range(p)])
        else:
            scales = np.ones(p)
        Z = X / scales
        M = (~np.isnan(Z)).astype(float)

        k = self.n_components
        if k is None:
            k = self._choose_k(Z)
        if k == 0:
            # no subspace: cellPCA with 2 components only for the weights (no iterations)
            model = self._cellpca(2, max_iter=0).fit(Z)
            subspace = np.zeros((p, p))
            Z_fit = np.tile(model.location_, (n, 1))
        else:
            model = self._cellpca(k).fit(Z)
            k = model.components_.shape[1]
            V = model.components_
            subspace = V @ np.diag(model.explained_variance_) @ V.T
            Z_fit = model.fitted_
        W = model.weights_cellwise_ * np.sqrt(model.weights_rowwise_)[:, None] * M
        error = error_covariance(Z, Z_fit, W)
        eigenvalues, vectors = np.linalg.eigh(error)
        error = vectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ vectors.T

        shrinkage = None
        if self.regularize:
            shrinkage = self.shrinkage
            if shrinkage is None:
                grid = (
                    np.linspace(0.01, 1, 10)
                    if self.shrinkage_grid is None
                    else np.asarray(self.shrinkage_grid, dtype=float)
                )
                shrinkage, self.shrinkage_errors_ = shrinkage_cross_validation(
                    Z, Z_fit, W, grid, random_state=self.random_state
                )
            error = shrink_covariance(error, shrinkage)

        standardized = subspace + error
        self.n_components_ = k
        self.shrinkage_ = shrinkage
        self.scales_ = scales
        self.subspace_covariance_ = subspace
        self.error_covariance_ = error
        self.standardized_covariance_ = standardized
        self.cellpca_ = model
        self.location_ = scales * model.location_
        return np.outer(scales, scales) * standardized

    def _cellpca(self, k: int, max_iter: int | None = None) -> CellPCA:
        return CellPCA(
            k,
            b_1=self.b_1,
            tol=self.tol,
            max_iter=self.max_iter if max_iter is None else max_iter,
            random_state=0,
        )

    def _choose_k(self, Z: np.ndarray) -> int:
        """Number of components after which the objective decreases by less than for random
        normal data (``select_k`` and ``get_ranscree_app``)."""
        n, p = Z.shape
        kmax = min(self.max_components, p)
        objectives = [cellpca_objective(Z, np.tile(np.nanmedian(Z, axis=0), (n, 1)), self.b_1)]
        for k in range(1, kmax + 1):
            if k == p:
                objectives.append(0.0)
                continue
            model = self._cellpca(k).fit(Z)
            objectives.append(cellpca_objective(Z, model.fitted_, self.b_1, model.rho2_par_))
        self.objectives_ = np.array(objectives)
        thresholds = self.quantile_diff
        if thresholds is None:
            thresholds = random_scree(n, p, kmax, self.n_random_scree, self.b_1)
        below = np.flatnonzero(np.abs(np.diff(self.objectives_)) < thresholds)
        k = int(below[0]) if below.size else kmax
        return max(0, k)

    def fit(self, X: np.ndarray | pd.DataFrame, y=None) -> CellRCov:
        return super().fit(X)
