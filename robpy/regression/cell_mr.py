from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from robpy.covariance.cellrcov import CellRCov, random_scree
from robpy.pca.cellpca import CellPCA, rho2_parameters, scale_tanh, weight_tanh
from robpy.regression.base import RobustRegression

logger = logging.getLogger(__name__)


def _regression_from_covariance(S, mu, p, ridge):
    """Slopes and intercepts of the regression of the last q variables on the first p,
    from a joint location and covariance, with a ridge penalty."""
    G = np.linalg.pinv(S[:p, :p] + ridge * np.eye(p))
    B = G @ S[:p, p:]
    b0 = mu[p:] - B.T @ mu[:p]
    return B, b0, G


class CellMRRegression(RobustRegression):
    def __init__(
        self,
        n_components: int = 1,
        ridge: float = 1e-8,
        n_components_X: int | None = None,
        shrinkage_X: float | None = None,
        max_iter: int = 1000,
        random_state: int = 100,
    ):
        """Cellwise and casewise robust multivariate regression (cellMR) of Centofanti,
        Hubert & Rousseeuw (2026), following the authors' R code (``cellMR``).

        The joint location and covariance of the predictors X and the responses Y are
        estimated by cellRCov (cellPCA with ``n_components`` components on the standardized
        data, plus an error covariance). The slopes and intercepts follow from them as in
        least squares, with a ridge penalty on the covariance of X:
        B = (S_xx + ridge I)^+ S_xy and b0 = mu_y - B' mu_x.

        To predict, the predictors are first cleaned by cellPCA of X alone with
        ``n_components_X`` components: outlying cells are shrunk towards their fit and missing
        cells replaced by it.

        Args:
            n_components (int, optional): Number of components k of cellPCA of (X, Y).
                Defaults to 1.
            ridge (float, optional): Ridge penalty ``lambda``. Defaults to 1e-8.
            n_components_X (int | None, optional): Number of components of cellPCA of X used
                to impute the predictors (``kX``). If None, the predictors are not imputed and
                cannot contain missing values when predicting. Defaults to None.
            shrinkage_X (float | None, optional): Shrinkage of the error covariance of X
                (``delta_reg``); None chooses it by cross-validation. Defaults to None.
            max_iter (int, optional): Maximum number of iterations of cellPCA. Defaults to 1000.
            random_state (int, optional): Seed of the cross-validation of the shrinkage.

        Attributes:
            coef_ (np.ndarray): Slopes B, shape (n_features, n_targets).
            intercept_ (np.ndarray): Intercepts b0, shape (n_targets,).
            covariance_ (np.ndarray): Joint covariance of (X, Y) by cellRCov.
            location_ (np.ndarray): Joint location of (X, Y).
            error_covariance_ (np.ndarray): Covariance of the errors,
                S_yy - S_yx (S_xx + ridge I)^+ S_xy (``S_eps``), and its regularized version
                ``error_covariance_reg_`` (``S_eps_reg``).
            cellrcov_ (CellRCov): cellRCov of (X, Y).
            cellrcov_X_ (CellRCov | None): cellRCov of X (when ``n_components_X`` is given).
            X_imputed_ (np.ndarray | None): Imputed training predictors (when
                ``n_components_X`` is given).
            fitted_values_ (np.ndarray | None): Fitted responses from the imputed predictors
                (when ``n_components_X`` is given).

        References:
            - Centofanti, F., Hubert, M., & Rousseeuw, P. J. (2026). Cellwise and casewise
              robust multivariate regression with inference. arXiv:2605.08002.
        """
        self.n_components = n_components
        self.ridge = ridge
        self.n_components_X = n_components_X
        self.shrinkage_X = shrinkage_X
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray | pd.DataFrame, Y: np.ndarray | pd.DataFrame) -> CellMRRegression:
        X, Y = self._check_X_Y(X, Y)
        n, p = X.shape
        q = Y.shape[1]
        joint = CellRCov(
            n_components=self.n_components,
            max_components=min(30, p + q - 1),
            regularize=False,
            max_iter=self.max_iter,
        ).fit(np.column_stack([X, Y]))
        S, mu = joint.covariance_, joint.location_
        B, b0, G = _regression_from_covariance(S, mu, p, self.ridge)
        error = S[p:, p:] - S[p:, :p] @ G @ S[:p, p:]
        lambda_eps = max(0.0, 1e-6 * np.trace(error) / q - np.min(np.linalg.eigvalsh(error)))

        self.coef_, self.intercept_ = B, b0
        self.covariance_, self.location_ = S, mu
        self.predictor_covariance_ = S[:p, :p]
        self.error_covariance_ = error
        self.error_covariance_reg_ = error + lambda_eps * np.eye(q)
        self.cellrcov_ = joint
        self.cellrcov_X_ = None
        self.X_imputed_ = self.fitted_values_ = None
        if self.n_components_X is not None:
            covX = CellRCov(
                n_components=self.n_components_X,
                max_components=min(30, p + q - 1),
                regularize=True,
                shrinkage=self.shrinkage_X,
                max_iter=self.max_iter,
                random_state=self.random_state,
            ).fit(X)
            scales = covX.scales_
            self.cellrcov_X_ = covX
            self.X_imputed_ = covX.cellpca_.X_imputed_ * scales
            self.X_fitted_ = covX.cellpca_.fitted_ * scales
            self.X_fit_scales_ = np.array(
                [scale_tanh(X[:, j] - self.X_fitted_[:, j]) for j in range(p)]
            )
            self.fitted_values_ = self.X_imputed_ @ B + b0
        self.n_features_in_ = p
        self.n_targets_ = q
        return self

    def impute(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Shrink the outlying cells of (new) predictors towards their cellPCA fit and replace
        missing cells by it (``cellPCA_predict`` with the cellPCA of X)."""
        X = self._check_new_X(X)
        if self.cellrcov_X_ is None:
            if np.isnan(X).any():
                raise ValueError("X has missing values; fit with n_components_X to impute them.")
            return X
        scales = self.cellrcov_X_.scales_
        return self.cellrcov_X_.cellpca_.impute(X / scales) * scales

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict the responses of (new) predictors, after imputing them (see `impute`)."""
        X_imputed = self.impute(X)
        return X_imputed @ self.coef_ + self.intercept_

    @staticmethod
    def _check_X_Y(X, Y):
        X = np.array(X, dtype=float)
        Y = np.array(Y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y have a different number of rows ({X.shape[0]}, {Y.shape[0]})."
            )
        X[np.isinf(X)] = np.nan
        Y[np.isinf(Y)] = np.nan
        return X, Y

    def _check_new_X(self, X) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise NotFittedError("Model has not been fitted yet.")
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but the model was fitted with "
                f"{self.n_features_in_} features."
            )
        X[np.isinf(X)] = np.nan
        return X


def robust_prediction_error(Y: np.ndarray, Y_hat: np.ndarray, par: tuple) -> dict:
    """Robust (weighted) mean squared prediction error per response, downweighting outlying
    cells and rows of the residuals (``get_MSErobust``)."""
    n, q = Y.shape
    M = np.ones((n, q))
    R = Y - Y_hat
    scale_1 = np.array([scale_tanh(R[:, j]) for j in range(q)])
    weights_c = weight_tanh(R / scale_1, b=1.5)
    total = np.nansum(weights_c, axis=1)
    used = total != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        rowti = np.nansum(weights_c * R**2, axis=1) / total
    scale_2 = scale_tanh(rowti[used])
    weights_r = np.zeros(n)
    weights_r[used] = weight_tanh(rowti[used] / scale_2, 1.5, *par)
    Wr = np.tile(weights_r[:, None], (1, q))
    effective = np.nansum(M * Wr * weights_c**2, axis=0)
    numerator = np.nansum(M * Wr * (weights_c * R) ** 2, axis=0)
    return {"cv": numerator / effective, "neff": effective, "M": M}


def cellmr_tuning(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    k_grid,
    ridge_grid,
    k_grid_X=None,
    shrinkage_grid=None,
    n_folds: int = 10,
    random_state: int = 1,
    n_random_scree: int = 100,
) -> dict:
    """Choose the tuning parameters of cellMR by robust K-fold cross-validation, following
    ``cellMR_tuning`` of the authors' R code.

    For each fold and each k in ``k_grid``, cellRCov of (X, Y) is fitted on the training
    folds; cellPCA of X with k_X components (chosen per fold from ``k_grid_X`` by the random
    scree rule when it has several values) imputes the test predictors. Each combination of k
    and ridge in ``ridge_grid`` is scored by the robust mean squared prediction error of the
    test responses; combinations whose effective sample size is below half are discarded. The
    number of components and shrinkage for imputing X are chosen on the whole data.

    Args:
        X, Y: Predictors and responses.
        k_grid: Values of the number of components of cellMR.
        ridge_grid: Values of the ridge penalty.
        k_grid_X: Values of the number of components of cellPCA of X; defaults to k_grid.
        shrinkage_grid: Values of the shrinkage of the error covariance of X; None uses the
            default cross-validation grid of cellRCov.
        n_folds (int): Number of folds K. Defaults to 10.
        random_state (int): Seed of the folds. Defaults to 1.
        n_random_scree (int): Number of simulated data sets for the random scree rule.

    Returns:
        dict: ``best`` (n_components, ridge), ``n_components_X``, ``shrinkage_X`` and the table
        ``cv_table`` of mean robust prediction errors.
    """
    X, Y = CellMRRegression._check_X_Y(X, Y)
    n, p = X.shape
    q = Y.shape[1]
    k_grid = list(k_grid)
    k_grid_X = k_grid if k_grid_X is None else list(k_grid_X)
    rng = np.random.default_rng(random_state)
    folds = rng.permutation(np.resize(np.arange(n_folds), n))
    par = rho2_parameters(q, fast=True)

    joint_fits = _fit_joint_by_fold(X, Y, k_grid, folds, n_folds)
    X_test_imputed = _impute_test_folds(
        X, folds, n_folds, joint_fits, k_grid[0], k_grid_X, n_random_scree
    )

    covX = CellRCov(
        n_components=None if len(k_grid_X) > 1 else k_grid_X[0],
        max_components=max(k_grid_X),
        regularize=True,
        shrinkage=None,
        shrinkage_grid=shrinkage_grid,
        max_iter=1000,
        n_random_scree=n_random_scree,
    ).fit(X)

    rows = [
        {
            "n_components": k,
            "ridge": ridge,
            "mean_cv_error": _cv_error(
                X, Y, folds, n_folds, joint_fits, X_test_imputed, k, ridge, par
            ),
        }
        for k in k_grid
        for ridge in ridge_grid
    ]
    table = pd.DataFrame(rows)
    best = table.sort_values(["mean_cv_error", "ridge", "n_components"], na_position="last").iloc[0]
    return {
        "best": {"n_components": int(best.n_components), "ridge": float(best.ridge)},
        "n_components_X": covX.n_components_,
        "shrinkage_X": covX.shrinkage_,
        "cv_table": table,
    }


def _objective(X, fit, par=None):
    from robpy.covariance.cellrcov import cellpca_objective

    n = X.shape[0]
    fit = np.tile(fit, (n, 1)) if np.ndim(fit) == 1 else fit
    return cellpca_objective(X, fit, 1.5, par)


def _fit_joint_by_fold(X, Y, k_grid, folds, n_folds) -> dict:
    """cellRCov of (X, Y) on the training folds, for each k."""
    fits = {}
    for k in k_grid:
        for fold in range(n_folds):
            train = folds != fold
            try:
                fits[(k, fold)] = (
                    CellMRRegression(n_components=k, ridge=0.1).fit(X[train], Y[train]).cellrcov_
                )
            except (ValueError, np.linalg.LinAlgError) as e:
                logger.warning(f"cellRCov failed in fold {fold} for k={k}: {e}")
    return fits


def _impute_test_folds(X, folds, n_folds, joint_fits, k_reference, k_grid_X, n_random_scree):
    """Impute the test predictors of each fold by cellPCA of the training predictors, with
    the number of components chosen per fold by the random scree rule."""
    n, p = X.shape
    thresholds = None
    if len(k_grid_X) > 1:
        thresholds = random_scree(
            round(n / n_folds * (n_folds - 1)), p, max(k_grid_X), n_random_scree
        )
    imputed = {}
    for fold in range(n_folds):
        reference = joint_fits.get((k_reference, fold))
        if reference is None:
            continue
        train, test = folds != fold, folds == fold
        scales = reference.scales_[:p]
        X_train = X[train] / scales
        models = [CellPCA(k, principal_axes=False).fit(X_train) for k in k_grid_X]
        model = models[0]
        if len(k_grid_X) > 1:
            objectives = [_objective(X_train, np.nanmedian(X_train, axis=0))] + [
                _objective(X_train, m.fitted_, m.rho2_par_) for m in models
            ]
            below = np.flatnonzero(np.abs(np.diff(objectives)) < thresholds)
            k_fold = int(below[0]) if below.size else max(k_grid_X)
            model = models[min(max(1, k_fold), len(models)) - 1]
        imputed[fold] = model.impute(X[test] / scales) * scales
    return imputed


def _cv_error(X, Y, folds, n_folds, joint_fits, X_test_imputed, k, ridge, par) -> float:
    """Mean robust prediction error over the test folds; nan if fewer than half of the cells
    get weight."""
    p = X.shape[1]
    predictions, responses = [], []
    for fold in range(n_folds):
        fit = joint_fits.get((k, fold))
        if fit is None or fold not in X_test_imputed:
            continue
        B, b0, _ = _regression_from_covariance(fit.covariance_, fit.location_, p, ridge)
        predictions.append(X_test_imputed[fold] @ B + b0)
        responses.append(Y[folds == fold])
    error = robust_prediction_error(np.vstack(responses), np.vstack(predictions), par)
    if error["neff"].sum() / error["M"].sum() < 0.5:
        return np.nan
    return float(np.mean(error["cv"]))
