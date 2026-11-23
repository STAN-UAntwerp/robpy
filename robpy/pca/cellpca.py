from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm

from robpy.covariance.utils.robustbase_mcd import h_alpha_n, robustbase_mcd
from robpy.outliers.ddc import DDC
from robpy.pca._cellpca_constants import RHO2_PAR, RHO2_PAR_FAST
from robpy.pca.base import RobustPCA
from robpy.univariate._cellwise import est_loc_scale, uni_mcd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------- tanh functions

DEFAULT_TANH = (1.5, 4.0, 1.540793, 0.8622731)  # b, c, q1, q2 of rho_1 (b = 1.5)


def _logcosh(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def rho_tanh(
    x: np.ndarray,
    b: float = 1.5,
    bb: float = 1.5,
    cc: float = 4.0,
    q1: float = 1.540793,
    q2: float = 0.8622731,
) -> np.ndarray:
    """Rho function of the hyperbolic tangent (wrapping) estimator, as ``rho_tanh_cpp_vec``:
    quadratic up to bb, constant from cc on, with x rescaled by 1.5 / b."""
    xx = (1.5 / b) * np.asarray(x, dtype=float)
    dd = bb**2 / 2 + (q1 / q2) * _logcosh(q2 * (cc - bb))
    a = np.abs(xx)
    with np.errstate(invalid="ignore"):
        out = dd - (q1 / q2) * _logcosh(q2 * (cc - a))
        out = np.where(a <= bb, 0.5 * xx**2, out)
        out = np.where(a >= cc, dd, out)
    return out


def weight_tanh(
    x: np.ndarray,
    b: float = 1.5,
    bb: float = 1.5,
    cc: float = 4.0,
    q1: float = 1.540793,
    q2: float = 0.8622731,
) -> np.ndarray:
    """Weight function psi(x) / x of the hyperbolic tangent estimator, as
    ``weight_tanh_cpp_vec`` (nan for non-finite input)."""
    xx = (1.5 / b) * np.asarray(x, dtype=float)
    a = np.abs(xx)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = q1 * np.tanh(q2 * (cc - a)) / a
        out = np.where(a <= bb, 1.0, out)
        out = np.where(a >= cc, 0.0, out)
    return np.where(np.isfinite(xx), out, np.nan)


def _psi_wrap(x, b, c, A, B, k):
    a = np.abs(x)
    out = np.where(
        (a >= b) & (a <= c),
        np.sqrt(A * (k - 1)) * np.tanh(0.5 * np.sqrt((k - 1) * B**2 / A) * (c - a)) * np.sign(x),
        x,
    )
    return np.where(a >= c, 0.0, out)


def calculate_q1q2(b: float, c: float, maxit: int = 500, prec: float = 1e-10) -> tuple:
    """Constants q1, q2 that make the wrapping psi function with corner b and rejection point c
    continuous (``calculateq1q2``)."""
    A = 2 * norm.cdf(c) - 1 - 2 * c * norm.pdf(c)
    B = 2 * norm.cdf(c) - 1
    k = max(1.0, c)
    A_new, B_new, k_new = A, B, k
    for _ in range(maxit):
        A, B, k = A_new, B_new, k_new
        k_new = minimize_scalar(
            lambda y: abs(
                b - np.sqrt(A * (y - 1)) * np.tanh(0.5 * np.sqrt((y - 1) * B**2 / A) * (c - b))
            ),
            bounds=(1 + prec, 1000),
            method="bounded",
            options={"xatol": prec},
        ).x
        A_new = quad(lambda y: _psi_wrap(y, b, c, A, B, k) ** 2 * norm.pdf(y), -c, c)[0]
        B_new = quad(lambda y: abs(_psi_wrap(y, b, c, A, B, k)) * abs(y) * norm.pdf(y), -c, c)[0]
        if max(abs(A_new - A), abs(B_new - B), abs(k_new - k)) < prec:
            break
    q1 = np.sqrt(A_new * (k_new - 1))
    q2 = B_new / 2 * np.sqrt((k_new - 1) / A_new)
    return q1, q2


def _chi_wrap(x, b=1.5, c=4.0, A=0.5536433, B=0.8167726, k=6.580856, a=0.2516216):
    """Chi function of the wrapping M-estimator of scale (``chiWrap``)."""
    ax = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        middle = np.sqrt(A * (k - 1)) * np.tanh(
            0.5 * np.sqrt((k - 1) * B**2 / A) * (np.log(c) - np.log(ax))
        )
    out = np.where(ax <= b, x**2 - 1 + a, middle)
    return np.where(ax >= c, 0.0, out)


def scale_tanh(u: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> float:
    """Wrapping M-estimator of scale around zero (``scale_tanh_new``), started from the
    normalized median absolute value; missing values are ignored."""
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    n = u.size
    s0 = np.median(np.abs(u)) / 0.6745
    if s0 < np.finfo(float).eps:
        return 0.0
    for _ in range(max_iter):
        s1 = s0 * (np.sum(_chi_wrap(u / s0)) + n) / n
        err = abs(s1 - s0) / s0
        s0 = s1
        if err <= tol:
            break
    return float(s0)


def rho2_parameters(p: int, fast: bool = False) -> tuple:
    """Tuning constants (b, c, q1, q2) of the rowwise rho function for p variables
    (``get_tuning_const_rho2``; ``fast`` for ``get_tuning_const_rho2_fast``)."""
    d = int(np.ceil(min(p / 2, 10)))
    return tuple((RHO2_PAR_FAST if fast else RHO2_PAR)[d - 1])


# ---------------------------------------------------------------------------- MacroPCA start


def _trunc_pc(X: np.ndarray, ncomp: int | None = None, center: bool = True) -> dict:
    """Truncated PCA by SVD (``truncPC``): rank, eigenvalues, loadings (sign-flipped so that
    the largest absolute entry is positive) and center."""
    n, d = X.shape
    mu = X.mean(axis=0) if center else np.zeros(d)
    Y = X - mu
    _, s, Vt = np.linalg.svd(Y, full_matrices=False)
    ncomp = min(n, d) if ncomp is None else min(ncomp, n, d)
    s, V = s[:ncomp], Vt[:ncomp].T
    rank = int(np.sum(s > 1e-10))
    if rank == 0:
        raise ValueError("The data has rank zero.")
    V = _flip_columns(V[:, :rank])
    return {"rank": rank, "eigenvalues": s[:rank] ** 2 / (n - 1), "loadings": V, "center": mu}


def _flip_columns(V: np.ndarray) -> np.ndarray:
    """Flip each column so that its largest absolute entry is positive."""
    signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    signs[signs == 0] = 1
    return V * signs


def _uniran(seed: float) -> tuple[float, float]:
    seed = np.floor(seed * 5761) + 999
    quot = np.floor(seed / 65536)
    seed = np.floor(seed) - np.floor(quot * 65536)
    return seed, seed / 65536


def _make_directions(X: np.ndarray, ndirect: int, all_pairs: bool) -> np.ndarray:
    """Differences of pairs of rows (``makeDirections``): all pairs, or ndirect pairs drawn by
    the deterministic pseudo-random generator of cellWise."""
    n = X.shape[0]
    if all_pairs:
        i, j = np.array([(a, b) for a in range(n) for b in range(a + 1, n)]).T
        return X[i] - X[j]
    B = np.zeros((ndirect, X.shape[1]))
    seed = 0.0
    for r in range(ndirect):
        chosen: list[int] = []
        for _ in range(2):
            seed, random = _uniran(seed)
            number = int(np.floor(random * n))
            while number in chosen:
                seed, random = _uniran(seed)
                number = int(np.floor(random * n))
            chosen.append(number)
        B[r] = X[chosen[0]] - X[chosen[1]]
    return B


def _max_angle(P1: np.ndarray, P2: np.ndarray) -> float:
    M = P1.T @ P2 @ P2.T @ P1
    smallest = min(np.min(np.linalg.eigvalsh(M)), 1.0)
    return float(np.arccos(np.sqrt(max(smallest, 0.0))) / (np.pi / 2))


@dataclass
class _MacroPCAModel:
    """The parts of a MacroPCA fit that cellPCA uses."""

    ddc: DDC
    k: int
    loadings: np.ndarray
    center: np.ndarray
    eigenvalues: np.ndarray
    scores: np.ndarray = field(repr=False)


def _macropca(X: np.ndarray, k: int, alpha: float = 0.5, maxdir: int = 250) -> _MacroPCAModel:
    """MacroPCA as used to start cellPCA (``new_macropca`` with ``kmax = k + 1``, no scaling,
    DDC with ``tolProb = 0.99``); returns the scores of the fully imputed data (``scoresfi``),
    the loadings and the center. ``checkDataSet`` is not applied."""
    n, d = X.shape
    kmax = min(k + 1, d)
    h = h_alpha_n(alpha, n, k)
    ddc = DDC(chi2_quantile=0.99, min_unique_values=0).fit(pd.DataFrame(X))
    Ti = ddc.row_scores_
    flagged_rows = np.flatnonzero(np.abs(Ti) > np.sqrt(chi2.ppf(0.99, 1)))
    most_outlying = np.argsort(-np.abs(Ti), kind="stable")[: n - h]
    indrows2 = np.intersect1d(most_outlying, flagged_rows)

    missing = np.isnan(X)
    imputed_cells = ddc.cellwise_outliers_ | missing
    ddc_imputed = ddc.impute(pd.DataFrame(X)).to_numpy()
    Xci = np.where(imputed_cells, ddc_imputed, X)
    Xnai = np.where(missing, Xci, X)
    XOimp = Xci.copy()
    Xind = imputed_cells
    full_rank = _trunc_pc(Xci)["rank"]

    cells_per_row = ddc.cellwise_outliers_.sum(axis=1).astype(float)
    cells_per_row[indrows2] = d
    h_rows = np.argsort(cells_per_row, kind="stable")[:h]
    other_rows = np.setdiff1d(np.arange(n), h_rows)
    Xci[other_rows] = Xnai[other_rows]

    all_directions = n * (n - 1) // 2
    ndir = min(maxdir, all_directions)
    B = _make_directions(Xci, ndir, ndir == all_directions)
    norms = np.linalg.norm(B, axis=1)
    B = B[norms > 1e-12] / norms[norms > 1e-12, None]
    Y = Xci @ B.T
    loc, scale = est_loc_scale(Y, "mcd", alpha=alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.abs((Y - loc) / scale)
    H0 = np.argsort(Z.max(axis=1), kind="stable")
    H0 = H0[~np.isin(H0, indrows2)][:h]

    Xci[other_rows] = XOimp[other_rows]
    svd = _trunc_pc(Xci[H0], ncomp=kmax)
    k = min(k, svd["rank"])
    P = svd["loadings"][:, :k]
    center_h = svd["center"]
    if Xind.any():
        previous = P
        for _ in range(20):
            Xhat = (Xci - center_h) @ P @ P.T + center_h
            Xci[Xind] = Xhat[Xind]
            svd = _trunc_pc(Xci[H0], ncomp=k)
            k = min(k, svd["rank"])
            P, center_h = svd["loadings"][:, :k], svd["center"]
            angle = _max_angle(P, previous)
            previous = P
            if angle <= 0.005:
                break
    Xhat = (Xci - center_h) @ P @ P.T + center_h
    Xci[Xind] = Xhat[Xind]
    Xnai[missing] = Xci[missing]
    not_h0 = np.setdiff1d(np.arange(n), H0)
    Xfi = Xci.copy()
    Xci[not_h0] = Xnai[not_h0]

    if k < full_rank:
        residuals = (Xci - center_h) - (Xci - center_h) @ P @ P.T
        od = np.linalg.norm(residuals, axis=1)
        mcd = uni_mcd(od ** (2 / 3), alpha)
        cutoff = np.sqrt(norm.ppf(0.99, mcd["loc"], mcd["scale"]) ** 3)
        h_star = od <= cutoff
        h_star[indrows2] = False
        svd = _trunc_pc(Xfi[h_star], ncomp=k)
        k = min(svd["rank"], k)
    else:
        h_star = np.zeros(n, dtype=bool)
        h_star[H0] = True

    Xci = Xfi.copy()
    Xci[~h_star] = Xnai[~h_star]
    center = svd["center"].copy()
    rotation = svd["loadings"]
    n1 = int(h_star.sum())
    h1 = h_alpha_n(alpha, n1, k)
    scores = ((Xci[h_star] - svd["center"]) @ svd["loadings"])[:, :k]
    rotation = rotation[:, :k]
    mah = np.sum(scores**2 / svd["eigenvalues"][:k], axis=1)
    old_objective = np.prod(svd["eigenvalues"][:k])
    for _ in range(100):
        subset = scores[np.argsort(mah, kind="stable")[:h1]]
        svd = _trunc_pc(subset, ncomp=k)
        objective = np.prod(svd["eigenvalues"])
        scores = (scores - svd["center"]) @ svd["loadings"]
        center = center + svd["center"] @ rotation.T
        rotation = rotation @ svd["loadings"]
        mah = np.sum(scores**2 / svd["eigenvalues"], axis=1)
        if svd["rank"] == k and abs(old_objective - objective) < 1e-12:
            break
        old_objective = objective
        if svd["rank"] < k:
            k = svd["rank"]

    mcd = robustbase_mcd(scores, alpha=h1 / n1, deterministic=True)
    # as in the reference code, the log determinant of the MCD is compared with the product of
    # the eigenvalues
    if mcd.crit < objective + 1e-16:
        final_cov, final_center = mcd.cov, mcd.center
    else:
        mah = mah / (np.median(mah) / chi2.ppf(0.5, k))
        keep = mah <= chi2.ppf(0.975, k)
        final_center = scores[keep].mean(axis=0)
        final_cov = np.atleast_2d(np.cov(scores[keep], rowvar=False, bias=True))
    eigenvalues, vectors = np.linalg.eigh(final_cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    center = center + final_center @ rotation.T
    loadings = _flip_columns(rotation @ vectors)
    full_scores = (Xfi - center) @ loadings
    return _MacroPCAModel(ddc, k, loadings, center, eigenvalues, full_scores)


def _macropca_predict(model: _MacroPCAModel, X: np.ndarray) -> np.ndarray:
    """Scores of new data after imputing their flagged and missing cells by DDC and by
    iterative projection on the fitted subspace (``MacroPCApredict``, ``scoresfi``)."""
    frame = pd.DataFrame(X)
    flags = model.ddc.predict(frame)
    imputed = model.ddc.impute(frame).to_numpy()
    imputed_cells = flags | np.isnan(X)
    Xfi = np.where(imputed_cells, imputed, X)
    P, center = model.loadings, model.center
    if imputed_cells.any():
        for _ in range(20):
            previous = Xfi[imputed_cells]
            Xhat = (Xfi - center) @ P @ P.T + center
            Xfi[imputed_cells] = Xhat[imputed_cells]
            if np.mean((Xfi[imputed_cells] - previous) ** 2) <= 0.005:
                break
    return (Xfi - center) @ P


# ---------------------------------------------------------------------------- cellPCA


def _mcd_of_scores(U: np.ndarray, alpha: float, random_state: int | None):
    """MCD of the scores for the principal axes (``rrcov::CovMcd(U, alpha, nsamp = 2000)``).
    A separate function so that tests can substitute the random result of the R code."""
    return robustbase_mcd(U, alpha, random_seed=random_state)


def _cellwise_weights(residuals, scale_1, scale_2, b_1, par, M):
    """Cellwise weights, rowwise deviations and rowwise weights from cell residuals."""
    standardized = residuals / scale_1
    standardized[~np.isfinite(standardized)] = 0.0
    weights_c = weight_tanh(standardized, b=b_1)
    rho = rho_tanh(standardized, b=b_1) * scale_1**2 * M
    rowti = np.sqrt(np.nansum(rho, axis=1) / M.sum(axis=1))
    return weights_c, rowti


class CellPCA(RobustPCA):
    def __init__(
        self,
        n_components: int,
        b_1: float = 1.5,
        tol: float = 1e-6,
        max_iter: int = 1000,
        update_location: bool = True,
        principal_axes: bool = True,
        signflip: bool = True,
        alpha_mcd: float = 0.5,
        random_state: int | None = 0,
    ):
        """Cellwise and casewise robust PCA (cellPCA) of Centofanti, Raymaekers, Rousseeuw &
        Hubert (2025), following the authors' R code (``cellPCA``, ``cellPCA_predict``).

        cellPCA fits a k-dimensional subspace by iteratively reweighted least squares with a
        cellwise weight (from the standardized residual of each cell, function rho_1) and a
        rowwise weight (from the total deviation of each row, function rho_2), so that both
        outlying cells and outlying rows are downweighted. Missing values are allowed. It
        starts from MacroPCA. Finally, the principal axes inside the subspace are obtained from
        the MCD of the scores (``principal_axes``).

        Args:
            n_components (int): Number of principal components k.
            b_1 (float, optional): Tuning constant of the cellwise rho function. Defaults to
                1.5.
            tol (float, optional): Convergence tolerance of the iterations (``nu``). Defaults
                to 1e-6.
            max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
            update_location (bool, optional): Whether to update the center in the iterations.
                Defaults to True.
            principal_axes (bool, optional): Whether to compute the principal axes and their
                eigenvalues from the MCD of the scores (``V_est``). Defaults to True.
            signflip (bool, optional): Flip the components so that their largest absolute
                entry is positive. Defaults to True.
            alpha_mcd (float, optional): Fraction alpha of the MCD of the scores. Defaults to
                0.5.
            random_state (int | None, optional): Seed of the FastMCD of the scores.

        Attributes:
            location_ (np.ndarray): Center of the subspace.
            components_ (np.ndarray): Loadings, shape (n_features, n_components).
            explained_variance_ (np.ndarray): Eigenvalues of the robust covariance of the scores
                (only with ``principal_axes``).
            scores_ (np.ndarray): Scores of the training data (U).
            weights_cellwise_ (np.ndarray): Cellwise weights (0 for missing cells).
            weights_rowwise_ (np.ndarray): Rowwise weights.
            fitted_ (np.ndarray): Fitted values ``location_ + scores_ @ components_.T``.
            X_imputed_ (np.ndarray): Data with outlying cells shrunk towards their fit and
                missing cells replaced by their fit.
            scale_1_ (np.ndarray): Cellwise scales of the residuals.
            scale_2_ (float): Scale of the rowwise deviations.
            standardized_residuals_ (np.ndarray): Cell residuals divided by their scales.
            rowwise_residuals_ (np.ndarray): Standardized rowwise deviations.
            mcd_ : The MCD of the scores (center, cov, raw_center, raw_cov).

        References:
            - Centofanti, F., Hubert, M., & Rousseeuw, P. J. (2026). Cellwise and casewise
              robust multivariate regression with inference. arXiv:2605.08002.
            - Centofanti, F., Hubert, M., Raymaekers, J., & Rousseeuw, P. J. (2025). Robust
              principal components by casewise and cellwise weighting. Journal of the American
              Statistical Association.
        """
        super().__init__(n_components=n_components)
        self.b_1 = b_1
        self.tol = tol
        self.max_iter = max_iter
        self.update_location = update_location
        self.principal_axes = principal_axes
        self.signflip = signflip
        self.alpha_mcd = alpha_mcd
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y=None,
        initial: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        scale_1: np.ndarray | None = None,
        scale_2: float | None = None,
        rho2_par: tuple | None = None,
    ) -> CellPCA:
        """Fit cellPCA.

        Args:
            X (np.ndarray | pd.DataFrame): Data of shape (n_samples, n_features), may contain
                missing values.
            initial (tuple | None, optional): Initial scores, loadings and center (U0, V0,
                mu0); by default MacroPCA.
            scale_1, scale_2 (optional): Fixed cellwise and rowwise scales; by default they are
                computed from the initial residuals.
            rho2_par (tuple | None, optional): Tuning constants of rho_2; by default those of
                the reference code for the number of variables.
        """
        X = np.array(X, dtype=float)
        X[np.isinf(X)] = np.nan
        n, p = X.shape
        k = self.n_components
        if not 1 <= k <= min(n, p):
            raise ValueError("n_components must be between 1 and min(n_samples, n_features).")
        b_1 = self.b_1
        par = rho2_parameters(p) if rho2_par is None else tuple(rho2_par)

        if initial is None:
            self._macropca = _macropca(X, k)
            U, V, mu = (
                self._macropca.scores[:, :k],
                self._macropca.loadings[:, :k],
                self._macropca.center,
            )
        else:
            self._macropca = None
            U, V, mu = (np.array(a, dtype=float) for a in initial)
        U, V = U.reshape(n, -1), V.reshape(p, -1)

        missing = np.isnan(X)
        M = (~missing).astype(float)
        UVt_old = U @ V.T
        residuals = X - mu - UVt_old
        computed_scale_1 = scale_1 is None
        if computed_scale_1:
            scale_1 = np.array([scale_tanh(residuals[:, j]) for j in range(p)])
        scale_1 = np.asarray(scale_1, dtype=float)
        weights_c, rowti = _cellwise_weights(residuals, scale_1, None, b_1, par, M)
        computed_scale_2 = scale_2 is None
        if computed_scale_2:
            scale_2 = scale_tanh(rowti)
        weights_r = weight_tanh(rowti / scale_2, 1.5, *par)
        W = weights_c * weights_r[:, None] * M
        W[missing] = 0.0
        weights_c[missing] = 0.0
        # missing cells get very outlying values (their weights are zero)
        X = np.where(missing, UVt_old + 1e15 * scale_1, X)

        diff, iteration = np.inf, 1
        while diff > self.tol and iteration <= self.max_iter:
            U, V = self._update_UV(X, mu, U, V, W, weights_c)
            U, V = self._normalize(U, V)
            if self.update_location:
                total = W.sum(axis=0)
                if np.min(total / n) < 1e-5:
                    raise ValueError(f"Variable {np.argmin(total)} has a tiny average weight.")
                mu = np.sum(W * (X - U @ V.T), axis=0) / total
                mu_new = np.sum(W * X, axis=0) / total
                u_new = V.T @ (mu_new - mu)
                mu = mu + V @ u_new
                U = U - u_new
            UVt = U @ V.T
            residuals = X - mu - UVt
            weights_c, rowti = _cellwise_weights(residuals, scale_1, None, b_1, par, M)
            weights_r = weight_tanh(rowti / scale_2, 1.5, *par)
            W = weights_c * weights_r[:, None] * M
            W[missing] = 0.0
            weights_c[missing] = 0.0
            diff = np.sqrt(np.mean((UVt - UVt_old) ** 2)) / (np.sqrt(np.mean(UVt_old**2)) + 1e-20)
            UVt_old = UVt
            iteration += 1
        self.n_iter_ = iteration - 1

        self.location_ = mu
        self.components_ = V
        self.scale_1_ = scale_1
        self.scale_2_ = float(scale_2)
        self.rho2_par_ = par
        self.explained_variance_ = None
        self.mcd_ = None
        if self.principal_axes:
            U, V, mu = self._principal_axes(X, U, V, mu, weights_r, missing)
        if self.signflip:
            signs = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
            signs[signs == 0] = 1
            V, U = V * signs, U * signs

        fitted = mu + U @ V.T
        X_imputed = fitted + weights_c * np.where(missing, 0.0, X - fitted)
        X = np.where(missing, np.nan, X)
        residuals = X - mu - U @ V.T
        scale_1_end = (
            np.array([scale_tanh(residuals[:, j]) for j in range(p)])
            if computed_scale_1
            else scale_1
        )
        standardized = residuals / scale_1_end
        rho = rho_tanh(standardized, b=b_1) * scale_1_end**2
        rowres = np.sqrt(np.nansum(rho, axis=1) / M.sum(axis=1))

        self.location_ = mu
        self.components_ = V
        self.scores_ = U
        self.weights_cellwise_ = weights_c
        self.weights_rowwise_ = weights_r
        self.W_ = W
        self.missing_mask_ = M
        self.fitted_ = fitted
        self.X_imputed_ = X_imputed
        self.standardized_residuals_ = standardized
        self.scale_1_end_ = scale_1_end
        self.rowwise_residuals_ = rowres / scale_2
        self.n_features_in_ = p
        return self

    @staticmethod
    def _update_UV(X, mu, U, V, W, weights_c):
        """One least squares update of V (with the weights W) and of U (with the cellwise
        weights), by generalized inverses (``upUV_geninv``)."""
        X_centered = X - mu
        Z = W * X_centered
        V = np.vstack(
            [np.linalg.pinv(U.T @ (U * W[:, [j]])) @ (U.T @ Z[:, j]) for j in range(X.shape[1])]
        )
        Zc = weights_c * X_centered
        U = np.vstack(
            [
                np.linalg.pinv(V.T @ (weights_c[i][:, None] * V)) @ (V.T @ Zc[i])
                for i in range(X.shape[0])
            ]
        )
        return U, V

    @staticmethod
    def _normalize(U, V):
        """Orthonormalize V and adjust U to keep U V' (``normalV``)."""
        u, _, vt = np.linalg.svd(V, full_matrices=False)
        E = u @ vt
        signs = np.sign(np.diag(E.T @ V))
        V2 = E * signs
        return U @ V.T @ V2, V2

    def _principal_axes(self, X, U, V, mu, weights_r, missing):
        """Principal axes and eigenvalues inside the subspace from the MCD of the scores, after
        recomputing the scores of rows with marginally outlying cells (``rotate_V``)."""
        n, p = X.shape
        not_used = np.flatnonzero(weights_r == 0)
        median = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - median), axis=0)  # R's mad
        filtered = np.where((X > median + 3 * mad) | (X < median - 3 * mad), np.nan, X)
        update = np.flatnonzero(np.isnan(filtered).any(axis=1))
        update = update[~np.isin(update, not_used)]
        U_new = U.copy()
        if update.size > 0:
            rows = filtered[update]
            if p > 2:
                model = _FixedSubspace(
                    V,
                    mu,
                    self.scale_1_,
                    self.scale_2_,
                    self.b_1,
                    self.rho2_par_,
                    self._macropca,
                    self.max_iter,
                )
                U_new[update] = model.scores(rows, self.tol)
            else:
                U_new[update] = (np.where(np.isnan(rows), 0.0, rows) - mu) @ V
        if 0 < not_used.size < n:
            alpha = min(0.5 * n / (n - not_used.size), 0.8)
            used = np.setdiff1d(np.arange(n), not_used)
            mcd = _mcd_of_scores(U_new[used], alpha, self.random_state)
        else:
            mcd = _mcd_of_scores(U_new, 0.5, self.random_state)
        eigenvalues, V_sub = np.linalg.eigh(mcd.cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, V_sub = eigenvalues[order], V_sub[:, order]
        mu = mu + V @ mcd.center
        V_new = V @ V_sub
        U = (U - mcd.center) @ V_sub
        signs = np.sign(np.diag(V_new.T @ V))
        self.explained_variance_ = eigenvalues
        self.mcd_ = mcd
        return U * signs, V_new * signs, mu

    # ------------------------------------------------------------------------ new data

    def predict_scores(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Scores of new data with fixed subspace, location and scales (``cellPCA_predict``)."""
        return self._fixed_subspace().scores(np.array(X, dtype=float), self.tol)

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Scores of (new) data, computed robustly with the cellwise weights (as
        ``predict_scores``)."""
        return self.predict_scores(X)

    def impute(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Impute outlying and missing cells of new data by shrinking them towards their fit
        (``cellPCA_predict``, ``Ximp``)."""
        return self._fixed_subspace().predict(np.array(X, dtype=float), self.tol)["X_imputed"]

    def _fixed_subspace(self) -> _FixedSubspace:
        if not hasattr(self, "components_"):
            raise ValueError("Model not fitted yet.")
        return _FixedSubspace(
            self.components_,
            self.location_,
            self.scale_1_,
            self.scale_2_,
            self.b_1,
            self.rho2_par_,
            self._macropca,
            self.max_iter,
            self.mcd_,
        )


class _FixedSubspace:
    """cellPCA with a fixed subspace: scores of (new) rows (``cellPCA_predict``)."""

    def __init__(self, V, mu, scale_1, scale_2, b_1, par, macropca, max_iter, mcd=None):
        self.V, self.mu, self.scale_1, self.scale_2 = V, mu, scale_1, scale_2
        self.b_1, self.par, self.macropca, self.max_iter, self.mcd = (
            b_1,
            par,
            macropca,
            max_iter,
            mcd,
        )

    def scores(self, X: np.ndarray, tol: float) -> np.ndarray:
        return self.predict(X, tol)["U"]

    def predict(self, X: np.ndarray, tol: float, penalty: float = 0.0) -> dict:
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X[np.isinf(X)] = np.nan
        n, p = X.shape
        missing = np.isnan(X)
        M = (~missing).astype(float)
        if (M.sum(axis=1) == 0).any():
            raise ValueError("Rows without observed cells need to be removed first.")
        V, mu = self.V, self.mu
        k = V.shape[1]
        if self.macropca is not None:
            U = _macropca_predict(self.macropca, X)[:, :k]
        else:
            U = (np.where(missing, mu, X) - mu) @ V
        X = np.where(missing, U @ V.T, X)
        weights_c = weight_tanh((X - mu - U @ V.T) / self.scale_1, b=self.b_1) * M
        diff, iteration = np.inf, 1
        while diff > tol and iteration <= self.max_iter:
            U_old = U
            Zc = weights_c * (X - mu)
            U = np.vstack(
                [
                    np.linalg.pinv(V.T @ (weights_c[i][:, None] * V) + penalty * np.eye(k))
                    @ (V.T @ Zc[i])
                    for i in range(n)
                ]
            )
            diff = np.sqrt(np.sum((U - U_old) ** 2)) / (np.sqrt(np.sum(U_old**2)) + 1e-20)
            iteration += 1
            standardized = (X - mu - U @ V.T) / self.scale_1
            weights_c = weight_tanh(standardized, b=self.b_1) * M
        fitted = mu + U @ V.T
        rho = rho_tanh(standardized, b=self.b_1) * self.scale_1**2 * M
        rowti = np.sqrt(np.nansum(rho, axis=1) / M.sum(axis=1))
        weights_r = weight_tanh(rowti / self.scale_2, 1.5, *self.par)
        X_imputed = fitted + weights_c * (X - fitted)
        return {
            "U": U,
            "fitted": fitted,
            "X_imputed": X_imputed,
            "weights_c": weights_c,
            "weights_r": weights_r,
        }
