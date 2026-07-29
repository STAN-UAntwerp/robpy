from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import median_abs_deviation, chi2

from robpy.pca.base import RobustPCA
from robpy.pca.macropca import MacroPCA
from robpy.preprocessing.data_cleaner import DataCleaner
from robpy.covariance import DetMCD
from robpy.univariate import UnivariateMCD
from robpy.utils.rho import TanhWrapping, scale_tanh


def _up_uv_geninv(
    X: np.ndarray,
    mu: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    W: np.ndarray | None,
    weights_c: np.ndarray,
    update_V: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the scores ``U`` and loadings ``V`` by weighted generalized inverse.

    Port of ``upUVgeninv`` in the R cellPCA code. The V-update uses the full
    weights ``W``; the U-update uses only the cellwise weights ``weights_c``.
    """
    n, p = X.shape
    Xc = X - mu  # broadcasts over rows
    if update_V:
        Zimp = W * Xc
        Ut = U.T
        V = np.array(
            [np.linalg.pinv(Ut @ (W[:, ii, None] * U)) @ Ut @ Zimp[:, ii] for ii in range(p)]
        )
    Zimpc = weights_c * Xc
    Vt = V.T
    U = np.array(
        [np.linalg.pinv(Vt @ (weights_c[ii, :, None] * V)) @ Vt @ Zimpc[ii, :] for ii in range(n)]
    )
    return U, V


def _normal_v(U1: np.ndarray, V1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the loadings ``V1`` to an orthonormal basis, adjusting the scores.

    Port of ``normalV`` in the R cellPCA code.
    """
    u, _, vt = np.linalg.svd(V1, full_matrices=False)
    e = u @ vt  # orthonormal polar factor of V1 (p x k)
    signs = np.sign(np.diag(e.T @ V1))
    V2 = e * signs  # scales each column by its sign
    U2 = U1 @ V1.T @ V2
    return U2, V2


def _cellpca_obj(
    Ri: np.ndarray,
    rho1: TanhWrapping,
    rho2: TanhWrapping,
    scale_1: np.ndarray,
    scale_2: float,
    M: np.ndarray,
) -> float:
    """Evaluate the cellPCA objective function.

    Port of ``cellPCAobj`` in the R cellPCA code. Assumes ``Ri = X - mu - U V^T``.
    """
    n, p = Ri.shape
    scale1mat = np.broadcast_to(scale_1, (n, p))
    Ri2 = rho1.rho(Ri / scale1mat)
    Ri3 = Ri2 * scale1mat**2 * M
    m_i = M.sum(axis=1)
    rowti = np.sqrt(np.nansum(Ri3, axis=1) / m_i)
    d03 = rho2.rho(rowti / scale_2) * m_i
    return float(scale_2**2 * np.sum(d03) / np.sum(M))


def _get_tuning_const_rho2(
    p: int, b_1: float = 1.5, qmax: int = 10, random_state: int = 10
) -> np.ndarray:
    """Compute the tuning constants ``(bb, cc, q1, q2)`` for the casewise rho (rho_2).

    Port of ``getTuningConstRho2`` in the R cellPCA code. The constants are a
    calibration obtained from a large standard-normal sample; because R's and
    NumPy's random number generators differ, the resulting constants match the R
    values only up to Monte-Carlo error (which the cellPCA fit is insensitive to).
    """
    rng = np.random.default_rng(random_state)
    n = 10000
    d = int(np.ceil(min(p / 2, qmax)))
    Ri = rng.standard_normal((n, d))
    rho1 = TanhWrapping(b=b_1)
    scale_1 = np.array([scale_tanh(Ri[:, j]) for j in range(d)])
    Ri2 = rho1.rho(Ri / scale_1)
    Ri3 = Ri2 * scale_1**2
    rowti = np.sqrt(Ri3.sum(axis=1) / d)
    scale_2 = scale_tanh(rowti)
    rowti = rowti / scale_2
    bb = max(np.quantile(rowti, 0.7), 0.0)
    cc = max(np.quantile(rowti, 0.99), 0.3)
    if bb < 100:
        rho2 = TanhWrapping(bb=bb, cc=cc)
        q1, q2 = rho2.q1, rho2.q2
    else:
        q1 = q2 = np.nan
    return np.array([bb, cc, q1, q2])


def _cellpca_predict(
    X: np.ndarray,
    V: np.ndarray,
    mu: np.ndarray,
    scale_1: np.ndarray,
    scale_2: float,
    b_1: float,
    par: np.ndarray,
    max_iter: int,
    nu: float,
    U0: np.ndarray | None = None,
) -> dict:
    """Project new observations onto a fitted cellPCA model by IRLS on the scores.

    Port of ``cellPCApredict`` in the R cellPCA code (with the column scale already applied).
    ``X`` is assumed to be in the standardized model space.
    """
    n, p = X.shape
    k = V.shape[1]
    scale1mat = np.broadcast_to(scale_1, (n, p))
    rho1 = TanhWrapping(b=b_1)
    rho2 = TanhWrapping(bb=par[0], cc=par[1], q1=par[2], q2=par[3])

    na_mask = np.isnan(X)
    M = (~na_mask).astype(float)
    X = X.copy()

    if U0 is None:
        if na_mask.any():
            U = np.zeros((n, k))
            for i in range(n):
                obs = ~na_mask[i]
                U[i, :] = (X[i, obs] - mu[obs]) @ V[obs, :]
        else:
            U = (X - mu) @ V
    else:
        U = U0

    UVt_old = U @ V.T
    if na_mask.any():
        X[na_mask] = UVt_old[na_mask]

    Ri = X - mu - UVt_old
    Ri2 = Ri / scale1mat
    weights_c = rho1.weight(Ri2) * M

    diffU = np.inf
    it = 0
    while diffU > nu and it < max_iter:
        U_old = U
        U, _ = _up_uv_geninv(X, mu, U_old, V, None, weights_c, update_V=False)
        diffU = np.sqrt(np.sum((U - U_old) ** 2)) / (np.sqrt(np.sum(U_old**2)) + 1e-20)
        it += 1
        Ri = X - mu - U @ V.T
        Ri2 = Ri / scale1mat
        weights_c = rho1.weight(Ri2) * M

    Xfit = mu + U @ V.T
    Ri2 = rho1.rho(Ri2)
    Ri3 = Ri2 * scale1mat**2 * M
    m_i = M.sum(axis=1)
    rowti = np.sqrt(np.nansum(Ri3, axis=1) / m_i)
    weights_r = rho2.weight(rowti / scale_2)
    W = weights_r[:, None] * weights_c * M
    if na_mask.any():
        W[na_mask] = 0.0
    Ximp = Xfit + weights_c * (X - Xfit)
    return {
        "U": U,
        "Xfit": Xfit,
        "Ximp": Ximp,
        "weights_c": weights_c,
        "weights_r": weights_r,
        "W": W,
    }


class CellPCA(RobustPCA):
    def __init__(
        self,
        *,
        n_components: int = 2,
        b_1: float = 1.5,
        nu: float = 1e-6,
        max_iter: int = 1000,
        max_col_frac: float = 0.5,
        signflip: bool = True,
        update_mu: bool = True,
        v_est: bool = True,
        par: np.ndarray | None = None,
        scale: bool | np.ndarray = False,
    ):
        """Cellwise and casewise robust PCA (cellPCA).

        Principal component analysis that is robust against both cellwise outliers (individual
        anomalous cells) and casewise outliers (anomalous rows), fitted by iteratively reweighted
        least squares with a wrapping M-estimator, and initialized with
        :class:`~robpy.pca.macropca.MacroPCA`. Python port of ``cellPCA`` from the R code
        accompanying Leyder et al. (2024).

        Args:
            n_components (int, optional): number of principal components ``k``. Defaults to 2.
            b_1 (float, optional): tuning constant of the cellwise rho (rho_1). Defaults to 1.5.
            nu (float, optional): convergence tolerance of the IRLS loop. Defaults to 1e-6.
            max_iter (int, optional): maximum number of IRLS iterations. Defaults to 1000.
            max_col_frac (float, optional): upper bound on the fraction of zero cellwise weights
                allowed in any column before falling back to the previous iteration.
                Defaults to 0.5.
            signflip (bool, optional): flip each loading so its largest entry is positive.
                Defaults to True.
            update_mu (bool, optional): whether to update the center during the fit.
                Defaults to True.
            v_est (bool, optional): whether to perform the final step estimating the principal
                directions and center via a robust (MCD) rotation. Defaults to True.
            par (np.ndarray | None, optional): tuning vector ``(bb, cc, q1, q2)`` for the casewise
                rho (rho_2); computed internally when None. Defaults to None.
            scale (bool | np.ndarray, optional): robustly standardize the columns before fitting
                (``True``), leave them unchanged (``False``) or use a given vector of scales.
                Defaults to False.

        Attributes:
            components_ (np.ndarray): loadings of shape (n_features_kept, n_components_). They
                act on the *standardized* data, i.e. on ``X / colscale_``.
            location_ (np.ndarray): robust center, in the **original** units of the data.
            explained_variance_ (np.ndarray): variances of the scores, in the standardized space;
                they match the output of :meth:`~robpy.pca.base.RobustPCA.transform`. With
                ``v_est=True`` these are the eigenvalues of the robust score covariance and are
                decreasing; with ``v_est=False`` the loadings are not rotated to the principal
                axes, so these are the rowwise-weighted variances of the score columns and need
                not be decreasing.
            n_components_ (int): number of components retained (equal to ``n_components``).
            colscale_ (np.ndarray | None): column scales used to standardize the data, or None
                when ``scale=False``.
            scores_ (np.ndarray): robust IRLS scores ``U``. These are *not* the same as
                ``transform(X)``: the latter simply projects the observed data and equals
                ``distances()["U_orig"]``, while ``scores_`` downweights outlying cells.
            fitted_, imputed_, cell_residuals_, residuals_scaled_ (np.ndarray): fitted values,
                imputed values and (standardized) cellwise residuals, in the **standardized**
                space; multiply by ``colscale_`` to return to the original units.
            columns_kept_ (np.ndarray): positional indices of the columns of the input data that
                survived the :class:`~robpy.preprocessing.data_cleaner.DataCleaner` step. All
                other fitted attributes refer to these columns only, and
                :meth:`~robpy.pca.base.RobustPCA.transform` accepts new data with either the
                original or the retained columns.

        Notes:
            ``DataCleaner`` may also drop *rows* (those with too many missing values), in which
            case the rowwise attributes have fewer rows than the input data.

            Fidelity relative to the original R implementation (validated on the octane and glass
            data sets):

            - The IRLS core is exact: given the same initialization it reproduces R's fitted
              values, residuals and imputations to machine precision.
            - The fit is initialized with :class:`~robpy.pca.macropca.MacroPCA`, which relies on
              robpy's :class:`~robpy.outliers.ddc.DDC`. robpy's DDC differs slightly from cellWise's
              C++ DDC in its prediction step (neighbour selection / combination), so a small
              fraction of cells are flagged differently. The IRLS largely absorbs this: fitted
              values match R to about ``1e-5`` of the data scale on typical cells, with larger
              deviations only at heavily contaminated cells.
            - The final ``v_est`` rotation uses robpy's :class:`~robpy.covariance.mcd.DetMCD`, which
              can select a slightly different subset than rrcov's ``CovMcd``. As a result the
              loadings match R to about ``1e-3`` and the eigenvalues to a few percent. Because the
              fitted values are invariant to this rotation, they (and the outlier diagnostics) are
              unaffected.

        References:
            - Leyder, S., Raymaekers, J., Rousseeuw, P. J., Servotte, T., & Verdonck, T. (2024).
              RobPy: a Python Package for Robust Statistical Methods. arXiv:2411.01954.
        """
        super().__init__(n_components=n_components)
        self.b_1 = b_1
        self.nu = nu
        self.max_iter = max_iter
        self.max_col_frac = max_col_frac
        self.signflip = signflip
        self.update_mu = update_mu
        self.v_est = v_est
        self.par = par
        self.scale = scale

    def fit(  # noqa: C901
        self,
        X: np.ndarray | pd.DataFrame,
        U0: np.ndarray | None = None,
        V0: np.ndarray | None = None,
        mu0: np.ndarray | None = None,
    ) -> "CellPCA":
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        self.n_features_in_ = X_df.shape[1]
        cleaned = DataCleaner(min_n_rows=3).fit_transform(X_df)
        self.columns_kept_ = X_df.columns.get_indexer(cleaned.columns)
        X = np.asarray(cleaned, dtype=float)
        n, p = X.shape
        k = int(self.n_components)
        if k < 1 or k > min(n, p):
            raise ValueError("k must be between 1 and min(n, p)")

        # optional robust column standardization
        Xorig = X.copy()
        if isinstance(self.scale, bool):
            if self.scale:
                colscale = np.array(
                    [
                        scale_tanh(X[:, j][~np.isnan(X[:, j])] - np.nanmedian(X[:, j]))
                        for j in range(p)
                    ]
                )
            else:
                colscale = None
        else:
            colscale = np.asarray(self.scale, dtype=float)
        if colscale is not None:
            X = X / colscale

        rho1 = TanhWrapping(b=self.b_1)
        par = _get_tuning_const_rho2(p, self.b_1) if self.par is None else np.asarray(self.par)
        rho2 = TanhWrapping(bb=par[0], cc=par[1], q1=par[2], q2=par[3])

        # --- initialization: MacroPCA by default, or user-provided estimates ---
        if U0 is None:
            macro = MacroPCA(n_components=k, kmax=k + 1, scale=False, alpha=0.5).fit(X)
            U = macro.full_scores_
            V = macro.components_
            mu = macro.location_
        else:
            U, V, mu = np.asarray(U0), np.asarray(V0), np.asarray(mu0)

        na_mask = np.isnan(X)
        ind_NA = na_mask.any()
        M = (~na_mask).astype(float)

        UVt_old = U @ V.T
        Ri = X - mu - UVt_old
        scale_1 = np.array([scale_tanh(Ri[:, j][~np.isnan(Ri[:, j])]) for j in range(p)])
        scale1mat = np.broadcast_to(scale_1, (n, p))
        Ri2 = Ri / scale1mat
        weights_c = rho1.weight(Ri2)
        Ri2 = rho1.rho(Ri2)
        Ri3 = Ri2 * scale1mat**2 * M
        m_i = M.sum(axis=1)
        rowti = np.sqrt(np.nansum(Ri3, axis=1) / m_i)
        scale_2 = scale_tanh(rowti)
        weights_r = rho2.weight(rowti / scale_2)
        W = weights_r[:, None] * weights_c * M

        X = X.copy()
        if ind_NA:
            W[na_mask] = 0.0
            weights_c[na_mask] = 0.0
            X[na_mask] = UVt_old[na_mask] + 1e15 * scale1mat[na_mask]

        obj = _cellpca_obj(Ri, rho1, rho2, scale_1, scale_2, M)
        obj_vec = [obj]

        # --- IRLS loop ---
        diff = np.inf
        it = 1
        while diff > self.nu and it <= self.max_iter:
            U_old, V_old, W_old, mu_old = U, V, W, mu
            U, V = _up_uv_geninv(X, mu, U_old, V_old, W_old, weights_c, update_V=True)
            U, V = _normal_v(U, V)

            if self.update_mu:
                A = W.sum(axis=0)
                if np.min(A / n) < 1e-5:
                    raise ValueError(f"Variable {int(np.argmin(A))} has tiny average weight")
                A = 1.0 / A
                mu = (W * (X - U @ V.T)).sum(axis=0) * A
                munew = (W * X).sum(axis=0) * A
                unew = V.T @ (munew - mu)
                munew = mu + V @ unew
                U = U - unew
                mu = munew

            UVt = U @ V.T
            Ri = X - mu - UVt
            Ri2 = Ri / scale1mat
            weights_c = rho1.weight(Ri2)
            Ri2 = rho1.rho(Ri2)
            Ri3 = Ri2 * scale1mat**2 * M
            rowti = np.sqrt(np.nansum(Ri3, axis=1) / m_i)
            weights_r = rho2.weight(rowti / scale_2)
            W = weights_c * weights_r[:, None] * M
            if ind_NA:
                W[na_mask] = 0.0
                weights_c[na_mask] = 0.0

            diff = np.sqrt(np.mean((UVt - UVt_old) ** 2)) / (np.sqrt(np.mean(UVt_old**2)) + 1e-20)
            UVt_old = UVt
            it += 1
            obj = _cellpca_obj(Ri, rho1, rho2, scale_1, scale_2, M)
            obj_vec.append(obj)

            if self.max_col_frac < 1:
                nflags = np.sum(W == 0, axis=0)
                if nflags.max() > self.max_col_frac * n:
                    U, V, mu, W = U_old, V_old, mu_old, W_old
                    it = self.max_iter + 1

        # --- V_est: robust rotation of the PC subspace ---
        eigenvalues = None
        if self.v_est:
            U, V, mu, eigenvalues = self._estimate_directions(
                X, U, V, mu, scale_1, scale_2, par, weights_r
            )

        if self.signflip:
            colsigns = _col_signs(V)
            V = V * colsigns
            U = U * colsigns

        if eigenvalues is None:
            # Without the v_est rotation the loadings are not the principal axes of the scores,
            # so report the rowwise-weighted variances of the score columns instead. They play
            # the same role in the score distances but need not be decreasing.
            total_weight = weights_r.sum()
            weights = weights_r if total_weight > 0 else np.ones(n)
            weights = weights / weights.sum()
            eigenvalues = (weights[:, None] * (U - weights @ U) ** 2).sum(axis=0)

        Xfit = mu + U @ V.T
        Ximp = Xfit + weights_c * (X - Xfit)

        # final standardized residuals for graphical display
        scale_1_end = np.array([scale_tanh(Ri[:, j][~np.isnan(Ri[:, j])]) for j in range(p)])
        Ri_std = Ri / np.broadcast_to(scale_1_end, (n, p))
        Ri2 = rho1.rho(Ri_std)
        Ri3 = Ri2 * np.broadcast_to(scale_1_end**2, (n, p))
        rowres = np.sqrt(np.nansum(Ri3, axis=1) / m_i)
        scale_2_end = scale_tanh(rowres)
        rowres_std = rowres / scale_2_end

        # --- store results ---
        self.n_components_ = k
        self.components_ = V
        # mu lives in the standardized space; location_ is exposed in the original units
        self.location_ = mu if colscale is None else mu * colscale
        self.scores_ = U
        self.explained_variance_ = eigenvalues
        self.fitted_ = Xfit
        self.imputed_ = Ximp
        self.cell_residuals_ = Ri
        self.residuals_scaled_ = Ri_std
        self.row_residuals_scaled_ = rowres_std
        self.cellwise_weights_ = weights_c
        self.rowwise_weights_ = weights_r
        self.scale_1_ = scale_1
        self.scale_2_ = scale_2
        self.scale_1_end_ = scale_1_end
        self.scale_2_end_ = scale_2_end
        self.par_ = par
        self.objective_ = obj_vec[-1]
        self.colscale_ = colscale
        self.Xorig_ = Xorig
        self._X_internal = X
        return self

    @property
    def _column_scales(self) -> np.ndarray | float:
        """Column scales used during the fit, so that :meth:`~robpy.pca.base.RobustPCA.transform`
        and :meth:`~robpy.pca.base.RobustPCA.project` can accept data in the original units."""
        return 1.0 if self.colscale_ is None else self.colscale_

    @property
    def _location_scaled(self) -> np.ndarray:
        """The center in the standardized space the IRLS machinery operates in."""
        return self.location_ if self.colscale_ is None else self.location_ / self.colscale_

    def predict(self, X: np.ndarray | pd.DataFrame, nu: float | None = None) -> dict:
        """Project new observations onto the fitted cellPCA model (out-of-sample).

        Port of ``cellPCApredict``. The column standardization used during fitting (if any) is
        re-applied, so ``X`` should be passed in the original units.

        Args:
            X (np.ndarray | pd.DataFrame): new data with the same columns as the training data.
            nu (float | None, optional): convergence tolerance; uses the fitted value when None.

        Returns:
            dict: with keys ``U`` (scores), ``Xfit`` (fitted values), ``Ximp`` (imputed values),
            ``weights_c``, ``weights_r`` and ``W``. As for the corresponding fitted attributes,
            ``Xfit`` and ``Ximp`` are expressed in the standardized space.
        """
        X = self._prepare_input(X)
        if self.colscale_ is not None:
            X = X / self.colscale_
        nu = self.nu if nu is None else nu
        return _cellpca_predict(
            X,
            self.components_,
            self._location_scaled,
            self.scale_1_,
            self.scale_2_,
            self.b_1,
            self.par_,
            self.max_iter,
            nu,
        )

    def row_residual_cutoffs(
        self, n_clean: int = 1000, crit: float = 0.99, random_state: int = 0
    ) -> tuple[float, float]:
        """Cutoffs for the standardized rowwise residuals used to color the casewise displays.

        Port of ``getCutoffRowres``: a clean reference data set is simulated from the fitted model
        and the ``crit`` quantile of its rowwise residuals gives the lower cutoff (the upper cutoff
        is 1.5 times the lower one). Because it relies on a random sample, the cutoffs match R only
        up to Monte-Carlo error.
        """
        n, p = self.residuals_scaled_.shape
        scale_1 = self.scale_1_end_
        scale_2 = self.scale_2_end_
        rho1 = TanhWrapping(b=self.b_1)
        X0 = self.scores_ @ self.components_.T
        rng = np.random.default_rng(random_state)
        idx = rng.integers(0, n, size=n_clean)
        E = rng.standard_normal((n_clean, p)) * scale_1
        X_clean = X0[idx] + E + self._location_scaled
        pred = _cellpca_predict(
            X_clean,
            self.components_,
            self._location_scaled,
            self.scale_1_,
            self.scale_2_,
            self.b_1,
            self.par_,
            self.max_iter,
            self.nu,
        )
        resid = X_clean - pred["Xfit"]
        Ri3 = rho1.rho(resid / scale_1) * scale_1**2
        rowres = np.sqrt(Ri3.sum(axis=1) / p) / scale_2
        cutoff_1 = float(np.quantile(rowres, crit))
        return cutoff_1, 1.5 * cutoff_1

    def distances(self) -> dict:
        """Casewise distances used by the diagnostic displays (port of ``getDist``).

        Returns a dict with the observed-to-imputed (``d1``), imputed-to-fitted (``d2``) and
        observed-to-projection (``d4``) distances, together with the score distances of the fitted
        (``sd_fit``) and observed (``sd_x``) points and the observed-data scores ``U_orig``.
        All quantities refer to the standardized space; ``U_orig`` equals
        :meth:`~robpy.pca.base.RobustPCA.transform` applied to the training data.
        """
        X = self._X_internal
        V, mu = self.components_, self._location_scaled
        eig = self.explained_variance_
        U_orig = (X - mu) @ V
        sd_x = np.sqrt(np.sum(U_orig**2 / eig, axis=1))
        sd_fit = np.sqrt(np.sum(self.scores_**2 / eig, axis=1))
        Px = U_orig @ V.T + mu
        d1 = np.sqrt(np.nansum((self.imputed_ - X) ** 2, axis=1))
        d2 = np.sqrt(np.nansum((self.imputed_ - self.fitted_) ** 2, axis=1))
        d4 = np.sqrt(np.nansum((X - Px) ** 2, axis=1))
        return {
            "d1": d1,
            "d2": d2,
            "d4": d4,
            "sd_fit": sd_fit,
            "sd_x": sd_x,
            "U_orig": U_orig,
        }

    # ------------------------------------------------------------------ displays
    @staticmethod
    def _residual_cmap():
        return LinearSegmentedColormap.from_list(
            "cellmap", ["#0000cc", "#8d8dde", "#ffffff", "#de8d8d", "#cc0000"]
        )

    def cell_map(
        self,
        darkest_color: float = 4.0,
        row_labels: bool = False,
        column_labels: bool = False,
        figsize: tuple[int, int] = (12, 5),
        ax=None,
    ):
        """Residual cell map: a heatmap of the standardized cellwise residuals.

        Cells are colored on a diverging blue-white-red scale (white at 0), with the color
        saturating at ``darkest_color``. This is the central display of Leyder et al. (2024).
        """
        R = np.clip(self.residuals_scaled_, -darkest_color, darkest_color)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        norm = TwoSlopeNorm(vmin=-darkest_color, vcenter=0.0, vmax=darkest_color)
        mesh = ax.pcolormesh(
            R, cmap=self._residual_cmap(), norm=norm, edgecolors="#dddddd", linewidth=0.2
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("variables")
        ax.set_ylabel("cases")
        if not column_labels:
            ax.set_xticks([])
        if not row_labels:
            ax.set_yticks([])
        ax.figure.colorbar(mesh, ax=ax, fraction=0.02, pad=0.02)
        return ax

    def total_deviation_plot(self, cutoffs=None, figsize=(8, 5), ax=None):
        """Scatter of the standardized rowwise total deviations with a casewise cutoff line."""
        if cutoffs is None:
            cutoffs = self.row_residual_cutoffs()
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            np.arange(len(self.row_residuals_scaled_)),
            self.row_residuals_scaled_,
            color="black",
            s=16,
        )
        ax.axhline(cutoffs[0], color="red", lw=1.5)
        ax.set_xlabel("index")
        ax.set_ylabel(r"$\tilde{t}_i$")
        return ax

    def enhanced_outlier_map(self, crit=0.99, figsize=(6, 5), ax=None, label_out=3):
        """Enhanced outlier map: orthogonal distance (OD) versus score distance (SD).

        Port of ``enhOutlierMap``. Points beyond the SD/OD cutoffs are casewise or orthogonal
        outliers; the ``label_out`` most extreme points are annotated.
        """
        R = self.residuals_scaled_
        OD = np.sqrt(np.nansum(R**2, axis=1))
        dist = self.distances()
        SD = dist["sd_x"]
        cutoff_sd = np.sqrt(chi2.ppf(crit, self.n_components_))
        # OD cutoff from a simulated clean sample (reuse the row-residual machinery)
        n, p = R.shape
        rng = np.random.default_rng(0)
        X0 = self.scores_ @ self.components_.T
        idx = rng.integers(0, n, size=1000)
        E = rng.standard_normal((1000, p)) * self.scale_1_end_
        Xc = X0[idx] + E + self._location_scaled
        pred = _cellpca_predict(
            Xc,
            self.components_,
            self._location_scaled,
            self.scale_1_,
            self.scale_2_,
            self.b_1,
            self.par_,
            self.max_iter,
            self.nu,
        )
        Rc = (Xc - pred["Xfit"]) / self.scale_1_end_
        cutoff_od = np.quantile(np.sqrt(np.nansum(Rc**2, axis=1)), crit)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.scatter(SD, OD, color="black", s=16)
        ax.axvline(cutoff_sd, color="red", lw=1.5)
        ax.axhline(cutoff_od, color="red", lw=1.5)
        ax.set_xlabel("SD")
        ax.set_ylabel(r"$\tilde{OD}$")
        ax.set_title("Enhanced Outlier Map")
        if label_out:
            score = np.maximum(SD / cutoff_sd, OD / cutoff_od)
            for i in np.argsort(-score)[:label_out]:
                ax.annotate(str(i), (SD[i], OD[i]), fontsize=8)
        return ax

    def drop_plot(self, crit=0.99, figsize=(9, 6), ax=None):
        """Drop plot linking each case's observed (SD, OD) to its imputed (SD_fit, d2) point.

        Port of ``dropplot``.
        """
        d = self.distances()
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        ax.vlines(d["sd_x"], 0, d["d4"], color="purple", ls="--", lw=0.8)
        ax.vlines(d["sd_fit"], 0, d["d2"], color="orange", ls="--", lw=0.8)
        for i in range(len(d["sd_x"])):
            ax.plot([d["sd_x"][i], d["sd_fit"][i]], [d["d4"][i], d["d2"][i]], color="gray", lw=0.8)
        ax.scatter(d["sd_x"], d["d4"], color="purple", s=40, label="observed", zorder=3)
        ax.scatter(d["sd_fit"], d["d2"], color="orange", s=40, label="imputed", zorder=3)
        ax.scatter(
            d["sd_x"],
            np.zeros_like(d["sd_x"]),
            color="deepskyblue",
            s=40,
            label="projected",
            zorder=3,
        )
        ax.scatter(
            d["sd_fit"],
            np.zeros_like(d["sd_fit"]),
            color="darkgreen",
            s=40,
            label="predicted",
            zorder=3,
        )
        ax.axvline(np.sqrt(chi2.ppf(crit, self.n_components_)), color="red", lw=1.5)
        ax.set_xlabel("score distance")
        ax.set_ylabel("orthogonal distance")
        ax.legend(loc="upper left")
        return ax

    def functional_cell_map(self, rows, cutoffs=None, figsize=(12, 6), ax=None):
        """Functional residual cell map: the standardized residual curves of selected ``rows``,
        colored by their casewise outlyingness (port of the ``spec_type = "res"`` case of
        ``funcCellMap``)."""
        if cutoffs is None:
            cutoffs = self.row_residual_cutoffs()
        rows = np.atleast_1d(rows)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        t = np.clip((self.row_residuals_scaled_ - cutoffs[0]) / (cutoffs[1] - cutoffs[0]), 0, 1)
        cmap = plt.get_cmap("autumn_r")
        for i in rows:
            ax.plot(self.residuals_scaled_[i], color=cmap(t[i]), lw=1)
        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("variables")
        ax.set_ylabel("standardized residual")
        return ax

    def silhouette_plot(self, cutoffs=None, figsize=(8, 6), ax=None):
        """Silhouette-style plot of the casewise outlyingness (standardized rowwise residuals),
        sorted and colored relative to the casewise cutoff (port of ``silhplot``)."""
        if cutoffs is None:
            cutoffs = self.row_residual_cutoffs()
        s = self.row_residuals_scaled_
        order = np.argsort(s)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        colors = np.where(s[order] > cutoffs[0], "#cc0000", "#4652a3")
        ax.barh(np.arange(len(s)), s[order], color=colors)
        ax.axvline(cutoffs[0], color="red", lw=1.5)
        ax.set_xlabel(r"$\tilde{t}_i$")
        ax.set_ylabel("cases (sorted)")
        return ax

    def _estimate_directions(self, X, U, V, mu, scale_1, scale_2, par, weights_r):
        n = X.shape[0]
        indnot = np.where(weights_r == 0)[0]

        # filter: mark cells more than const*MAD from the column median as missing
        const = 3
        center = np.median(X, axis=0)
        mad = median_abs_deviation(X, axis=0, scale="normal")
        outlier = (X > center + const * mad) | (X < center - const * mad)
        X_new = np.where(outlier, np.nan, X)
        ind_update = np.where(outlier.any(axis=1))[0]

        ind_ncases = np.array([i for i in ind_update if i not in set(indnot)], dtype=int)
        if len(ind_ncases) > 0:
            pred = _cellpca_predict(
                X_new[ind_ncases], V, mu, scale_1, scale_2, self.b_1, par, self.max_iter, self.nu
            )
            U_new = U.copy()
            U_new[ind_ncases] = pred["U"]
        else:
            U_new = U.copy()

        if 0 < len(indnot) < n:
            alpha2 = min((0.5 * n) / (n - len(indnot)), 0.8)
            keep = np.setdiff1d(np.arange(n), indnot)
            rot = _rotate_v(U_new[keep], V, mu, U, alpha2)
        else:
            rot = _rotate_v(U_new, V, mu, U, 0.5)
        return rot["U"], rot["V"], rot["mu"], rot["eigenvalues"]


def _rotate_v(U, V, mu, Utot, alpha):
    """Rotate the loadings ``V`` to the principal axes inside the score space via a robust
    (MCD) covariance of the scores. Port of ``rotateV`` in the R cellPCA code.
    """
    if U.shape[1] == 1:
        umcd = UnivariateMCD(alpha=alpha).fit(U[:, 0])
        mu2 = np.array([umcd.location])
        cov = np.array([[umcd.scale**2]])
    else:
        mcd = DetMCD(alpha=alpha).fit(U)
        mu2 = mcd.location_
        cov = mcd.covariance_
    mu_new = mu + V @ mu2
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(-evals)
    V_sub = evecs[:, order]
    eigenvalues = evals[order]
    V_new = V @ V_sub
    U_new = (Utot - mu2) @ V_sub
    sign_e = np.sign(np.diag(V_new.T @ V))
    V_new = V_new * sign_e
    U_new = U_new * sign_e
    return {"V": V_new, "U": U_new, "mu": mu_new, "eigenvalues": eigenvalues}


def _col_signs(V):
    return np.array([1 if V[np.argmax(np.abs(V[:, j])), j] >= 0 else -1 for j in range(V.shape[1])])


def _get_obj(X: np.ndarray, Xfit: np.ndarray, b_1: float, par: np.ndarray) -> float:
    """cellPCA objective computed from the data and a fitted matrix (port of ``getObj``)."""
    n, p = X.shape
    rho1 = TanhWrapping(b=b_1)
    rho2 = TanhWrapping(bb=par[0], cc=par[1], q1=par[2], q2=par[3])
    M = (~np.isnan(X)).astype(float)
    Ri = X - Xfit
    scale_1 = np.array([scale_tanh(Ri[:, j][~np.isnan(Ri[:, j])]) for j in range(p)])
    scale1mat = np.broadcast_to(scale_1, (n, p))
    Ri3 = rho1.rho(Ri / scale1mat) * scale1mat**2 * M
    m_i = M.sum(axis=1)
    rowti = np.sqrt(np.nansum(Ri3, axis=1) / m_i)
    scale_2 = scale_tanh(rowti)
    d03 = rho2.rho(rowti / scale_2) * m_i
    return float(scale_2**2 * np.sum(d03) / np.sum(M))


def _find_peaks(x: np.ndarray) -> np.ndarray:
    return np.where(np.diff(np.sign(np.diff(x))) < 0)[0] + 2


def _kneedle(x: np.ndarray, y: np.ndarray, sensitivity: float = 1.0) -> float:
    """Knee-point of a curve via the Kneedle algorithm (port of ``kneedle``), assuming a decreasing,
    concave-down curve as produced by the cellPCA objective across ``k``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    data = np.column_stack([x[order], y[order]])
    decreasing = (data[-1, 1] - data[0, 1]) < 0
    maxx, minx = x.max(), x.min()
    dn0 = (data[:, 0] - data[:, 0].min()) / (data[:, 0].max() - data[:, 0].min())
    dn1 = (data[:, 1] - data[:, 1].min()) / (data[:, 1].max() - data[:, 1].min())
    if decreasing:
        differ = np.abs(dn1 - (1 - dn0))
    else:
        differ = np.abs(dn1 - dn0)
    peak_indices = _find_peaks(differ) - 1
    diffx = np.diff(dn0)
    threshold_margin = sensitivity * np.mean(diffx)
    knee = None
    for i in range(len(peak_indices)):
        thr = differ[peak_indices[i]] - threshold_margin
        end = peak_indices[i + 1] if i + 1 < len(peak_indices) else len(differ)
        for j in range(peak_indices[i], end):
            if differ[j] < thr:
                knee = peak_indices[i]
                break
        if knee is not None:
            break
    if knee is None:
        return float(minx)
    return float((maxx - minx) * dn0[knee] + minx)


def select_k(
    X: np.ndarray | pd.DataFrame,
    k_vec,
    b_1: float = 1.5,
    par: np.ndarray | None = None,
    fev: float = 0.9,
    selection: str = "kneedle",
    max_iter: int = 1000,
) -> dict:
    """Select the number of cellPCA components.

    Port of ``selectK``. Fits :class:`CellPCA` for each candidate ``k`` in ``k_vec`` and selects a
    dimension either with the kneedle knee-point rule on the objective values
    (``selection="kneedle"``) or with a fraction-of-explained-variability threshold
    (``selection="fev"``).

    Args:
        X (np.ndarray | pd.DataFrame): the data matrix.
        k_vec (Sequence[int]): candidate numbers of components.
        b_1 (float, optional): tuning constant of the cellwise rho. Defaults to 1.5.
        par (np.ndarray | None, optional): tuning vector for the casewise rho. Defaults to None.
        fev (float, optional): fraction of explained variability for ``selection="fev"``.
        selection (str, optional): ``"kneedle"`` or ``"fev"``. Defaults to "kneedle".
        max_iter (int, optional): maximum IRLS iterations per fit. Defaults to 1000.

    Returns:
        dict: with the selected ``k``, the objective values ``objs`` and the fitted ``models``.
    """
    X = np.asarray(X.values if isinstance(X, pd.DataFrame) else X, dtype=float)
    n, p = X.shape
    k_vec = list(k_vec)
    if par is None:
        par = _get_tuning_const_rho2(p, b_1)

    objs = []
    models = []
    for k in k_vec:
        if k != p:
            model = CellPCA(n_components=k, b_1=b_1, par=par, max_iter=max_iter).fit(X)
            objs.append(_get_obj(X, model.fitted_, b_1, model.par_))
            models.append(model)
        else:
            objs.append(0.0)
            models.append(None)

    mu0 = np.nanmedian(X, axis=0)
    v0 = _get_obj(X, np.broadcast_to(mu0, (n, p)), b_1, par)
    objs = [v0] + objs

    if selection == "fev":
        ratios = 1 - np.array(objs[1:]) / objs[0]
        above = np.where(ratios > fev)[0]
        k = k_vec[above[0]] if len(above) else k_vec[-1]
    else:
        xseq = np.arange(0, len(k_vec) + 1)
        k = int(round(_kneedle(xseq, np.sort(objs)[::-1])))
    return {"k": k, "objs": np.array(objs), "models": models}
