from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

from scipy.stats import chi2, norm

from robpy.pca.base import RobustPCA
from robpy.preprocessing.data_cleaner import DataCleaner
from robpy.outliers.ddc import DDC
from robpy.univariate import UnivariateMCD, CellwiseOneStepM
from robpy.covariance import DetMCD


def h_alpha_n(alpha: float, n: int, p: int) -> int:
    """Size of the h-subset, matching ``robustbase::h.alpha.n``."""
    n2 = (n + p + 1) // 2
    return int(np.floor(2 * n2 - n + 2 * (n - n2) * alpha))


def _flip_columns(loadings: np.ndarray) -> np.ndarray:
    """Flip the sign of each column so its largest-magnitude entry is positive."""
    loadings = loadings.copy()
    for j in range(loadings.shape[1]):
        if loadings[np.argmax(np.abs(loadings[:, j])), j] < 0:
            loadings[:, j] *= -1
    return loadings


def trunc_pc(X: np.ndarray, ncomp: int | None = None, center: bool = True, signflip: bool = True):
    """Truncated principal components via the SVD.

    Port of ``truncPC`` in the cellWise package. Returns a dict with ``rank``,
    ``eigenvalues``, ``loadings`` and ``center``.
    """
    Y = np.asarray(X, dtype=float)
    n, d = Y.shape
    center_vec = Y.mean(axis=0) if center else np.zeros(d)
    Yc = Y - center_vec
    _, s, vt = np.linalg.svd(Yc, full_matrices=False)
    if ncomp is not None:
        ncomp = min(ncomp, n, d)
        s = s[:ncomp]
        vt = vt[:ncomp]
    rank = int(np.sum(s > 1e-10))
    if rank == 0:
        raise ValueError("The data has rank zero")
    eigenvalues = (s[:rank] ** 2) / (n - 1)
    loadings = vt.T[:, :rank]
    if signflip:
        loadings = _flip_columns(loadings)
    return {"rank": rank, "eigenvalues": eigenvalues, "loadings": loadings, "center": center_vec}


def _uniran(seed: float) -> tuple[float, float]:
    """Deterministic pseudo-random generator used by ``makeDirections``."""
    seed = np.floor(seed * 5761) + 999
    quot = np.floor(seed / 65536)
    seed = np.floor(seed) - quot * 65536
    return seed, seed / 65536


def _randomset(n: int, k: int, seed: float) -> tuple[float, np.ndarray]:
    """Draw ``k`` distinct 1-based indices in ``1..n`` (port of ``randomset``)."""
    ranset = np.zeros(k, dtype=int)
    for j in range(k):
        seed, rnd = _uniran(seed)
        num = int(np.floor(rnd * n)) + 1
        if j > 0:
            while np.any(ranset[:j] == num):
                seed, rnd = _uniran(seed)
                num = int(np.floor(rnd * n)) + 1
        ranset[j] = num
    return seed, ranset


def make_directions(data: np.ndarray, ndirect: int, all_dirs: bool = True) -> np.ndarray:
    """Directions through pairs of data points (port of ``makeDirections`` for the
    non-fixed-center case).
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    if all_dirs:
        pairs = np.array(list(itertools.combinations(range(n), 2)))
        return data[pairs[:, 0]] - data[pairs[:, 1]]
    B2 = np.zeros((ndirect, data.shape[1]))
    seed = 0.0
    for r in range(ndirect):
        seed, ranset = _randomset(n, 2, seed)
        B2[r, :] = data[ranset[0] - 1, :] - data[ranset[1] - 1, :]
    return B2


def max_angle(P_true: np.ndarray, P_hat: np.ndarray) -> float:
    """Maximal angle between the column spaces of ``P_true`` and ``P_hat`` (port of
    ``maxAngle``), rescaled so that a right angle equals 1.
    """
    M = P_true.T @ P_hat @ P_hat.T @ P_true
    lam_min = min(np.linalg.eigvalsh(M).min(), 1.0)
    lam_min = max(lam_min, 0.0)
    return float(np.arccos(np.sqrt(lam_min)) / (np.pi / 2))


def _uni_mcd_loc_scale(x: np.ndarray, alpha: float) -> tuple[float, float]:
    """Univariate MCD location/scale (``estLocScale`` type ``mcd``)."""
    est = UnivariateMCD(alpha=alpha).fit(x)
    return est.location, est.scale


class MacroPCA(RobustPCA):
    def __init__(
        self,
        *,
        n_components: int = 2,
        kmax: int = 10,
        alpha: float = 0.5,
        scale: bool | np.ndarray = True,
        maxdir: int = 250,
        distprob: float = 0.99,
        maxiter: int = 20,
        tol: float = 0.005,
        tolprob: float = 0.99,
    ):
        """MacroPCA: robust PCA handling both missing values and cellwise/casewise outliers.

        This is a Python port of ``MacroPCA`` from the R package cellWise, reusing robpy's
        :class:`~robpy.outliers.ddc.DDC`, :class:`~robpy.covariance.mcd.DetMCD` and
        :class:`~robpy.univariate.mcd.UnivariateMCD`. It is used, among others, to initialize
        :class:`~robpy.pca.cellpca.CellPCA`.

        Args:
            n_components (int, optional): number of principal components ``k``. Defaults to 2.
            kmax (int, optional): maximum number of components considered internally.
                Defaults to 10.
            alpha (float, optional): coverage parameter in [0.5, 1]. Defaults to 0.5.
            scale (bool | np.ndarray, optional): whether to robustly standardize the columns before
                fitting (``True``), leave them unchanged (``False``) or use a given vector of column
                scales. Defaults to True.
            maxdir (int, optional): number of directions used to find the initial h-subset.
                Defaults to 250.
            distprob (float, optional): quantile for the orthogonal-distance cutoff.
                Defaults to 0.99.
            maxiter (int, optional): maximum number of imputation iterations. Defaults to 20.
            tol (float, optional): convergence tolerance of the imputation loop. Defaults to 0.005.
            tolprob (float, optional): quantile used by DDC to flag cells. Defaults to 0.99.

        Notes:
            This port reuses robpy's :class:`~robpy.outliers.ddc.DDC`, whose prediction step
            differs slightly from cellWise's C++ DDC, so the flagged/imputed cells - and therefore
            the loadings and eigenvalues - can differ modestly from the R implementation. The
            results are close but not bit-identical; see :class:`~robpy.pca.cellpca.CellPCA` for a
            quantitative comparison.

        References:
            - Hubert, M., Rousseeuw, P. J., & Van den Bossche, W. (2019). MacroPCA: an all-in-one
              PCA method allowing for missing values as well as cellwise and rowwise outliers.
              Technometrics, 61(4), 459-473.
        """
        super().__init__(n_components=n_components)
        self.kmax = kmax
        self.alpha = alpha
        self.scale = scale
        self.maxdir = maxdir
        self.distprob = distprob
        self.maxiter = maxiter
        self.tol = tol
        self.tolprob = tolprob

    def fit(self, X: np.ndarray | pd.DataFrame) -> "MacroPCA":  # noqa: C901
        cutoff1 = np.sqrt(chi2.ppf(0.99, 1))
        # --- checkDataSet ---
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X, dtype=float))
        cleaned = DataCleaner(min_n_rows=3).fit_transform(X_df)
        X = np.asarray(cleaned, dtype=float)
        n, d = X.shape

        k = int(self.n_components)
        kmax = min(self.kmax, d)
        alpha = self.alpha
        h = h_alpha_n(alpha, n, k)

        # --- DDC ---
        ddc = DDC(chi2_quantile=self.tolprob).fit(pd.DataFrame(X))
        cell_mask = ddc.cellwise_outliers_
        std_res = ddc.standardized_residuals_
        raw_t = np.nanmean(chi2.cdf(std_res**2, df=1), axis=1)
        Ti = (raw_t - ddc.t_scale_est.location) / ddc.t_scale_est.scale
        na_mask = np.isnan(X)
        Ximp = np.where(cell_mask | na_mask, ddc.rescaled_predictions_, X)

        indrows1 = np.where(np.abs(Ti) > cutoff1)[0]
        order_ti = np.argsort(-np.abs(Ti))[: n - h]
        indrows2 = np.array([i for i in order_ti if i in set(indrows1)], dtype=int)

        imp_mask = cell_mask | na_mask
        XO = X.copy()
        Xnai = X.copy()
        Xci = X.copy()
        Xci[imp_mask] = Ximp[imp_mask]

        # --- column scaling ---
        if isinstance(self.scale, bool):
            if self.scale:
                scaleX = np.array([CellwiseOneStepM().fit(X[:, j]).scale for j in range(d)])
            else:
                scaleX = np.ones(d)
        else:
            scaleX = np.asarray(self.scale, dtype=float)

        XO = XO / scaleX
        Xnai = Xnai / scaleX
        Xci = Xci / scaleX
        XOimp = Xci.copy()
        Xind = imp_mask.copy()  # cells that were imputed
        Xnai[na_mask] = XOimp[na_mask]

        XciSVD = trunc_pc(Xci, center=True)
        if XciSVD["rank"] == 0:
            raise ValueError("All data points collapse!")

        center = np.zeros(d)
        rot = np.eye(d)

        rowcellind = cell_mask.sum(axis=1).astype(float)
        rowcellind[indrows2] = d
        hrows_low = np.argsort(rowcellind, kind="stable")[:h]
        indComb = np.setdiff1d(np.arange(n), np.unique(hrows_low))
        if len(indComb) > 0:
            Xci[indComb, :] = Xnai[indComb, :]

        # --- initial h-subset H0 via random directions ---
        alldir = n * (n - 1) // 2
        ndir = min(self.maxdir, alldir)
        B = make_directions(Xci, ndir, all_dirs=(ndir == alldir))
        Bnorm = np.linalg.norm(B, axis=1)
        nonz = Bnorm > 1e-12
        A = B[nonz] / Bnorm[nonz][:, None]
        Y = Xci @ A.T
        umcd = np.array([_uni_mcd_loc_scale(Y[:, j], alpha) for j in range(Y.shape[1])])
        Z = np.abs((Y - umcd[:, 0]) / umcd[:, 1])
        H0 = np.argsort(np.max(Z, axis=1), kind="stable")
        H0 = np.array([i for i in H0 if i not in set(indrows2)], dtype=int)[:h]

        if len(indComb) > 0:
            Xci[indComb, :] = XOimp[indComb, :]

        Xcih = Xci[H0, :]
        Xcih_SVD = trunc_pc(Xcih, ncomp=kmax, center=True)
        kmax = min(Xcih_SVD["rank"], kmax)

        k = min(k, Xcih_SVD["rank"])
        Pr = Xcih_SVD["loadings"][:, :k]
        PrPrev = Pr
        mXci = Xcih_SVD["center"]

        # --- imputation loop ---
        if Xind.any() and self.maxiter > 0:
            diff = 100.0
            it = 0
            while it < self.maxiter and diff > self.tol:
                it += 1
                XciC = Xci - mXci
                Tr = XciC @ Pr
                Xcihat = Tr @ Pr.T + mXci
                Xci[Xind] = Xcihat[Xind]
                Xcih = Xci[H0, :]
                Xcih_SVD = trunc_pc(Xcih, ncomp=k, center=True)
                k = min(k, Xcih_SVD["rank"])
                Pr = Xcih_SVD["loadings"][:, :k]
                mXci = Xcih_SVD["center"]
                diff = max_angle(Pr, PrPrev)
                PrPrev = Pr

        XciC = Xci - mXci
        Tr = XciC @ Pr
        Xcihat = Tr @ Pr.T + mXci
        Xci[Xind] = Xcihat[Xind]
        Xnai[na_mask] = Xci[na_mask]
        indrows3 = np.setdiff1d(np.arange(n), H0)
        Xfi = Xci.copy()
        if len(indrows3) > 0:
            Xci[indrows3, :] = Xnai[indrows3, :]

        # --- optional extra reweighting when k < rank ---
        if k < XciSVD["rank"]:
            XciRc = Xci - mXci
            Xcihat = XciRc @ Pr @ Pr.T
            Rdiff = XciRc - Xcihat
            ODh = np.linalg.norm(Rdiff, axis=1)
            umcd_od = UnivariateMCD(alpha=alpha).fit(ODh ** (2 / 3))
            cutoffODh = np.sqrt(
                norm.ppf(self.distprob, loc=umcd_od.location, scale=umcd_od.scale) ** 3
            )
            Hstar = ODh <= cutoffODh
            Hstar[indrows2] = False
            Xcih_SVD = trunc_pc(Xfi[Hstar, :], ncomp=k, center=True)
            k = min(Xcih_SVD["rank"], k)
        else:
            Hstar = np.zeros(n, dtype=bool)
            Hstar[H0] = True

        indrows4 = np.where(~Hstar)[0]
        Xci = Xfi.copy()
        if len(indrows4) > 0:
            Xci[indrows4, :] = Xnai[indrows4, :]

        center = center + Xcih_SVD["center"] @ rot.T
        rot = rot @ Xcih_SVD["loadings"]
        n1 = int(np.sum(Hstar))
        h1 = h_alpha_n(alpha, n1, k)
        Xci1 = (Xci[Hstar, :] - Xcih_SVD["center"]) @ Xcih_SVD["loadings"]
        Xci1 = Xci1[:, :k]
        rot = rot[:, :k]

        eig = Xcih_SVD["eigenvalues"][:k]
        mah = np.sum(Xci1**2 / eig, axis=1)
        oldobj = np.prod(eig)
        for _ in range(100):
            Xcih = Xci1[np.argsort(mah, kind="stable")[:h1], :]
            Xcih_SVD = trunc_pc(Xcih, ncomp=k, center=True)
            obj = np.prod(Xcih_SVD["eigenvalues"])
            Xci1 = (Xci1 - Xcih_SVD["center"]) @ Xcih_SVD["loadings"]
            center = center + Xcih_SVD["center"] @ rot.T
            rot = rot @ Xcih_SVD["loadings"]
            eig = Xcih_SVD["eigenvalues"]
            mah = np.sum(Xci1**2 / eig, axis=1)
            if Xcih_SVD["rank"] == k and abs(oldobj - obj) < 1e-12:
                break
            oldobj = obj
            if Xcih_SVD["rank"] < k:
                k = Xcih_SVD["rank"]

        # --- final MCD / reweighting step ---
        eps = 1e-16
        if k == 1:
            umcd = UnivariateMCD(alpha=h1 / n1).fit(Xci1[:, 0])
            Xci2cov = np.array([[umcd.scale**2]])
            Xci2center = np.array([umcd.location])
            crit = umcd.raw_scale_**2
        else:
            detmcd = DetMCD(alpha=h1 / n1).fit(Xci1)
            crit = detmcd.best_subset.determinant
            Xci2cov = detmcd.covariance_
            Xci2center = detmcd.location_
        if crit < obj + eps:
            pass
        else:
            consistencyfactor = np.median(mah) / chi2.ppf(0.5, k)
            mah = mah / consistencyfactor
            weights = mah <= chi2.ppf(0.975, k)
            Xci2center = np.average(Xci1, axis=0, weights=weights)
            Xci2cov = np.cov(Xci1[weights], rowvar=False, bias=True)

        Xci2cov = np.atleast_2d(Xci2cov)
        Xci2center = np.atleast_1d(Xci2center)
        evals, evecs = np.linalg.eigh(Xci2cov)
        order = np.argsort(-evals)
        eigenvalues = evals[order]
        P6 = evecs[:, order]
        center = center + Xci2center @ rot.T
        loadings = rot @ P6
        loadings = _flip_columns(loadings)

        center = center * scaleX

        # --- store results ---
        self.n_components = k
        self.components_ = loadings
        self.location_ = center
        self.explained_variance_ = eigenvalues
        self.scaleX_ = scaleX
        self.X_NA_imputed_ = Xnai * scaleX
        self.cell_imputed_ = Xci * scaleX
        self.full_imputed_ = Xfi * scaleX
        self.scores_ = (Xnai - center / scaleX) @ loadings
        self.full_scores_ = (Xfi - center / scaleX) @ loadings
        return self
