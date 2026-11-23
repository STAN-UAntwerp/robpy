"""MCD estimates with the consistency and small-sample corrections of the R package robustbase
(``covMcd``, used through ``rrcov::CovMcd`` by the cellwise methods of cellWise and of
Centofanti, Hubert & Rousseeuw).

The h-subset itself is found by robpy's FastMCD or DetMCD; this module computes the raw and
reweighted estimates from it exactly as robustbase does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammainc
from scipy.stats import chi2

from robpy.covariance.utils import _mcd_cnp2_tables as tables


def h_alpha_n(alpha: float, n: int, p: int) -> int:
    """Size of the h-subset for a fraction alpha (robustbase ``h.alpha.n``)."""
    n2 = (n + p + 1) // 2
    return int(np.floor(2 * n2 - n + 2 * (n - n2) * alpha))


def mcd_consistency(p: int, alpha: float) -> float:
    """Consistency factor of the MCD covariance at the normal (robustbase ``.MCDcons``)."""
    q = chi2.ppf(alpha, p)
    return float(alpha / gammainc(p / 2 + 1, q / 2))


def _simulated(table, p: int, n: int) -> float | None:
    if not tables.P_MIN <= p <= tables.P_MAX:
        return None
    column = p - tables.P_MIN
    if not tables.N_MIN[column] <= n <= tables.N_MAX[column]:
        return None
    return table[n - tables.N_MIN[column]][column]


def _small_sample(p, n, alpha, coef_875, coef_500, p2, p1, simulated) -> float:
    if p > 2:
        coef_875, coef_500 = np.asarray(coef_875), np.asarray(coef_500)

        def fit(coef):
            y = np.log(-coef[:, 0] / p ** coef[:, 1])
            A = np.column_stack([np.ones(2), -np.log(coef[:, 2] * p**2)])
            return np.linalg.solve(A, y)

        c500, c875 = fit(coef_500), fit(coef_875)
        fp_500 = 1 - np.exp(c500[0]) / n ** c500[1]
        fp_875 = 1 - np.exp(c875[0]) / n ** c875[1]
    elif p == 2:
        fp_500 = 1 - np.exp(p2[0]) / n ** p2[1]
        fp_875 = 1 - np.exp(p2[2]) / n ** p2[3]
    else:
        fp_500 = 1 - np.exp(p1[0]) / n ** p1[1]
        fp_875 = 1 - np.exp(p1[2]) / n ** p1[3]
    if alpha == 0.5:
        value = _simulated(simulated, p, n)
        if value is not None:
            fp_500 = 1 / value
    if alpha <= 0.875:
        fp = fp_500 + (fp_875 - fp_500) / 0.375 * (alpha - 0.5)
    else:
        fp = fp_875 + (1 - fp_875) / 0.125 * (alpha - 0.875)
    return float(1 / fp)


def mcd_small_sample(p: int, n: int, alpha: float) -> float:
    """Small-sample correction of the raw MCD covariance (robustbase ``.MCDcnp2``)."""
    return _small_sample(
        p,
        n,
        alpha,
        [[-0.455179464070565, 1.11192541278794, 2], [-0.294241208320834, 1.09649329149811, 3]],
        [[-1.42764571687802, 1.26263336932151, 2], [-1.06141115981725, 1.28907991440387, 3]],
        [0.673292623522027, 0.691365864961895, 0.446537815635445, 1.06690782995919],
        [0.262024211897096, 0.604756680630497, -0.351584646688712, 1.01646567502486],
        tables.SIM_RAW,
    )


def mcd_small_sample_reweighted(p: int, n: int, alpha: float) -> float:
    """Small-sample correction of the reweighted MCD covariance (``.MCDcnp2.rew``)."""
    return _small_sample(
        p,
        n,
        alpha,
        [[-0.544482443573914, 1.25994483222292, 2], [-0.343791072183285, 1.25159004257133, 3]],
        [[-1.02842572724793, 1.67659883081926, 2], [-0.26800273450853, 1.35968562893582, 3]],
        [3.11101712909049, 1.91401056721863, 0.79473550581058, 1.10081930350091],
        [1.11098143415027, 1.5182890270453, -0.66046776772861, 0.88939595831888],
        tables.SIM_REWEIGHTED,
    )


@dataclass
class MCDResult:
    center: np.ndarray
    cov: np.ndarray
    raw_center: np.ndarray
    raw_cov: np.ndarray
    crit: float
    best: np.ndarray


def _qn(x: np.ndarray) -> float:
    """Qn scale estimator as ``robustbase::Qn`` (consistency constant 2.21914 and the
    finite-sample corrections of robustbase)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    k = (n // 2 + 1) * (n // 2) // 2
    r = 2.21914 * np.partition(np.abs(x[i] - x[j]), k - 1)[k - 1]
    if n <= 12:
        factors = [0.399356, 0.99365, 0.51321, 0.84401, 0.6122, 0.85877]
        factors += [0.66993, 0.87344, 0.72014, 0.88906, 0.75743]
        return float(r * factors[n - 2])
    if n % 2:
        c = (1.60188 + (-2.1284 - 5.172 / n) / n) / n + 1
    else:
        c = (3.67561 + (1.9654 + (6.987 - 77 / n) / n) / n) / n + 1
    return float(r / c)


def _scale_tau2(x: np.ndarray, c1: float = 4.5, c2: float = 3.0) -> float:
    """Tau-scale estimator as ``robustbase::scaleTau2`` (one iteration, consistent)."""
    from scipy.stats import norm

    x = np.asarray(x, dtype=float)
    n = x.size
    mu0 = np.median(x)
    xa = np.abs(x - mu0)
    sigma0 = np.median(xa)
    if sigma0 <= 0:
        return 0.0
    b = c2 * norm.ppf(0.75)
    expected = 2 * ((1 - b**2) * norm.cdf(b) - b * norm.pdf(b) + b**2) - 1
    w = np.maximum(0.0, 1 - (xa / (sigma0 * c1)) ** 2) ** 2
    mu = np.sum(x * w) / np.sum(w)
    rho = np.minimum(((x - mu) / sigma0) ** 2, c2**2)
    return float(sigma0 * np.sqrt(np.sum(rho) / (n * expected)))


def _class_pc(X: np.ndarray):
    """Classical PCA by SVD (``robustbase::classPC`` without sign flip)."""
    center = X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X - center, full_matrices=False)
    rank = int(np.sum(s > max(X.shape) * np.finfo(float).eps * (s[0] if s.size else 0)))
    return center, Vt[:rank].T, s[:rank] ** 2 / (X.shape[0] - 1), rank


def _r6pack(z: np.ndarray, h: int, scalefn) -> list[np.ndarray]:
    """The six deterministic initial orderings of DetMCD (``robustbase::r6pack``)."""
    from scipy.stats import norm, rankdata

    n, p = z.shape

    def initset(P):
        lam = np.array([scalefn(c - np.median(c)) for c in (z @ P).T])
        sqrtcov = P @ (lam[:, None] * P.T)
        sqrtinvcov = P @ (P.T / lam[:, None])
        estloc = np.median(z @ sqrtinvcov, axis=0) @ sqrtcov
        centered = (z - estloc) @ P
        return np.argsort(np.sum((centered / lam) ** 2, axis=1), kind="stable")[:h]

    def vectors(M):
        return np.linalg.eigh(M)[1][:, ::-1]

    ranks = np.apply_along_axis(rankdata, 0, z)
    norms = np.sqrt(np.sum(z**2, axis=1))
    normalized = np.where(norms[:, None] > np.finfo(float).eps, z / norms[:, None], z)
    half = np.argsort(norms, kind="stable")[: int(np.ceil(n / 2))]
    U = np.eye(p)
    for i in range(1, p):
        for j in range(i):
            U[i, j] = U[j, i] = (
                scalefn(z[:, i] + z[:, j]) ** 2 - scalefn(z[:, i] - z[:, j]) ** 2
            ) / 4
    starts = [
        vectors(np.corrcoef(np.tanh(z), rowvar=False)),
        vectors(np.corrcoef(ranks, rowvar=False)),
        vectors(np.corrcoef(norm.ppf((ranks - 1 / 3) / (n + 1 / 3)), rowvar=False)),
        vectors(normalized.T @ normalized),
        vectors(np.cov(z[half], rowvar=False)),
        vectors(U),
    ]
    orderings = []
    for P in starts:
        subset = initset(np.atleast_2d(P))
        center, loadings, eigenvalues, rank = _class_pc(z[subset])
        if rank < p:
            raise ValueError("More than half of the observations lie on a hyperplane.")
        score = (z - center) @ loadings
        distances = np.sum((score / np.sqrt(np.abs(eigenvalues))) ** 2, axis=1)
        orderings.append(np.argsort(distances, kind="stable"))
    return orderings


def _detmcd_subset(X: np.ndarray, h: int, max_csteps: int = 200) -> np.ndarray:
    """h-subset of the deterministic MCD of robustbase (``.detmcd``): C-steps from the six
    initial subsets of r6pack on the data standardized by median and Qn (scaleTau2 for
    n >= 1000); the subset with the smallest determinant."""
    n, p = X.shape
    scalefn = _qn if n < 1000 else _scale_tau2
    median = np.median(X, axis=0)
    scale = np.array([scalefn(c) for c in (X - median).T])
    z = (X - median) / scale
    best, best_objective = None, np.inf
    for ordering in _r6pack(z, h, scalefn):
        subset = ordering[:h]
        previous_objective, previous_subset = None, None
        center = loadings = eigenvalues = None
        for step in range(max_csteps):
            if step > 0:
                score = (z - center) @ loadings
                distances = np.sum((score / np.sqrt(np.abs(eigenvalues))) ** 2, axis=1)
                subset = np.argsort(distances, kind="stable")[:h]
            center, loadings, eigenvalues, rank = _class_pc(z[subset])
            if rank < p:
                raise ValueError("More than h of the observations lie on a hyperplane.")
            objective = np.sum(np.log(eigenvalues))
            if step >= 1 and objective == previous_objective:
                if np.array_equal(subset, previous_subset):
                    break
            previous_objective, previous_subset = objective, subset
        if objective < best_objective:
            best, best_objective = subset, objective
    return np.sort(best)


def _best_subset(X: np.ndarray, h: int, deterministic: bool, random_seed: int | None):
    """Indices of the h-subset with the smallest covariance determinant: the deterministic
    MCD of robustbase (``deterministic``), or the better of it and FastMCD. robustbase uses
    FastMCD only (2000 subsets, 2 initial C-steps, 10 best), whose random search does not
    always find the best subset; FastMCD here takes 4 initial C-steps and refines the 20 best
    subsets."""
    import warnings

    from robpy.covariance import FastMCD

    candidates = [_detmcd_subset(X, h)]
    if not deterministic:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator = FastMCD(
                alpha=h,
                n_initial_subsets=2000,
                n_initial_c_steps=4,
                n_best_subsets=20,
                reweighting=False,
                random_seed=random_seed,
            ).fit(X)
        # without reweighting, the fitted location and covariance are those of the h-subset
        diff = X - estimator.location_
        scatter = np.atleast_2d(estimator.covariance_)
        distances = np.einsum("ij,jk,ik->i", diff, np.linalg.pinv(scatter), diff)
        candidates.append(np.sort(np.argsort(distances, kind="stable")[:h]))
    criteria = [np.linalg.slogdet(np.cov(X[c], rowvar=False))[1] for c in candidates]
    return candidates[int(np.argmin(criteria))]


def robustbase_mcd(
    X: np.ndarray,
    alpha: float = 0.5,
    deterministic: bool = False,
    use_correction: bool = True,
    random_seed: int | None = 0,
) -> MCDResult:
    """MCD as ``robustbase::covMcd(X, alpha, nsamp = 2000 or "deterministic")``.

    Args:
        X (np.ndarray): Data of shape (n, p).
        alpha (float): Fraction alpha; the h-subset has size ``h_alpha_n(alpha, n, p)``.
        deterministic (bool): Use only the deterministic MCD of robustbase to find the
            h-subset (``nsamp = "deterministic"``); otherwise the better of FastMCD and it.
        use_correction (bool): Apply the small-sample corrections.
        random_seed (int | None): Seed of FastMCD.

    Returns:
        MCDResult: reweighted and raw center and covariance, the criterion (log determinant
        of the covariance of the h-subset) and the h-subset.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    h = h_alpha_n(alpha, n, p)
    calpha = mcd_consistency(p, h / n)
    correct = mcd_small_sample(p, n, alpha) if use_correction else 1.0
    cutoff = chi2.ppf(0.975, p)

    if p == 1:
        x = X[:, 0]
        s = np.sort(x)
        sums = np.convolve(s, np.ones(h), "valid")
        sq = np.convolve(s**2, np.ones(h), "valid") - sums**2 / h
        start = int(np.argmin(sq))
        subset = s[start : start + h]
        center = float(subset.mean())
        scale = np.sqrt(calpha * correct) * np.sqrt(np.mean((subset - center) ** 2))
        crit = float(np.log(np.sum(np.sort((x - center) ** 2)[:h]) / max(1, h - 1)))
        weights = ((x - center) / scale) ** 2 <= cutoff
        rew_center = float(x[weights].mean())
        rew_var = float(np.var(x[weights], ddof=1))
        if weights.sum() != n:
            rew_var *= mcd_consistency(1, 0.975) * (
                mcd_small_sample_reweighted(1, n, alpha) if use_correction else 1.0
            )
        best = np.sort(np.argsort(np.abs(x - center), kind="stable")[:h])
        return MCDResult(
            np.array([rew_center]),
            np.array([[rew_var]]),
            np.array([center]),
            np.array([[scale**2]]),
            crit,
            best,
        )

    best = _best_subset(X, h, deterministic, random_seed)
    raw_center = X[best].mean(axis=0)
    subset_cov = np.cov(X[best], rowvar=False)
    crit = float(np.linalg.slogdet(subset_cov)[1])
    raw_cov = subset_cov * calpha * correct
    diff = X - raw_center
    distances = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(raw_cov), diff)
    weights = distances < cutoff
    center = X[weights].mean(axis=0)
    cov = np.cov(X[weights], rowvar=False)
    if weights.sum() != n:
        cov = cov * mcd_consistency(p, 0.975)
        if use_correction:
            cov = cov * mcd_small_sample_reweighted(p, n, alpha)
    return MCDResult(center, cov, raw_center, raw_cov, crit, best)
