"""Faithful ports of the univariate location and scale estimators of the R package cellWise.

The functions in this module follow ``src/LocScaleEstimators.cpp`` and ``Wrap_cpp`` in
``src/Main.cpp`` of cellWise 2.5.7 line by line, including their constants and finite-sample
corrections, so that robpy's cellwise methods (DDC, cellMCD and the methods built on them)
reproduce the results of cellWise. They are used by the public estimators in
:mod:`robpy.univariate.onestep_m`.

Unlike ``cellWise::estLocScale``, :func:`est_loc_scale` never subsamples: cellWise uses a random
subsample of 25000 values (``nLocScale``) for columns with more observations.

References:
    - Raymaekers, J., & Rousseeuw, P. J. (2023). cellWise: Analyzing Data with Cellwise
      Outliers. R package version 2.5.7.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

MAD_CONSTANT = 1.482602218505602

# tuning constants of the hyperbolic tangent (wrapping) psi function with b = 1.5 and c = 4
TANH_B, TANH_C, TANH_K, TANH_A, TANH_BB = 1.5, 4.0, 4.1517212, 0.7532528, 0.8430849

EST_LOC_SCALE_TYPES = ("1stepM", "hubhub", "wrap", "mcd", "rawmcd", "wrapmedmad")


# ---------------------------------------------------------------- weight and rho functions


def loc_weight_biweight(x: np.ndarray) -> np.ndarray:
    """Biweight weights for the location M-estimator (``locWeightBiweight``)."""
    u = 1 - (x * MAD_CONSTANT / 3) ** 2
    return ((u + np.abs(u)) / 2) ** 2


def loc_weight_huber15(x: np.ndarray) -> np.ndarray:
    """Huber weights with b = 1.5 for the location M-estimator (``locWeightHuber15``)."""
    a = np.abs(x)
    return np.where(a < 1.5, 1.0, 1.5 / np.where(a < 1.5, 1.0, a))


def loc_weight_tanh154(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent weights with b = 1.5 and c = 4 (``locWeightTanh154``)."""
    a = np.abs(x)
    w = np.ones_like(a, dtype=float)
    w[a > TANH_C] = 0.0
    mid = (a >= TANH_B) & (a <= TANH_C)
    w[mid] = (
        np.sqrt(TANH_A * (TANH_K - 1))
        * np.tanh(0.5 * np.sqrt((TANH_K - 1) * TANH_BB**2 / TANH_A) * (TANH_C - a[mid]))
        / a[mid]
    )
    return w


def psi_tanh(
    x: np.ndarray,
    b: float = TANH_B,
    c: float = TANH_C,
    k: float = TANH_K,
    A: float = TANH_A,
    B: float = TANH_BB,
) -> np.ndarray:
    """Psi function of the hyperbolic tangent estimator (``psiTanh``), used for wrapping."""
    x = np.where(np.abs(x) > c, 0.0, np.asarray(x, dtype=float))
    a = np.abs(x)
    return np.where(
        a > b,
        np.sqrt(A * (k - 1)) * np.tanh(0.5 * np.sqrt((k - 1) * B**2 / A) * (c - a)) * np.sign(x),
        x,
    )


def rho_tanh154(x: np.ndarray) -> np.ndarray:
    """Rho function of the hyperbolic tangent estimator (``rhoTanh154``)."""
    return psi_tanh(x) ** 2 / 1.506506


def rho_huber25(x: np.ndarray) -> np.ndarray:
    """Huber rho function with b = 2.5 * qnorm(0.75) (``rhoHuber25``)."""
    return np.minimum(np.asarray(x, dtype=float) ** 2, 2.8433526444973292) / 1.688942410165249


def rho_huber15(x: np.ndarray) -> np.ndarray:
    """Huber rho function with b = 1.5 (``rhoHuber15``)."""
    return np.minimum(np.asarray(x, dtype=float) ** 2, 2.25) / 1.556931


# ---------------------------------------------------------------- one-step M-estimators


def loc_1step_m(
    x: np.ndarray,
    weight_function,
    init_loc: float = np.nan,
    init_scale: float = np.nan,
    prec_scale: float = 1e-12,
) -> float:
    """First step of a location M-estimator, started from (median, MAD) unless an initial
    location and scale are given (``loc1StepM``). Non-finite values are ignored."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    m0 = np.median(x) if not np.isfinite(init_loc) else init_loc
    s0 = MAD_CONSTANT * np.median(np.abs(x - m0)) if not np.isfinite(init_scale) else init_scale
    if s0 > prec_scale:
        w = weight_function((x - m0) / s0)
        return float(np.sum(x * w) / np.sum(w))
    return float(m0)


def scale_1step_m(
    x: np.ndarray, rho_function, init_scale: float = np.nan, prec_scale: float = 1e-12
) -> float:
    """First step of a scale M-estimator around zero (``scale1StepM``), started from the MAD
    around zero unless an initial scale is given. Non-finite values are ignored."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    s0 = MAD_CONSTANT * np.median(np.abs(x)) if not np.isfinite(init_scale) else init_scale
    if s0 < prec_scale:
        return 0.0
    return float(s0 * np.sqrt(np.sum(rho_function(x / s0)) / (0.5 * x.size)))


# ---------------------------------------------------------------- univariate MCD


def uni_mcd(y: np.ndarray, alpha: float = 0.5, center: bool = True) -> dict:
    """Reweighted univariate MCD of cellWise (``uniMcd``).

    Args:
        y (np.ndarray): Univariate data; non-finite values are ignored.
        alpha (float, optional): Fraction of the observations in the h-subset
            (h = ceil(alpha * n)). Defaults to 0.5.
        center (bool, optional): If False, the scale is computed around a fixed center of 0.
            Defaults to True.

    Returns:
        dict: ``loc``, ``scale``, ``rawloc``, ``rawscale`` and the boolean reweighting
        ``weights`` (aligned with the input, False for non-finite values).
    """
    y = np.asarray(y, dtype=float)
    finite = np.flatnonzero(np.isfinite(y))
    out = {
        "loc": 0.0,
        "scale": 1.0,
        "rawloc": 0.0,
        "rawscale": 1.0,
        "weights": np.zeros(y.size, dtype=bool),
    }
    quan = int(np.ceil(finite.size * alpha))
    y = y[finite]
    n = y.size
    length = n - quan + 1

    if not center:
        y2 = y**2
        order = np.argsort(y2, kind="stable")
        y2 = y2[order]
        # cellWise uses the order statistic with (0-based) index quan here
        rawsqscale = y2[quan] / chi2.ppf(quan / n, 1)
        kept = y2 <= rawsqscale * chi2.ppf(0.975, 1)
        out["rawscale"] = np.sqrt(rawsqscale)
        weights = np.zeros(n, dtype=bool)
        weights[order] = kept
        out["weights"][finite] = weights
        sqscale = max(0.0, np.sum(y2[kept])) / (kept.sum() - 1)
        out["scale"] = 1.0835 * np.sqrt(sqscale)
        return out

    if length == 1:
        out["loc"] = out["rawloc"] = float(np.mean(y))
        out["scale"] = out["rawscale"] = float(np.sqrt(np.var(y, ddof=1)))
        return out

    order = np.argsort(y, kind="stable")
    y = y[order]
    # running sums over the windows of quan consecutive order statistics, updated recursively
    # exactly as in the C++ code (so that ties in the minimum are found the same way)
    yl = y.tolist()
    sh = [0.0] * length
    sh2 = [0.0] * length
    sq = [0.0] * length
    sh[0] = float(np.sum(y[:quan]))
    sh2[0] = sh[0] ** 2 / quan
    sq[0] = float(np.sum(y[:quan] ** 2)) - sh2[0]
    for i in range(1, length):
        sh[i] = sh[i - 1] - yl[i - 1] + yl[i + quan - 1]
        sh2[i] = sh[i] ** 2 / quan
        sq[i] = sq[i - 1] - yl[i - 1] ** 2 + yl[i + quan - 1] ** 2 - sh2[i] + sh2[i - 1]
    sq_arr, sh_arr = np.asarray(sq), np.asarray(sh)
    tied = np.flatnonzero(sq_arr == sq_arr.min())
    initmean = float(np.mean(sh_arr[tied]) / quan)
    sqres = (y - initmean) ** 2
    sortsqres = np.sort(sqres)

    if n < 10:
        cfac1 = [4.252, 1.970, 2.217, 1.648, 1.767, 1.482, 1.555][n - 3]
    else:
        cfac1 = n / (n - 3.0) if n % 2 == 0 else n / (n - 3.4)
    rawsqscale = cfac1**2 * sortsqres[quan - 1] / chi2.ppf(quan / n, 1)
    kept = sqres <= rawsqscale * chi2.ppf(0.975, 1)

    out["rawloc"] = initmean
    out["rawscale"] = float(np.sqrt(rawsqscale))
    out["loc"] = float(np.sum(y[kept]) / kept.sum())
    weights = np.zeros(n, dtype=bool)
    weights[order] = kept
    out["weights"][finite] = weights

    tempscale = np.sqrt(max(0.0, np.sum((y[kept] - out["loc"]) ** 2)) / (kept.sum() - 1))
    if n < 16:
        cfac2 = [
            1.475,
            1.223,
            1.253,
            1.180,
            1.181,
            1.140,
            1.143,
            1.105,
            1.114,
            1.098,
            1.103,
            1.094,
            1.093,
        ][n - 3]
    else:
        cfac2 = n / (n - 1.4)
    out["scale"] = float(cfac2 * 1.0835 * tempscale)
    return out


# ---------------------------------------------------------------- estLocScale and wrap


def loc_scale_1d(
    x: np.ndarray,
    type: str = "wrap",
    center: bool = True,
    alpha: float = 0.5,
    prec_scale: float = 1e-12,
) -> tuple[float, float]:
    """Location and scale of one column, as ``cellWise::estLocScale`` (non-finite values are
    ignored). With ``center=False`` the location is 0 and the scale is computed around 0.

    Types: "1stepM" (biweight location and Huber scale), "hubhub" (Huber location and scale),
    "wrap" (univariate MCD followed by one wrapping step for the location), "mcd", "rawmcd"
    and "wrapmedmad" (one wrapping step for location and scale, started from median and MAD).
    """
    x = np.asarray(x, dtype=float)
    if type == "1stepM":
        loc = loc_1step_m(x, loc_weight_biweight, prec_scale=prec_scale) if center else 0.0
        return loc, scale_1step_m(x - loc, rho_huber25, prec_scale=prec_scale)
    if type == "hubhub":
        loc = loc_1step_m(x, loc_weight_huber15, prec_scale=prec_scale) if center else 0.0
        return loc, scale_1step_m(x - loc, rho_huber15, prec_scale=prec_scale)
    if type in ("wrap", "mcd", "rawmcd"):
        uni = uni_mcd(x, alpha, center)
        if type == "mcd":
            return uni["loc"], uni["scale"]
        if type == "rawmcd":
            return uni["rawloc"], uni["rawscale"]
        if not center:
            return 0.0, uni["scale"]
        loc = loc_1step_m(x, loc_weight_tanh154, uni["loc"], uni["scale"], prec_scale)
        return loc, uni["scale"]
    if type == "wrapmedmad":
        loc = loc_1step_m(x, loc_weight_tanh154, prec_scale=prec_scale) if center else 0.0
        scale = scale_1step_m(x - loc, rho_tanh154, prec_scale=prec_scale)
        size = x.size  # cellWise uses the column length, missing values included
        if size < 16:
            cn = [
                1.728728,
                1.329473,
                1.391057,
                1.241474,
                1.222204,
                1.165270,
                1.168463,
                1.130316,
                1.129584,
                1.107986,
                1.107362,
                1.094637,
                1.090304,
            ][size - 3]
        else:
            cn = size / (size - 1.208)
        return loc, scale * cn
    raise ValueError(f"Unknown type '{type}'. Use one of {EST_LOC_SCALE_TYPES}.")


def est_loc_scale(
    X: np.ndarray,
    type: str = "wrap",
    center: bool = True,
    alpha: float = 0.5,
    prec_scale: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Columnwise location and scale, as ``cellWise::estLocScale`` (see :func:`loc_scale_1d`).

    Returns:
        tuple[np.ndarray, np.ndarray]: locations and scales, one per column.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    results = [loc_scale_1d(X[:, j], type, center, alpha, prec_scale) for j in range(X.shape[1])]
    loc = np.array([r[0] for r in results], dtype=float)
    scale = np.array([r[1] for r in results], dtype=float)
    return loc, scale


def wrap(X: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Wrapping transformation of cellWise (``Wrap_cpp``).

    Each column is standardized with the given location and scale, transformed with the
    hyperbolic tangent psi function and rescaled so that the wrapped values have the given
    location and scale as mean and standard deviation. Missing values are replaced by the
    location.
    """
    X = np.asarray(X, dtype=float)
    Xw = X.copy()
    for j in range(X.shape[1]):
        finite = np.isfinite(X[:, j])
        u = psi_tanh((X[finite, j] - loc[j]) / scale[j])
        new_loc, new_scale = np.mean(u), np.std(u, ddof=1)
        col = np.full(X.shape[0], loc[j], dtype=float)
        col[finite] = u * (scale[j] / new_scale) + (loc[j] - new_loc * (scale[j] / new_scale))
        Xw[:, j] = col
    return Xw
