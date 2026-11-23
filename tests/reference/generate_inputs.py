"""Generate the input data for the R reference results in this directory.

Run from the repository root:

    python tests/reference/generate_inputs.py
    Rscript tests/reference/cellwise_reference.R

The data are simulated with elementwise numpy operations only (no ``multivariate_normal``,
whose output depends on the platform's LAPACK), and written with 10 significant digits, so
that robpy and R read exactly the same numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


def _write(name: str, X: np.ndarray, prefix: str = "x"):
    DATA.mkdir(exist_ok=True)
    frame = pd.DataFrame(X, columns=[f"{prefix}{j + 1}" for j in range(X.shape[1])])
    frame.to_csv(DATA / f"{name}.csv", index=False, float_format="%.10g")


def univariate() -> np.ndarray:
    """Columns with outliers, missing values and small sizes (for the finite-sample factors)."""
    rng = np.random.default_rng(5)
    cols = [
        rng.normal(size=200),
        rng.standard_t(2, size=200) * 3 + 10,
        rng.lognormal(size=200),
        rng.normal(size=12),
        rng.normal(size=7) + 2,
    ]
    cols[0][rng.random(200) < 0.1] = np.nan
    cols[1][:15] = 80.0
    X = np.full((200, len(cols)), np.nan)
    for j, c in enumerate(cols):
        X[: c.size, j] = c
    return X


def correlated(n: int, p: int, seed: int, rho: float = 0.6) -> np.ndarray:
    """Equicorrelated normal data from a common factor."""
    rng = np.random.default_rng(seed)
    return np.sqrt(rho) * rng.normal(size=(n, 1)) + np.sqrt(1 - rho) * rng.normal(size=(n, p))


def contaminated(n: int, p: int, seed: int, missing: float = 0.03) -> np.ndarray:
    """Correlated data with cellwise outliers, a few outlying rows and missing values."""
    rng = np.random.default_rng(seed)
    X = correlated(n, p, seed)
    X[: n // 20, 1] += 6.0  # cellwise outliers in one column
    cells = rng.random(X.shape) < 0.03  # scattered cellwise outliers
    X[cells] += rng.choice([-5.0, 5.0], cells.sum())
    X[n // 20 : n // 20 + 5] += 3.0  # outlying rows
    X[rng.random(X.shape) < missing] = np.nan
    return X


def regression(n: int, seed: int) -> np.ndarray:
    """Linear regression data (y last) with cellwise outliers and missing values in the
    predictors and outliers in the response."""
    rng = np.random.default_rng(seed)
    X = correlated(n, 4, seed, rho=0.5) * np.array([1.0, 2.0, 0.5, 1.5]) + np.array([5, 0, -2, 1])
    y = 1.0 + X @ np.array([2.0, -1.0, 0.5, 1.0]) + rng.normal(size=n)
    cells = rng.random(X.shape) < 0.04
    X[cells] += rng.choice([-6.0, 6.0], cells.sum()) * X.std(axis=0)[np.nonzero(cells)[1]]
    y[: n // 25] += 10.0
    X[rng.random(X.shape) < 0.03] = np.nan
    return np.column_stack([X, y])


def multivariate_regression(n: int, seed: int) -> np.ndarray:
    """Multivariate regression data (8 predictors from 2 factors, then 3 responses) with
    cellwise outliers and missing values."""
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 2))
    loadings = np.array(
        [[1.0, 0.8, 0.6, 0.0, 0.2, 0.5, 0.9, 0.3], [0.0, 0.4, 0.7, 1.0, 0.9, 0.5, 0.1, 0.6]]
    )
    X = F @ loadings + 0.5 * rng.normal(size=(n, 8))
    B = rng.normal(size=(8, 3)) / 2
    Y = 1.0 + X @ B + 0.5 * rng.normal(size=(n, 3))
    Z = np.column_stack([X, Y])
    cells = rng.random(Z.shape) < 0.03
    Z[cells] += rng.choice([-6.0, 6.0], cells.sum())
    Z[: n // 20] += 4.0  # outlying rows
    Z[rng.random(Z.shape) < 0.02] = np.nan
    return Z


def main():
    _write("univariate", univariate(), prefix="v")
    _write("ddc_small", contaminated(400, 8, seed=11))
    _write("ddc_neighbors", contaminated(120, 12, seed=12, missing=0.0))
    _write("cellmcd", contaminated(300, 5, seed=13))
    _write("cell_lts", regression(400, seed=14))
    _write("cell_lts_new", regression(100, seed=15))
    _write("cell_mr", multivariate_regression(100, seed=16))
    _write("cell_mr_new", multivariate_regression(30, seed=17))


if __name__ == "__main__":
    main()
