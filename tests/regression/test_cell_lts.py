"""Tests of cellLTS against the authors' R code (tests/reference/cellLTS_reference.R)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import robpy.regression.cell_lts as cell_lts
from robpy.regression import CellLTSRegression
from robpy.regression.cell_lts import _best_window, _symmetrize

REFERENCE = Path(__file__).parents[1] / "reference"


def _reference(name: str) -> np.ndarray:
    return pd.read_csv(REFERENCE / "expected" / f"{name}.csv").to_numpy()


def _data(name: str = "cell_lts") -> tuple[np.ndarray, np.ndarray]:
    D = pd.read_csv(REFERENCE / "data" / f"{name}.csv").to_numpy()
    return D[:, :4], D[:, 4]


CASES = {
    # case: (arguments, columns of X)
    "cell_lts_default": ({}, slice(0, 4)),
    "cell_lts_none": (dict(symmetrize="none"), slice(0, 4)),
    "cell_lts_iterate": (dict(iterate_location=True), slice(0, 4)),
    "cell_lts_univariate": ({}, slice(0, 1)),
}


@pytest.fixture
def r_orders(monkeypatch):
    """Use the random orderings of the observations that the R code draws."""
    orders = {
        (400, 123): _reference("cell_lts_orders_123"),
        (400, 321): _reference("cell_lts_orders_321"),
    }
    univariate = _reference("cell_lts_orders_123_univariate")
    orders[(univariate.shape[1], 123)] = univariate

    def random_orders(n, k, seed):
        return list(orders[(n, seed)][:k])

    monkeypatch.setattr(cell_lts, "_random_orders", random_orders)


def _use_r_raw_lts(monkeypatch, case: str):
    """Use the raw LTS fit of the R code (its random subsets cannot be reproduced)."""
    raw = _reference(f"{case}_raw_lts")[:, 0]
    monkeypatch.setattr(cell_lts, "_raw_lts_ridge", lambda *args, **kwargs: (raw[0], raw[1:]))


@pytest.mark.parametrize("case", CASES)
def test_cell_lts_matches_r_code_given_its_random_choices(case, r_orders, monkeypatch):
    # given: the same pairs of observations and the same raw LTS fit as the R code
    _use_r_raw_lts(monkeypatch, case)
    kwargs, columns = CASES[case]
    X, y = _data()
    X = X[:, columns]
    X_new = _data("cell_lts_new")[0][:, columns]
    # when
    model = CellLTSRegression(**kwargs).fit(X, y)
    # then: everything else is deterministic and equals the R results
    np.testing.assert_allclose(
        np.r_[model.intercept_, model.coef_], _reference(f"{case}_coefficients")[:, 0], rtol=1e-7
    )
    np.testing.assert_allclose(model.scale_, _reference(f"{case}_scale")[0, 0], rtol=1e-7)
    np.testing.assert_allclose(model.predictor_location_, _reference(f"{case}_mu")[:, 0], rtol=1e-7)
    np.testing.assert_allclose(
        model.predictor_covariance_, _reference(f"{case}_covariance"), rtol=1e-7, atol=1e-10
    )
    np.testing.assert_array_equal(model.W_, _reference(f"{case}_W"))
    np.testing.assert_allclose(model.X_imputed_, _reference(f"{case}_Ximp"), rtol=1e-7)
    np.testing.assert_allclose(model.fitted_values_, _reference(f"{case}_preds")[:, 0], rtol=1e-7)
    np.testing.assert_array_equal(model.case_weights_, _reference(f"{case}_v")[:, 0])
    np.testing.assert_allclose(
        model.predict(X_new), _reference(f"{case}_new_preds")[:, 0], rtol=1e-7
    )


@pytest.mark.parametrize("case", ["cell_lts_default", "cell_lts_none"])
def test_cell_lts_finds_an_lts_fit_at_least_as_good_as_the_r_code(case, r_orders):
    # given: the same pairs of observations as the R code, but robpy's own LTS search
    kwargs, _ = CASES[case]
    X, y = _data()
    model = CellLTSRegression(**kwargs).fit(X, y)
    expected = _reference(f"{case}_coefficients")[:, 0]
    # then: the coefficients are close to those of the R code
    np.testing.assert_allclose(np.r_[model.intercept_, model.coef_], expected, atol=0.02)
    # and the raw LTS objective is not larger than that of the R code's raw fit
    if case == "cell_lts_none":
        X_fit, y_fit = model.X_imputed_, y
    else:
        X_fit, y_fit = _symmetrize(model.X_imputed_, y, "rDNsym", 20, 321)
    n, d = X_fit.shape
    n2 = (n + d + 1) // 2
    h = int(np.floor(2 * n2 - n + 2 * (n - n2) * 0.5))
    complete = np.ones(n, dtype=bool)
    ours = cell_lts._raw_lts_ridge(X_fit, y_fit, complete, h, 1e-4, 123, 500)
    theirs = _reference(f"{case}_raw_lts")[:, 0]
    objective_ours = _best_window(y_fit - X_fit @ ours[1], h)[2]
    objective_theirs = _best_window(y_fit - X_fit @ theirs[1:], h)[2]
    assert objective_ours <= objective_theirs * (1 + 1e-10)


def test_cell_lts_with_its_own_random_choices_is_close_to_r():
    X, y = _data()
    model = CellLTSRegression().fit(X, y)
    expected = _reference("cell_lts_default_coefficients")[:, 0]
    np.testing.assert_allclose(np.r_[model.intercept_, model.coef_], expected, atol=0.1)
    assert abs(model.scale_ - _reference("cell_lts_default_scale")[0, 0]) < 0.05


def test_cell_lts_is_robust_to_cellwise_outliers():
    # given: clean regression data, then 5% of the predictor cells replaced by outliers
    rng = np.random.default_rng(0)
    n, beta = 500, np.array([1.0, -2.0, 0.5])
    X = np.sqrt(0.5) * rng.normal(size=(n, 1)) + np.sqrt(0.5) * rng.normal(size=(n, 3))
    y = 2.0 + X @ beta + 0.5 * rng.normal(size=n)
    X_bad = X.copy()
    cells = rng.random(X.shape) < 0.05
    X_bad[cells] = 10.0
    # when
    model = CellLTSRegression().fit(X_bad, y)
    ols = np.linalg.lstsq(np.column_stack([np.ones(n), X_bad]), y, rcond=None)[0]
    # then
    assert np.max(np.abs(model.coef_ - beta)) < 0.15
    assert np.max(np.abs(ols[1:] - beta)) > 0.3
    assert abs(model.intercept_ - 2.0) < 0.15
    # most outlying cells are flagged
    assert model.cellwise_outliers_[cells].mean() > 0.9


def test_symmetrize_variants():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(30, 2)), rng.normal(size=30)
    for method, rows in [("rDNsym", 30 * 5), ("DNsym", 30 * 5), ("qsym", 30 * 5), ("sym", 435)]:
        D, e = _symmetrize(X, y, method, 5, 1)
        assert D.shape == (rows, 2) and e.shape == (rows,)
    # "sym" gives all pairs i < j as X_i - X_j
    D, e = _symmetrize(X, y, "sym", 5, 1)
    np.testing.assert_allclose(D[0], X[0] - X[1])
    np.testing.assert_allclose(e[-1], y[28] - y[29])
    with pytest.raises(ValueError, match="Unknown symmetrization"):
        _symmetrize(X, y, "other", 5, 1)


def test_cell_lts_handles_missing_values_and_dataframes():
    X, y = _data()
    frame = pd.DataFrame(X, columns=list("abcd"))
    y = y.copy()
    y[:3] = np.nan
    model = CellLTSRegression(n_starts=100).fit(frame, pd.Series(y))
    assert np.isfinite(model.coef_).all()
    assert np.isnan(model.residuals_[:3]).all()
    # missing cells of new data are imputed before predicting
    X_new = X[:5].copy()
    X_new[0, 1] = np.nan
    predictions = model.predict(X_new)
    assert np.isfinite(predictions).all()
    assert not np.isnan(model.impute(X_new)).any()
    ax = model.cellmap(frame, y, rows=np.arange(20), annotate=True)
    assert ax is not None


def test_cell_lts_validates_input():
    X, y = _data()
    with pytest.raises(NotFittedError):
        CellLTSRegression().predict(X)
    with pytest.raises(ValueError, match="different number of rows"):
        CellLTSRegression().fit(X, y[:-1])
    with pytest.raises(ValueError, match="Unknown symmetrization"):
        CellLTSRegression(symmetrize="other").fit(X, y)
    model = CellLTSRegression(symmetrize="none", n_starts=50).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(X[:, :3])
    assert CellLTSRegression(n_pairs=5).get_params()["n_pairs"] == 5
