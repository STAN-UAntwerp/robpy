"""Tests of DDC. The reference results are from cellWise 2.5.7, see
tests/reference/cellwise_reference.R."""

from pathlib import Path

import pandas as pd
import pytest
import numpy as np
from robpy.outliers import DDC
from robpy.univariate import HuberOneStepM, OneStepWrapping

REFERENCE = Path(__file__).parents[1] / "reference"


def _reference(name: str) -> np.ndarray:
    return pd.read_csv(REFERENCE / "expected" / f"{name}.csv").to_numpy()


def _input(name: str, folder: str = "data") -> pd.DataFrame:
    return pd.read_csv(REFERENCE / folder / f"{name}.csv")


def _check_against_reference(case: str, flags, residuals, imputed, rows):
    np.testing.assert_array_equal(flags, _reference(f"{case}_flags") == 1)
    np.testing.assert_allclose(residuals, _reference(f"{case}_residuals"), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(imputed, _reference(f"{case}_imputed"), rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(rows, _reference(f"{case}_rows")[:, 0] == 1)


REFERENCE_CASES = {
    # case: (input, folder, number of fitted rows, DDC arguments)
    "ddc_default": ("ddc_small", "data", 200, {}),
    "ddc_hubhub": (
        "ddc_small",
        "data",
        200,
        dict(chi2_quantile=0.999, scale_estimator=HuberOneStepM(), min_correlation=0.3),
    ),
    "ddc_wrap_cell09": (
        "ddc_small",
        "data",
        200,
        dict(cell_quantile=0.9, scale_estimator=OneStepWrapping()),
    ),
    "ddc_center0": (
        "ddc_center0_input",
        "expected",
        200,
        dict(cell_quantile=0.9, scale_estimator=OneStepWrapping(), center=np.zeros(8)),
    ),
    "ddc_neighbors": ("ddc_neighbors", "data", 80, dict(n_neighbors=4)),
}


@pytest.mark.parametrize("case", REFERENCE_CASES)
def test_ddc_fit_matches_cellwise(case):
    name, folder, n_fit, kwargs = REFERENCE_CASES[case]
    X = _input(name, folder).iloc[:n_fit]
    ddc = DDC(**kwargs).fit(X)
    _check_against_reference(
        case,
        ddc.cellwise_outliers_,
        ddc.standardized_residuals_,
        ddc.impute(X),
        ddc.row_outliers_,
    )
    # predict on the training data gives the same flags as fit
    np.testing.assert_array_equal(ddc.predict(X), ddc.cellwise_outliers_)


@pytest.mark.parametrize("case", ["ddc_default", "ddc_neighbors"])
def test_ddc_predict_matches_cellwise_ddcpredict(case):
    name, folder, n_fit, kwargs = REFERENCE_CASES[case]
    X = _input(name, folder)
    ddc = DDC(**kwargs).fit(X.iloc[:n_fit])
    X_new = X.iloc[n_fit:]
    _check_against_reference(
        f"{case}_new",
        ddc.predict(X_new),
        ddc.standardized_residuals(X_new),
        ddc.impute(X_new),
        ddc.predict(X_new, rowwise=True),
    )


def test_ddc_neighbors_are_limited_to_n_neighbors():
    X = _input("ddc_neighbors").iloc[:80]
    ddc = DDC(n_neighbors=4).fit(X)
    # the column itself and its 4 most correlated columns
    assert ddc.neighbors_.shape == (12, 5)
    np.testing.assert_array_equal(ddc.neighbors_[:, 0], np.arange(12))
    assert np.all(np.isnan(ddc.slopes_).sum(axis=0) >= 12 - 4)


def test_ddc_handles_infinite_and_missing_values():
    X = pd.DataFrame(
        {
            "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
            "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
            "V3": [2, np.inf, 3, -4, 5, 6, 7, -2, 8],
            "Vna": [1, -4, 2, np.nan, 3, -np.inf, np.nan, 6, 5],
        }
    )
    ddc = DDC().fit(X)
    imputed = ddc.impute(X)
    assert not imputed.isna().any().any()
    # infinite values are treated as missing: never flagged, but imputed
    assert not ddc.cellwise_outliers_[1, 2] and not ddc.cellwise_outliers_[5, 3]
    assert np.isfinite(imputed.iloc[1, 2]) and np.isfinite(imputed.iloc[5, 3])


def test_ddc_predict_returns_array_and_standardized_residuals_are_separate():
    # given
    X = pd.DataFrame(
        {
            "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
            "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
            "V3": [2, np.inf, 3, -4, 5, 6, 7, -2, 8],
            "Vna": [1, -4, 2, np.nan, 3, -np.inf, np.nan, 6, 5],
        }
    )
    ddc = DDC().fit(X)
    # when
    flags = ddc.predict(X)
    residuals = ddc.standardized_residuals(X)
    # then
    assert isinstance(flags, np.ndarray) and flags.dtype == bool
    assert flags.shape == X.shape == residuals.shape
    np.testing.assert_array_equal(flags, np.abs(residuals) > ddc.cutoff)
    np.testing.assert_allclose(residuals, ddc.standardized_residuals_)


def test_ddc_impute_does_not_round():
    # given
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 3)) * 1e-3, columns=["a", "b", "c"])
    X["b"] = X["a"] + 1e-4 * rng.normal(size=50)
    X_new = X.iloc[:5].copy()
    X_new.iloc[0, 1] = np.nan
    ddc = DDC().fit(X)
    # when
    imputed = ddc.impute(X_new, impute_outliers=False)
    # then
    assert imputed.iloc[0, 1] != 0
    assert abs(imputed.iloc[0, 1] - X_new.iloc[0, 0]) < 5e-4


def _data_with_binary_treatment(seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Column "a" depends on the binary column "d"; 5% of the cells of "a" get a 3 sd shift."""
    rng = np.random.default_rng(seed)
    n = 2000
    d = rng.binomial(1, 0.5, n).astype(float)
    a = 1.5 * (d - 0.5) + 0.6 * rng.normal(size=n)
    X = pd.DataFrame({"a": a, "b": a + rng.normal(size=n), "c": rng.normal(size=n), "d": d})
    bad = rng.random(n) < 0.05
    X.loc[bad, "a"] += 3 * X["a"].std() * rng.choice([-1, 1], bad.sum())
    return X, bad


def test_ddc_rejects_binary_column_unless_it_is_a_predictor():
    # given
    X, _ = _data_with_binary_treatment()
    # then
    with pytest.raises(ValueError, match=r"3 or fewer unique values.*\['d'\]"):
        DDC().fit(X)
    # when
    ddc = DDC(predictor_columns=["d"]).fit(X)
    flags = ddc.predict(X)
    # then
    d = X.columns.get_loc("d")
    assert flags.shape == X.shape
    assert not flags[:, d].any() and not ddc.cellwise_outliers_[:, d].any()
    assert np.isnan(ddc.standardized_residuals(X)[:, d]).all()
    assert np.isnan(ddc.standardized_residuals_[:, d]).all()
    pd.testing.assert_series_equal(ddc.impute(X)["d"], X["d"])
    assert np.array_equal(ddc.predictor_mask_, [False, False, False, True])
    # a single label works as well
    np.testing.assert_array_equal(
        DDC(predictor_columns="d").fit(X).cellwise_outliers_, ddc.cellwise_outliers_
    )


def test_ddc_predictor_column_improves_detection():
    # given
    X, bad = _data_with_binary_treatment()
    # when
    with_d = DDC(predictor_columns=["d"], predictor_min_correlation=0.3).fit(X)
    without_d = DDC().fit(X.drop(columns="d"))
    flags_with = with_d.cellwise_outliers_[:, 0]
    flags_without = without_d.cellwise_outliers_[:, 0]
    # then (recall 1.00 with and 0.84 without "d")
    assert flags_with[bad].mean() > flags_without[bad].mean()
    assert flags_with[~bad].mean() < 0.01
    # "d" is used in the prediction of "a" only, and not predicted itself
    assert not np.isnan(with_d.slopes_[3, 0])
    assert np.isnan(with_d.slopes_[:, 3]).all()
    assert np.isnan(with_d.deshrinkage_[3]) and np.isnan(with_d.residual_scales[3])


def test_ddc_predictor_min_correlation_defaults_to_min_correlation():
    # given: corr(a, d) is about 0.65 (robust estimate), between the two thresholds
    X, _ = _data_with_binary_treatment()
    # when
    default = DDC(predictor_columns=["d"], min_correlation=0.7).fit(X)
    lower = DDC(predictor_columns=["d"], min_correlation=0.7, predictor_min_correlation=0.3).fit(X)
    # then
    assert np.isnan(default.slopes_[3, 0])
    assert not np.isnan(lower.slopes_[3, 0])


def test_ddc_with_predictor_columns_predicts_new_rows():
    # given
    X, _ = _data_with_binary_treatment()
    train, test = X.iloc[:1500], X.iloc[1500:]
    # when
    ddc = DDC(predictor_columns=["d"], predictor_min_correlation=0.3).fit(train)
    flags = ddc.predict(test)
    imputed = ddc.impute(test)
    rows = ddc.predict(test, rowwise=True)
    # then
    assert flags.shape == test.shape and rows.shape == (len(test),)
    assert not flags[:, 3].any()
    pd.testing.assert_series_equal(imputed["d"], test["d"])
    assert flags[:, 0].any()
    # predictor columns must be present, at the same position and without missing values
    with pytest.raises(ValueError, match="same columns"):
        ddc.predict(test[["a", "b", "d", "c"]])
    with pytest.raises(ValueError, match=r"missing values: \['d'\]"):
        ddc.impute(test.assign(d=test["d"].where(test.index != 1600)))


@pytest.mark.parametrize(
    "change, predictor_columns, match",
    [
        (lambda X: X.assign(d=1.0), ["d"], r"constant: \['d'\]"),
        (lambda X: X.assign(d=X["d"].where(X.index != 5)), ["d"], r"missing values: \['d'\]"),
        (lambda X: X.assign(d=X["d"].where(X.index != 5, np.inf)), ["d"], "missing values"),
        (lambda X: X, ["d", "e"], r"Unknown predictor columns: \['e'\]"),
        (lambda X: X, ["a", "b", "c", "d"], "At least one column"),
    ],
)
def test_ddc_predictor_column_errors(change, predictor_columns, match):
    X, _ = _data_with_binary_treatment()
    with pytest.raises(ValueError, match=match):
        DDC(predictor_columns=predictor_columns).fit(change(X))


def test_ddc_min_unique_values():
    # given: a column with 4 distinct values
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
    X["d"] = np.tile([0.0, 1.0, 2.0, 3.0], 50)
    # then
    DDC().fit(X)
    with pytest.raises(ValueError, match=r"4 or fewer unique values.*\['d'\]"):
        DDC(min_unique_values=4).fit(X)


def test_ddc_min_correlation_sets_the_prediction_weights():
    # given: corr(a, b) about 0.4, corr(a, c) about 0.9, corr(a, e) about 0.6
    rng = np.random.default_rng(3)
    z = rng.normal(size=(1000, 4))
    a = z[:, 0]
    X = pd.DataFrame(
        {
            "a": a,
            "b": 0.4 * a + 0.92 * z[:, 1],
            "c": 0.9 * a + 0.44 * z[:, 2],
            "e": 0.6 * a + 0.8 * z[:, 3],
        }
    )
    # when
    default = DDC().fit(X)
    low = DDC(min_correlation=0.3).fit(X)
    high = DDC(min_correlation=0.7).fit(X)
    # then
    # "b" has no correlation above 0.5, so by default it is not predicted (standalone)
    assert np.isnan(default.deshrinkage_[1]) and not np.isnan(low.deshrinkage_[1])
    # with min_correlation 0.7, "a" is predicted from itself and "c" only
    weights = high._neighbor_weights[0]
    used = set(high.neighbors_[0, weights > 0])
    assert used == {0, 2}
    # weights below the minimum correlation are zero
    assert np.all(
        (high._neighbor_weights[:, 1:] == 0) | (np.abs(high.neighbor_correlations_[:, 1:]) >= 0.7)
    )
