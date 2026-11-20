import pandas as pd
import pytest
import numpy as np
from robpy.outliers import DDC


def test_ddc_correctly_flags_dummy_data():
    # given
    X = pd.DataFrame(
        {
            "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
            "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
            "V3": [2, np.inf, 3, -4, 5, 6, 7, -2, 8],
            "Vna": [1, -4, 2, np.nan, 3, -np.inf, np.nan, 6, 5],
        }
    )

    expected_outliers = [(4, 0), (7, 1), (1, 3)]
    expected_rowwise_outliers = np.array(
        [False, True, False, False, False, False, False, False, False]
    )
    ddc = DDC()
    # when
    ddc = ddc.fit(X)
    cellwise_prediction = ddc.predict(X)
    rowwise_prediction = ddc.predict(X, rowwise=True)

    predicted_cellwise_idx = list(zip(*np.where(cellwise_prediction)))
    # then
    assert set(predicted_cellwise_idx) == set(expected_outliers)
    assert np.array_equal(rowwise_prediction, expected_rowwise_outliers)


def test_ddc_can_predict_new_data():
    # given
    X = pd.DataFrame(
        {
            "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
            "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
            "V3": [2, np.inf, 3, -4, 5, 6, 7, -2, 8],
            "Vna": [1, -4, 2, np.nan, 3, -np.inf, np.nan, 6, 5],
        }
    )

    X_new = pd.DataFrame(
        {
            "V1": [1.2, 10],
            "V2": [3, -3],
            "V3": [-30, -5],
            "Vna": [np.nan, 5],
        }
    )

    expected_outliers = [(0, 2), (1, 0), (1, 1)]
    expected_rowwise_outliers = np.array([False, True])
    ddc = DDC()
    # when
    ddc = ddc.fit(X)
    cellwise_prediction = ddc.predict(X_new)
    rowwise_prediction = ddc.predict(X_new, rowwise=True)

    predicted_cellwise_idx = list(zip(*np.where(cellwise_prediction)))
    # then
    assert set(predicted_cellwise_idx) == set(expected_outliers)
    assert np.array_equal(rowwise_prediction, expected_rowwise_outliers)


def test_ddc_can_impute_missing_data():
    # given
    X = pd.DataFrame(
        {
            "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
            "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
            "V3": [2, np.inf, 3, -4, 5, 6, 7, -2, 8],
            "Vna": [1, -4, 2, np.nan, 3, -np.inf, np.nan, 6, 5],
        }
    )

    X_new = pd.DataFrame(
        {
            "V1": [1.2, 10.0],
            "V2": [3.0, -3.0],
            "V3": [-30.0, -5.0],
            "Vna": [np.nan, 5.0],
        }
    )

    expected_output = X_new.copy()
    expected_output.loc[0, "Vna"] = 3.039
    ddc = DDC()
    # when
    ddc = ddc.fit(X)
    imputed = ddc.impute(X_new, impute_outliers=False)

    # then
    # expected values are rounded to 3 decimals; impute itself does not round
    pd.testing.assert_frame_equal(imputed, expected_output, check_exact=False, rtol=0, atol=5e-4)


def test_ddc_can_impute_outlying_data():
    # given
    X = pd.DataFrame(
        {
            "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
            "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
            "V3": [2, np.inf, 3, -4, 5, 6, 7, -2, 8],
            "Vna": [1, -4, 2, np.nan, 3, -np.inf, np.nan, 6, 5],
        }
    )

    X_new = pd.DataFrame(
        {
            "V1": [1.2, 10.0],
            "V2": [3.0, -3.0],
            "V3": [-30.0, -5.0],
            "Vna": [np.nan, 5.0],
        }
    )

    expected_output = X_new.copy()
    expected_output.loc[0, "Vna"] = 3.039
    expected_output.loc[1, "V1"] = -4.913
    expected_output.loc[1, "V2"] = 9.477
    expected_output.loc[0, "V3"] = 4.119
    ddc = DDC()
    # when
    ddc = ddc.fit(X)
    imputed = ddc.impute(X_new, impute_outliers=True)

    # then
    # expected values are rounded to 3 decimals; impute itself does not round
    pd.testing.assert_frame_equal(imputed, expected_output, check_exact=False, rtol=0, atol=5e-4)


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


def _correlated_data_with_missing_values() -> pd.DataFrame:
    # correlation 0.6 between all columns, from a common factor: no linear algebra, so the
    # data (and the fixture below) do not depend on the platform's LAPACK
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        np.sqrt(0.6) * rng.normal(size=(200, 1)) + np.sqrt(0.4) * rng.normal(size=(200, 5)),
        columns=list("abcde"),
    )
    X = X.mask(rng.random(X.shape) < 0.05)
    X.iloc[:10, 2] += 6
    return X


def test_ddc_without_predictor_columns_reproduces_previous_release():
    # given
    X = _correlated_data_with_missing_values()
    # when
    ddc = DDC().fit(X)
    ddc_none = DDC(predictor_columns=None).fit(X)
    # then: identical to each other and to robpy 0.1.1 (no residual lies within 0.09 of the
    # cutoff, so the flags are not sensitive to rounding)
    np.testing.assert_array_equal(ddc.cellwise_outliers_, ddc_none.cellwise_outliers_)
    np.testing.assert_array_equal(ddc.standardized_residuals_, ddc_none.standardized_residuals_)
    pd.testing.assert_frame_equal(ddc.impute(X), ddc_none.impute(X))
    assert list(np.flatnonzero(ddc.cellwise_outliers_)) == [
        2,
        7,
        12,
        17,
        22,
        27,
        32,
        37,
        42,
        47,
        84,
        151,
        276,
        278,
        851,
    ]
    np.testing.assert_allclose(
        np.nansum(np.abs(ddc.standardized_residuals_), axis=0),
        [136.82692193, 134.42514583, 194.16102295, 140.7798212, 144.98588884],
    )
    np.testing.assert_allclose(
        ddc.impute(X).to_numpy().sum(axis=0),
        [-12.18681419, -14.39220356, -3.0219623, -0.3492783, 20.87115259],
    )
    assert list(np.flatnonzero(ddc.row_outliers_)) == [55]


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
    # then (recall 1.00 with and 0.83 without "d")
    assert flags_with[bad].mean() > flags_without[bad].mean()
    assert flags_with[~bad].mean() < 0.01
    # "d" is used in the prediction of "a" only, and not predicted itself
    assert not np.isnan(with_d.slopes_[3, 0])
    assert np.isnan(with_d.slopes_[:, 3]).all()
    assert np.all(with_d.raw_predictions_[:, 3] == 0)


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


def test_ddc_min_correlation_is_used_in_the_prediction_weights():
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
    too_high = DDC(min_correlation=0.95).fit(X)
    # then
    # "b" has no correlation above 0.5, but it is predicted with min_correlation 0.3
    assert np.all(default.raw_predictions_[:, 1] == 0)
    assert not np.all(low.raw_predictions_[:, 1] == 0)
    # with min_correlation 0.7, "a" is predicted from "c" alone, with weight 1 ("e" is not used)
    U = high._univariate_outliers(high._standardize(X))
    rows = U["c"].notna().to_numpy()
    np.testing.assert_allclose(
        high.raw_predictions_[rows, 0], high.slopes_[2, 0] * U.loc[rows, "c"].to_numpy()
    )
    # with a threshold above every correlation, nothing is predicted
    assert np.all(too_high.raw_predictions_ == 0)
