import pytest
import pandas as pd
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
    Z = high._standardize(X)
    U = Z.where(np.abs(Z) <= high.cutoff)
    rows = U["c"].notna().to_numpy()
    np.testing.assert_allclose(
        high.raw_predictions_[rows, 0], high.slopes_[2, 0] * U.loc[rows, "c"].to_numpy()
    )
    # with a threshold above every correlation, nothing is predicted
    assert np.all(too_high.raw_predictions_ == 0)
