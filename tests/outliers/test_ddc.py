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
    cellwise_prediction, _ = ddc.predict(X)
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
    cellwise_prediction, _ = ddc.predict(X_new)
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
    pd.testing.assert_frame_equal(imputed, expected_output)


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
    pd.testing.assert_frame_equal(imputed, expected_output)
