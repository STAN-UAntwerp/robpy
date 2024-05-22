import pandas as pd
import numpy as np
from robpy.outliers import DDCEstimator


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
    expected_outliers = np.full(X.shape, False)
    expected_outliers[4, 0] = True
    expected_outliers[7, 1] = True
    ddc = DDCEstimator()
    # when
    ddc = ddc.fit(X)
    # then
    assert np.array_equal(ddc.cellwise_outliers_, expected_outliers)
