import numpy as np
import pandas as pd
from robpy.preprocessing.data_cleaner import DataCleaner


def test_check_dataset():
    """
    [https://rdrr.io/cran/cellWise/man/checkDataSet.html]
    """
    # given data
    d = {
        "i": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "name": ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii"],
        "logic": [True, False, True, False, False, True, True, True, False],
        "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
        "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
        "V3": [2, np.inf, 3, -4, 5, 6, 7, -2, 8],
        "Vna": [1, -4, 2, np.nan, np.nan, -np.inf, np.nan, 6, np.nan],
        "Vdis": [1, 1, 2, 2, 3, 3, 3, 1, 2],
        "V0s": [1, 1.5, 2, 2, 2, 2, 2, 3, 2.5],
    }
    df = pd.DataFrame(data=d)
    # clean data
    d_clean = {
        "V1": [1.3, np.nan, 4.5, 2.7, 20.0, 4.4, -2.1, 1.1, -5],
        "V2": [2.3, np.nan, 5, 6, 7, 8, 4, -10, 0.5],
        "V3": [2, np.nan, 3, -4, 5, 6, 7, -2, 8],
    }
    df_clean = pd.DataFrame(data=d_clean)
    # then
    result1 = DataCleaner().fit_transform(df)
    np.testing.assert_array_almost_equal(result1, df_clean)
