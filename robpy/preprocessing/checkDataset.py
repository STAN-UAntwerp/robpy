import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)


class CheckDataset(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cleans a dataset before an analysis.

    Typically used before DDC, cellMCD, transfo...

    [https://rdrr.io/cran/cellWise/man/checkDataSet.html]
    """

    def __init__(
        self,
        fracNA: float = 0.5,
        numDiscrete: int = 3,
        precScale: float = 1e-12,
        cleanNAfirst: str = "automatic",
    ):
        """Initialize CheckDataset

        Args:
            fracNA (float, optional): Keep only the columns and rows that have a proportion of
                            missing values lower than this threshold.
                           Defaults to 0.5.
            numDiscrete (int, optional): Any column with numDiscrete or fewer distinct values will
                            be classified as discrete and excluded from the cleaned dataset.
                            Defaults to 3.
            precScale (float, optional): Only columns whose scale is larger than precScale will be
                            considered (scale is measure bu the mad).
                            Defaults to 1e-12.
            cleanNAfirst (str, optional): One out of "automatic", "columns", "rows". Decides which
                            are first checked for NAs. If "automatic", columns are checked first if
                            if p >= 5n, else rows are checked first.
                            Defaults to "automatic".
        """

        self.fracNA = fracNA
        self.numDiscrete = numDiscrete
        self.precScale = precScale
        self.cleanNAfirst = cleanNAfirst

    def fit(
        self,
        df: pd.DataFrame,
    ):
        """
        X (np.ndarray or pd.DataFrame): input dataset.
        """

        X = df.copy()
        n, p = X.shape

        # 1)    Check that there are at least 3 rows:
        if n < 3:
            raise ValueError(
                f"The input data must have atleast 3 rows/observations, but received only {n}."
            )
        print(f"The input data has {n} rows and {p} columns.")

        # 2)    Only retain the numeric columns (only needed if dataframe):
        non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
        if len(non_numeric_cols) > 0:
            print(
                f"\nThe input data contains {len(non_numeric_cols)} non-numeric column(s). "
                f"Their column names are:\n\t{', '.join(non_numeric_cols)}."
                f"\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=non_numeric_cols, inplace=True)
            self._check_enough_cols(X)

        # 3)    Check that no column consists of the rownumbers:
        cols_rownumbers = []
        for col in X.columns:
            if np.all(X[col] == np.arange(0, n)):
                cols_rownumbers.append(col)
        if len(cols_rownumbers) > 0:
            print(
                f"\nThe input data contains {len(cols_rownumbers)} column(s) that is identical to "
                f"the row numbers. Their column names are:\n\t{', '.join(cols_rownumbers)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=cols_rownumbers, inplace=True)
            self._check_enough_cols(X)

        # 4)    Clean NAs:
        if self.cleanNAfirst == "automatic":
            if X.shape[1] >= 5 * X.shape[0]:
                self.cleanNAfirst = "columns"
            else:
                self.cleanNAfirst = "rows"

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self.cleanNAfirst == "columns":
            X = self._clean_cols(X)
            X = self._clean_rows(X)
        elif self.cleanNAfirst == "rows":
            X = self._clean_rows(X)
            X = self._clean_cols(X)
        else:
            raise ValueError(
                'The argument cleanNAfirst should be "automatic", "rows" or "columns", '
                f'but received "{self.cleanNAfirst}".'
            )

        # 5)    Remove discrete columns:
        cols_discrete = X.columns[X.nunique() <= self.numDiscrete].tolist()
        if len(cols_discrete) > 0:
            print(
                f"\nThe data contains {len(cols_discrete)} discrete column(s) with "
                f"{self.numDiscrete} or fewer unique values. "
                f"Their column names are:\n\t{', '.join(cols_discrete)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=cols_discrete, inplace=True)
            self._check_enough_cols(X)

        # 6)    Remove columns with scale smaller than precScale
        cols_bad_scale = X.columns[
            median_abs_deviation(X, axis=0, nan_policy="omit") <= self.precScale
        ].tolist()
        if len(cols_bad_scale) > 0:
            print(
                f"\nThe data contains {len(cols_bad_scale)} column(s) with an (almost) zero "
                "median absolute deviation. "
                f"Their column names are:\n\t{', '.join(cols_bad_scale)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=cols_bad_scale, inplace=True)
            self._check_enough_cols(X)

        # 7)    Conclusion:
        if X.shape[0] < n or X.shape[1] < p:
            print(f"\nThe final data has {X.shape[0]} rows and {X.shape[1]} columns.")

        return X

    def _clean_cols(self, X: pd.DataFrame):
        acceptNA = X.shape[0] * self.fracNA
        NAcounts = X.isna().sum()
        NAcol = X.columns[NAcounts > acceptNA].tolist()
        if len(NAcol) > 0:
            print(
                f"\nThe data contains {len(NAcol)} column(s) with over {100*self.fracNA}% NAs. "
                f"Their column names are: \n\t{', '.join(NAcol)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=NAcol, inplace=True)
            self._check_enough_cols(X)
        return X

    def _clean_rows(self, X: pd.DataFrame):
        acceptNA = X.shape[1] * self.fracNA
        NAcounts = X.isna().sum(axis=1)
        NArow = NAcounts[NAcounts > acceptNA].index.tolist()
        if len(NArow) > 0:
            print(
                f"\nThe data contains {len(NArow)} row(s) with over {100*self.fracNA}% NAs. "
                f"Their row names/indices are: \n\t{', '.join(map(str, NArow))}."
                "\nThese rows will be ignored in the analysis."
            )
            X.drop(NArow, inplace=True)
            self._check_enough_rows(X)
        return X

    def _check_enough_cols(self, X: pd.DataFrame):
        if X.shape[1] > 1:
            print(
                f"We continue with the remaining {X.shape[1]} columns: "
                f"\n\t{', '.join(X.columns)}."
            )
        elif X.shape[1] == 1:
            raise ValueError("Only 1 column remains, this is not enough.")
        else:
            raise ValueError("No columns remain.")

    def _check_enough_rows(self, X: pd.DataFrame):
        if X.shape[0] > 2:
            print(
                f"We continue with the remaining {X.shape[0]} rows: "
                f"\n\t{', '.join(map(str, X.index))}."
            )
        elif X.shape[0] == 2:
            raise ValueError("Only 2 rows remain, this is not enough.")
        elif X.shape[0] == 1:
            raise ValueError("Only 1 row remains, this is not enough.")
        else:
            raise ValueError("No rows remain.")

    def transform(self, X: np.ndarray | pd.DataFrame):
        raise NotImplementedError

    # TODO: I don't know what to put in fit and what to put in transform.
