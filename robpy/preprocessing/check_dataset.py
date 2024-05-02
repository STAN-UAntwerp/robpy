import numpy as np
import pandas as pd
import logging

from scipy.stats import median_abs_deviation
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)


class CheckDataset(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cleans a dataset before an analysis.

    Typically used before DDC, cellMCD, transfo...

    based on: [https://rdrr.io/cran/cellWise/man/checkDataSet.html]
    """

    def __init__(
        self,
        frac_na: float = 0.5,
        num_discrete: int = 3,
        prec_scale: float = 1e-12,
        clean_na_first: str = "automatic",
    ):
        """Initialize CheckDataset

        Args:
            frac_na (float, optional): Keep only the columns and rows that have a proportion of
                            missing values lower than this threshold.
                           Defaults to 0.5.
            num_discrete (int, optional): Any column with num_discrete or fewer distinct values will
                            be classified as discrete and excluded from the cleaned dataset.
                            Defaults to 3.
            prec_scale (float, optional): Only columns whose scale is larger than prec_scale will be
                            considered (scale is measure bu the mad).
                            Defaults to 1e-12.
            clean_na_first (str, optional): One out of "automatic", "columns", "rows". Decides which
                            are first checked for NAs. If "automatic", columns are checked first if
                            if p >= 5n, else rows are checked first.
                            Defaults to "automatic".
        """

        self.frac_na = frac_na
        self.num_discrete = num_discrete
        self.prec_scale = prec_scale
        self.clean_na_first = clean_na_first
        self.logger = logging.getLogger("checkDataset")

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset.

        X (pd.DataFrame): input dataset.
        """

        X = df.copy()
        n, p = X.shape

        self._check_minimum_rows(n)
        self.logger.info(f"The input data has {n} rows and {p} columns.")

        X = self._retain_numeric_columns(X)

        X = self._check_no_row_numbers(X, n)

        X = self._clean_missing_values(X)

        X = self._remove_discrete_columns(X)

        X = self._remove_columns_with_bad_scale(X)

        self.logger.info(f"The final data has {X.shape[0]} rows and {X.shape[1]} columns.")

        return self

    def _check_minimum_rows(self, n: int) -> None:
        """Check if there are enough rows in the dataset."""
        if n < 3:
            raise ValueError(f"The input data must have at least 3 rows, but received only {n}.")

    def _retain_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retain only numeric columns."""
        self.non_numeric_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
        if self.non_numeric_cols:
            self.logger.info(
                f"\nThe input data contains {len(self.non_numeric_cols)} non-numeric column(s). "
                f"Their column names are:\n\t{', '.join(self.non_numeric_cols)}."
                f"\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=self.non_numeric_cols, inplace=True)
            self._check_enough_cols(X)
        return X

    def _check_no_row_numbers(self, X: pd.DataFrame, n: int) -> pd.DataFrame:
        """Check that no column consists of the row numbers."""
        self.cols_rownumbers = [col for col in X.columns if np.all(X[col] == np.arange(0, n))]
        if self.cols_rownumbers:
            self.logger.info(
                f"\nThe input data contains {len(self.cols_rownumbers)} column(s) that is identical to "
                f"the row numbers. Their column names are:\n\t{', '.join(self.cols_rownumbers)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=self.cols_rownumbers, inplace=True)
            self._check_enough_cols(X)
        return X

    def _clean_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values."""
        if self.clean_na_first == "automatic":
            self.clean_na_first = "columns" if X.shape[1] >= 5 * X.shape[0] else "rows"

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self.clean_na_first == "columns":
            X = self._clean_cols(X)
            X = self._clean_rows(X)
        elif self.clean_na_first == "rows":
            X = self._clean_rows(X)
            X = self._clean_cols(X)
        else:
            raise ValueError(
                'The argument clean_na_first should be "automatic", "rows" or "columns", '
                f'but received "{self.clean_na_first}".'
            )
        return X

    def _remove_discrete_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with a small number of unique values."""
        self.cols_discrete = X.columns[X.nunique() <= self.num_discrete].tolist()
        if self.cols_discrete:
            self.logger.info(
                f"\nThe data contains {len(self.cols_discrete)} discrete column(s) with "
                f"{self.num_discrete} or fewer unique values. "
                f"Their column names are:\n\t{', '.join(self.cols_discrete)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=self.cols_discrete, inplace=True)
            self._check_enough_cols(X)
        return X

    def _remove_columns_with_bad_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with scale smaller than prec_scale."""
        self.cols_bad_scale = X.columns[
            median_abs_deviation(X, axis=0, nan_policy="omit") <= self.prec_scale
        ].tolist()
        if self.cols_bad_scale:
            self.logger.info(
                f"\nThe data contains {len(self.cols_bad_scale)} column(s) with an (almost) zero "
                "median absolute deviation. "
                f"Their column names are:\n\t{', '.join(self.cols_bad_scale)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=self.cols_bad_scale, inplace=True)
            self._check_enough_cols(X)
        return X

    def _clean_cols(self, X: pd.DataFrame):
        accept_na = X.shape[0] * self.frac_na
        na_counts = X.isna().sum()
        self.na_col = X.columns[na_counts > accept_na].tolist()
        if self.na_col:
            self.logger.info(
                f"\nThe data contains {len(self.na_col)} column(s) with over {100*self.frac_na}% NAs. "
                f"Their column names are: \n\t{', '.join(self.na_col)}."
                "\nThese columns will be ignored in the analysis."
            )
            X.drop(columns=self.na_col, inplace=True)
            self._check_enough_cols(X)
        return X

    def _clean_rows(self, X: pd.DataFrame):
        accept_na = X.shape[1] * self.frac_na
        na_counts = X.isna().sum(axis=1)
        self.na_row = na_counts[na_counts > accept_na].index.tolist()
        if self.na_row:
            self.logger.info(
                f"\nThe data contains {len(self.na_row)} row(s) with over {100*self.frac_na}% NAs. "
                f"Their row names/indices are: \n\t{', '.join(map(str, self.na_row))}."
                "\nThese rows will be ignored in the analysis."
            )
            X.drop(self.na_row, inplace=True)
            self._check_enough_rows(X)
        return X

    def _check_enough_cols(self, X: pd.DataFrame):
        if X.shape[1] > 1:
            self.logger.info(
                f"We continue with the remaining {X.shape[1]} columns: "
                f"\n\t{', '.join(X.columns)}."
            )
        elif X.shape[1] == 1:
            raise ValueError("Only 1 column remains, this is not enough.")
        else:
            raise ValueError("No columns remain.")

    def _check_enough_rows(self, X: pd.DataFrame):
        if X.shape[0] > 2:
            self.logger.info(
                f"We continue with the remaining {X.shape[0]} rows: "
                f"\n\t{', '.join(map(str, X.index))}."
            )
        elif X.shape[0] in [1, 2]:
            raise ValueError(f"Only {X.shape[0]} row(s) remain(s), this is not enough.")
        else:
            raise ValueError("No rows remain.")

    def transform(self, X: pd.DataFrame):

        X.drop(
            np.concatenate(
                (
                    self.non_numeric_cols,
                    self.cols_rownumbers,
                    self.na_col,
                    self.cols_discrete,
                    self.cols_bad_scale,
                )
            ),
            axis=1,
            inplace=True,
        )
        X.drop(self.na_row, axis=0, inplace=True)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        return X
