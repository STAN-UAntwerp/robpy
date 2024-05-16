import numpy as np
import pandas as pd
import logging

from scipy.stats import median_abs_deviation
from sklearn.base import (
    BaseEstimator,
    OneToOneFeatureMixin,
    TransformerMixin,
)
from sklearn.exceptions import NotFittedError


class DataCleaner(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Cleans a dataset before an analysis.

    Typically used before DDC, cellMCD, transfo...

    based on: [https://rdrr.io/cran/cellWise/man/checkDataSet.html]
    """

    def __init__(
        self,
        max_missing_frac_cols: float = 0.5,
        max_missing_frac_rows: float = 0.5,
        min_unique_values: int = 3,
        min_abs_scale: float = 1e-12,
        clean_na_first: str = "automatic",
        min_n_rows: int = 3,
    ):
        """Initialize DataCleaner

        Args:
            max_missing_frac_cols (float, optional): Keep only the columns that have a
                            proportion of missing values lower than this threshold.
                            Defaults to 0.5.
            max_missing_frac_rows (float, optional): Keep only the rows that have a
                            proportion of missing values lower than this threshold.
                            Defaults to 0.5.
            min_unique_values (int, optional): Any column with min_unique_values or fewer unique
                            values will be classified as discrete and excluded from the cleaned
                            dataset.
                            Defaults to 3.
            min_abs_scale (float, optional): Only columns whose scale is larger than min_abs_scale
                            will be considered (scale is measure by the mad).
                            Defaults to 1e-12.
            clean_na_first (str, optional): One out of "automatic", "columns", "rows". Decides which
                            are first checked for NAs. If "automatic", columns are checked first if
                            if p >= 5n, else rows are checked first.
                            Defaults to "automatic".
            min_n_rows (int, optional): Integer specifying the minimum number of rows/observations
                            wanted for the input data.
                            Defaults to 3.
        """

        self.max_missing_frac_cols = max_missing_frac_cols
        self.max_missing_frac_rows = max_missing_frac_rows
        self.min_unique_values = min_unique_values
        self.min_abs_scale = min_abs_scale
        self.clean_na_first = clean_na_first
        self.min_n_rows = min_n_rows
        self.logger = logging.getLogger("DataCleaner")

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        X (pd.DataFrame): input dataset.
        """

        X = df.copy()

        self._retain_numeric_columns(X)

        self._remove_discrete_columns(X)

        self._remove_columns_with_bad_scale(X)

        return self

    def transform(self, X: pd.DataFrame):
        """
        X (pd.DataFrame): input dataset.
        """

        n, p = X.shape

        self._check_minimum_rows(n)
        self.logger.info(f"The input data has {n} rows and {p} columns.")

        X.drop(
            np.concatenate(
                (
                    self.non_numeric_cols,
                    self.cols_discrete,
                    self.cols_bad_scale,
                )
            ),
            axis=1,
            inplace=True,
        )

        self._check_no_row_numbers(X, n)

        self._clean_missing_values(X)

        X.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.logger.info(f"The final data has {X.shape[0]} rows and {X.shape[1]} columns.")

        return X

    @property
    def dropped_columns(self):
        if not all(
            hasattr(self, attr)
            for attr in [
                "non_numeric_cols",
                "cols_rownumbers",
                "cols_discrete",
                "cols_bad_scale",
                "cols_missings",
            ]
        ):
            raise NotFittedError()
        return {
            "non_numeric_cols": self.non_numeric_cols,
            "cols_rownumbers": self.cols_rownumbers,
            "cols_discrete": self.cols_discrete,
            "cols_bad_scale": self.cols_bad_scale,
            "cols_missings": self.cols_missings,
        }

    @property
    def dropped_rows(self):
        if not hasattr(self, "rows_missings"):
            raise NotFittedError()
        return {"rows_missings": self.rows_missings}

    def _check_minimum_rows(self, n: int) -> None:
        """Check if there are enough rows in the input dataset."""
        if n < self.min_n_rows:
            raise ValueError(
                f"The input data must have at least {self.min_n_rows} rows, "
                f"but received only {n}."
            )

    def _retain_numeric_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retain only numeric columns."""
        self.non_numeric_cols = X.columns[
            X.apply(pd.to_numeric, errors="coerce").isna().all()
        ].tolist()
        if self.non_numeric_cols:
            X.drop(columns=self.non_numeric_cols, inplace=True)
        return self

    def _check_no_row_numbers(self, X: pd.DataFrame, n: int) -> pd.DataFrame:
        """Check that no column consists of the row numbers."""
        self.cols_rownumbers = [col for col in X.columns if np.all(X[col] == np.arange(0, n))]
        if self.cols_rownumbers:
            X.drop(columns=self.cols_rownumbers, inplace=True)
        return self

    def _clean_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values."""
        if self.clean_na_first == "automatic":
            self.clean_na_first = "columns" if X.shape[1] >= 5 * X.shape[0] else "rows"

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self.clean_na_first == "columns":
            self._clean_cols(X)
            self._clean_rows(X)
        elif self.clean_na_first == "rows":
            self._clean_rows(X)
            self._clean_cols(X)
        else:
            raise ValueError(
                'The argument clean_na_first should be "automatic", "rows" or "columns", '
                f'but received "{self.clean_na_first}".'
            )
        return self

    def _remove_discrete_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with a small number of unique values."""
        self.cols_discrete = X.columns[X.nunique() <= self.min_unique_values].tolist()
        if self.cols_discrete:
            X.drop(columns=self.cols_discrete, inplace=True)
        return self

    def _remove_columns_with_bad_scale(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with scale smaller than min_abs_scale."""
        self.cols_bad_scale = X.columns[
            median_abs_deviation(
                X.replace([np.inf, -np.inf, True, False], np.nan), axis=0, nan_policy="omit"
            )
            <= self.min_abs_scale
        ].tolist()
        if self.cols_bad_scale:
            X.drop(columns=self.cols_bad_scale, inplace=True)
        return self

    def _clean_cols(self, X: pd.DataFrame):
        """Remove columns with too many missings"""
        self.cols_missings = X.columns[X.isna().mean() >= self.max_missing_frac_cols].tolist()
        if self.cols_missings:
            X.drop(columns=self.cols_missings, inplace=True)
        return self

    def _clean_rows(self, X: pd.DataFrame):
        """Remove rows with too many missings"""
        self.rows_missings = X.index[X.isna().mean(axis=1) >= self.max_missing_frac_rows].tolist()
        if self.rows_missings:
            X.drop(index=self.rows_missings, inplace=True)
        return self
