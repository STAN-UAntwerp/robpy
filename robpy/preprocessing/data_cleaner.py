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

    def fit(self, X: pd.DataFrame):
        """
        X (pd.DataFrame): input dataset.
        """

        self._get_non_numeric_columns_to_drop(X)

        self._get_discrete_columns_to_drop(X)

        self._get_bad_scale_columns_to_drop(X)

        return self

    def transform(self, X: pd.DataFrame):
        """
        X (pd.DataFrame): input dataset.
        """

        n, p = X.shape

        self._check_minimum_rows(n)
        self.logger.info(f"The input data has {n} rows and {p} columns.")
        X = X.replace([np.inf, -np.inf], np.nan)
        self._set_row_numbers_cols(X)
        self._set_missing_cols_and_rows(X)

        X = X.drop(
            columns=(
                self.non_numeric_cols
                + self.cols_discrete
                + self.cols_bad_scale
                + self.cols_rownumbers
                + self.cols_missings
            ),
        )

        X = X.drop(index=self.rows_missings)

        self.logger.info(f"The final data has {X.shape[0]} rows and {X.shape[1]} columns.")

        return X

    @property
    def dropped_columns(self) -> dict[str, list]:
        """Return the columns names that were dropped during the cleaning process.

        Returns:
            dict[str, list]: Mapping from reason for dropping to list of column names.

        Raises:
            NotFittedError: if the dropped column attributes weren't set yet.
        """
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
    def dropped_rows(self) -> dict[str, list]:
        """
        Return the rows indices that were dropped during the cleaning process.

        Returns:
            dict[str, list]: mapping from reason for dropping to list of row indices.

        Raises:
            NotFittedError: if the dropped row attributes weren't set yet.
        """
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

    def _get_non_numeric_columns_to_drop(self, X: pd.DataFrame):
        """Store non-numeric columns to drop."""
        self.non_numeric_cols = X.columns[
            X.apply(lambda s: pd.to_numeric(s.fillna(0), errors="coerce")).isna().any()
        ].tolist()

    def _set_row_numbers_cols(self, X: pd.DataFrame):
        """Check that no column consists of the row numbers."""
        self.cols_rownumbers = [col for col in X.columns if np.all(X[col] == np.arange(0, len(X)))]

    def _set_missing_cols_and_rows(self, X: pd.DataFrame):
        """Clean missing values."""
        if self.clean_na_first == "automatic":
            self.clean_na_first = "columns" if X.shape[1] >= 5 * X.shape[0] else "rows"

        X = X.replace([np.inf, -np.inf], np.nan).drop(
            columns=self.non_numeric_cols + self.cols_discrete + self.cols_bad_scale
        )
        if self.clean_na_first == "columns":
            self._set_missing_cols(X)
            self._set_missing_rows(X)
        elif self.clean_na_first == "rows":
            self._set_missing_rows(X)
            self._set_missing_cols(X)
        else:
            raise ValueError(
                'The argument clean_na_first should be "automatic", "rows" or "columns", '
                f'but received "{self.clean_na_first}".'
            )

    def _get_discrete_columns_to_drop(self, X: pd.DataFrame):
        """Store columns with a small number of unique values (discrete columns) to drop."""
        self.cols_discrete = X.columns[X.nunique() <= self.min_unique_values].tolist()

    def _get_bad_scale_columns_to_drop(self, X: pd.DataFrame):
        """Store columns with a scale smaller than min_abs_scale to drop."""
        X = X.drop(columns=self.non_numeric_cols + self.cols_discrete).astype(float)
        self.cols_bad_scale = X.columns[
            median_abs_deviation(
                X.replace([np.inf, -np.inf, True, False], np.nan),
                axis=0,
                nan_policy="omit",
            )
            <= self.min_abs_scale
        ].tolist()

    def _set_missing_cols(self, X: pd.DataFrame):
        """Remove columns with too many missings"""
        if hasattr(self, "rows_missings"):
            X = X.drop(index=self.rows_missings)
        self.cols_missings = X.columns[X.isna().mean() >= self.max_missing_frac_cols].tolist()

    def _set_missing_rows(self, X: pd.DataFrame):
        """Remove rows with too many missings"""
        if hasattr(self, "cols_missings"):
            X = X.drop(columns=self.cols_missings)
        self.rows_missings = X.index[X.isna().mean(axis=1) >= self.max_missing_frac_rows].tolist()
