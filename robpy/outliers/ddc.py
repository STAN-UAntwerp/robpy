from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats as stats

from matplotlib.axes import Axes
from scipy.special import erf
from sklearn.base import OutlierMixin
from scipy.stats import chi2

from robpy.univariate import CellwiseOneStepM
from robpy.univariate._cellwise import rho_huber25, scale_1step_m
from robpy.univariate.base import RobustScale


def get_custom_cmap(vmax_clip: float, neutral_color: str = "#f7f286") -> matplotlib.colors.Colormap:
    norm = matplotlib.colors.Normalize(-vmax_clip, vmax_clip)
    colors = [
        [norm(-vmax_clip), "#4652a3"],
        [norm(-(vmax_clip + 2.5) / 2), "#8d6d8c"],
        [norm(-2.5), neutral_color],
        [norm(2.5), neutral_color],
        [norm(vmax_clip), "#ec2123"],
    ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


def _huber_scale(x: np.ndarray, prec_scale: float = 1e-12) -> float:
    """``scale1StepM(x, rhoHuber25)`` of cellWise: Huber scale around zero (nan ignored)."""
    return scale_1step_m(x, rho_huber25, prec_scale=prec_scale)


def _corr_gkwls(x: np.ndarray, y: np.ndarray, q_corr: float, prec_scale: float) -> float:
    """Robust correlation of two standardized columns (``corrGKWLS`` in cellWise): the
    Gnanadesikan-Kettenring estimate from the Huber scales of x + y and x - y, followed by an
    uncentered Pearson correlation on the pairs inside the corresponding tolerance ellipse."""
    s = x + y
    if np.count_nonzero(np.isfinite(s)) <= 3:
        return 0.0
    corr = (_huber_scale(s, prec_scale) ** 2 - _huber_scale(x - y, prec_scale) ** 2) / 4
    if not np.isfinite(corr):
        return 0.0
    corr = min(max(corr, -0.99), 0.99)
    with np.errstate(invalid="ignore"):
        rd2 = (x**2 - 2 * corr * x * y + y**2) / abs(1 - corr**2)
        keep = rd2 < q_corr
    if not keep.any():
        return 0.0
    xs, ys = x[keep], y[keep]
    return float(np.dot(xs, ys) / (np.linalg.norm(xs) * np.linalg.norm(ys)))


def _slope_med_wls(x: np.ndarray, y: np.ndarray, q_regr: float, prec_scale: float) -> float:
    """Slope of a robust regression of y on x without intercept (``slopeMedWLS`` in cellWise):
    the median of y / x (clipped to [-2, 2]), followed by least squares on the points whose
    residual is within q_regr Huber scales."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = y / x
    finite = np.isfinite(ratio)
    if np.count_nonzero(finite) <= 3:
        return 0.0
    raw = float(np.median(ratio[finite]))
    if not np.isfinite(raw):
        return 0.0
    raw = min(max(raw, -2.0), 2.0)
    r = y - raw * x
    cutoff = q_regr * _huber_scale(r, prec_scale)
    with np.errstate(invalid="ignore"):
        keep = np.abs(r) <= cutoff
    if not keep.any():
        return 0.0
    xw, yw = x[keep], y[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.dot(xw, yw) / np.dot(xw, xw)
    return float(slope) if np.isfinite(slope) else 0.0


class DDC(OutlierMixin):
    def __init__(
        self,
        chi2_quantile: float = 0.99,
        min_correlation: float = 0.5,
        scale_estimator: RobustScale = CellwiseOneStepM(),
        min_unique_values: int = 3,
        predictor_columns: list | None = None,
        predictor_min_correlation: float | None = None,
        center: np.ndarray | None = None,
        n_neighbors: int = 100,
        cell_quantile: float | None = None,
    ):
        """
        Implementation of the Detecting Deviating Cells (DDC) algorithm, following the R
        implementation ``DDC`` and ``DDCpredict`` in the package cellWise (version 2.5.7) with
        its default settings (``combinRule = "wmean"``, ``corrType = "gkwls"``,
        ``includeSelf = TRUE``, ``fastDDC = FALSE``). The data checks of
        ``cellWise::checkDataSet`` are not applied: discrete columns raise an error instead of
        being removed.

        Args:
            chi2_quantile (float, optional): Probability of the chi-squared distribution with
              one degree of freedom that sets the cutoff for flagging cells and rows and for the
              robust slopes (``tolProb`` in cellWise). The robust correlations use the same
              probability with two degrees of freedom. Default is 0.99.
            min_correlation (float, optional): Minimum absolute correlation between two
              variables for one to be used in the prediction of the other (``corrlim``).
              Default is 0.5.
            scale_estimator (RobustScale, optional) : robust location and scale estimator used
              to standardize the columns. Defaults to CellwiseOneStepM(), which is
              ``standType = "1stepM"`` in cellWise; OneStepWrapping() is ``"wrap"`` and
              HuberOneStepM() is ``"hubhub"``.
            min_unique_values (int, optional): Columns with this many or fewer distinct values
              are considered discrete and are not supported (as ``numDiscrete`` in cellWise),
              except as predictor columns. Defaults to 3.
            predictor_columns (list | None, optional): Labels of columns of X that are known to
              be clean and only serve to predict the other (target) columns, e.g. a binary
              treatment indicator. These columns are never flagged or imputed and may be
              discrete, but must be numeric, non-constant and without missing values. They are
              standardized with their mean and standard deviation instead of
              ``scale_estimator``, are not subject to the univariate outlier detection of step
              2, and their correlation with a target column is the Pearson correlation over the
              cells of the target column that survive step 2. Their standardized residuals are
              nan, and they are blank in ``cellmap``. This is an extension of DDC, not part of
              the published algorithm or of cellWise. Defaults to None (no predictor columns),
              which gives the original DDC.
            predictor_min_correlation (float | None, optional): Minimum absolute correlation
              between a predictor column and a target column for the predictor column to be
              used in the prediction of the target column. A discrete predictor is typically
              less correlated with a continuous variable than continuous variables are among
              each other, so a lower value than ``min_correlation`` can be appropriate.
              Defaults to None, which uses ``min_correlation``.
            center (np.ndarray | None, optional): Fixed center of the columns (``center`` in
              cellWise). The scales are then computed around this center, with the location
              of ``scale_estimator`` fixed at 0 (its ``center=False`` option). Defaults to None
              (the location is estimated).
            n_neighbors (int, optional): Maximum number of other variables used to predict a
              variable, those with the highest absolute robust correlations (``nbngbrs``).
              Defaults to 100.
            cell_quantile (float | None, optional): Probability that sets the cutoff for the
              cells only, i.e. for the univariate detection of step 2 and for flagging cells
              (``tolProbCell``); the cutoffs for rows, slopes and correlations keep
              ``chi2_quantile``. Defaults to None, which uses ``chi2_quantile``.

        Attributes:
            location_ (np.ndarray): Robust location of each column.
            scale_ (np.ndarray): Robust scale of each column.
            cellwise_outliers_ (np.ndarray): Boolean array (n_samples, n_features), True for
              flagged cells.
            standardized_residuals_ (np.ndarray): Standardized residuals of the cells of the
              training data (``stdResid``).
            row_outliers_ (np.ndarray): Boolean array (n_samples,), True for flagged rows.
            row_scores_ (np.ndarray): Standardized row outlyingness ``Ti``; a row is flagged if
              its score is positive and exceeds the cutoff.
            rescaled_predictions_ (np.ndarray): Predicted values of all cells on the original
              scale (``Xest``).
            robust_correlation_ (np.ndarray): Robust correlation matrix of the variables (nan
              for pairs of predictor columns, which are not needed).
            slopes_ (np.ndarray): ``slopes_[j, i]`` is the robust slope used to predict column i
              from column j, nan if column j is not used for column i.
            neighbors_ (np.ndarray), neighbor_correlations_ (np.ndarray),
              neighbor_slopes_ (np.ndarray): For each column, the neighbors used to predict it
              (the column itself first), their robust correlations and slopes, as ``ngbrs``,
              ``robcors`` and ``robslopes`` in cellWise (0-based column indices).
            deshrinkage_ (np.ndarray): Deshrinkage factor of each predicted column (nan for
              columns that are not predicted).
            residual_scales (np.ndarray): Scale of the raw residuals of each predicted column
              (``scalestres``; nan for columns that are not predicted).
            predictor_mask_ (np.ndarray): Boolean array (n_features,), True for the predictor
              columns.

        References:
            - Rousseeuw, P. J., & Van den Bossche, W. (2018). Detecting deviating data cells.
              Technometrics, 60(2), 135-145.
            - Raymaekers, J., & Rousseeuw, P. J. (2023). cellWise: Analyzing Data with Cellwise
              Outliers. R package version 2.5.7.

        """
        self.chi2_quantile = chi2_quantile
        self.cell_quantile = cell_quantile
        self.cutoff = np.sqrt(
            chi2.ppf(chi2_quantile if cell_quantile is None else cell_quantile, df=1)
        )
        self._quantile_cutoff = np.sqrt(chi2.ppf(chi2_quantile, df=1))
        self.min_correlation = min_correlation
        self.scale_estimator = scale_estimator
        self.min_unique_values = min_unique_values
        self.predictor_columns = predictor_columns
        self.predictor_min_correlation = predictor_min_correlation
        self.center = center
        self.n_neighbors = n_neighbors

    _prec_scale = 1e-12

    # ------------------------------------------------------------------------------ fit

    def fit(self, X: pd.DataFrame, y=None, verbose: bool = False):
        """Fit DDC on X (a DataFrame; infinite values are treated as missing)."""
        X = X.replace([np.inf, -np.inf], np.nan)
        self._check_data_conditions(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        values = X.to_numpy(dtype=float)
        n, d = values.shape

        # step 1: standardization
        self._fit_scaler(values)
        Z = self._standardize_array(values)
        # step 2: univariate outlier detection
        U = self._univariate_outliers_array(Z)
        # step 3: robust correlations, neighbors and slopes
        self.robust_correlation_ = self._robust_correlation(U, verbose=verbose)
        self._select_neighbors(U)

        # step 4: predictions; step 5: deshrinkage
        raw = self._raw_predictions(U)
        self.raw_predictions_ = np.where(np.isfinite(raw), raw, 0.0)
        self.deshrinkage_ = np.full(d, np.nan)
        for i in self._connected:
            self.deshrinkage_[i] = _slope_med_wls(
                raw[:, i], Z[:, i], self._quantile_cutoff, self._prec_scale
            )
        predictions = self._deshrink(raw)
        self.predictions_ = predictions

        # step 6: cellwise outliers
        raw_residuals = self._raw_residuals(Z, predictions)
        self.residual_scales = np.full(d, np.nan)
        for i in self._connected:
            self.residual_scales[i] = _huber_scale(raw_residuals[:, i], self._prec_scale)
        self.standardized_residuals_ = self._standardize_residuals(raw_residuals)
        with np.errstate(invalid="ignore"):
            self.cellwise_outliers_ = np.abs(self.standardized_residuals_) > self.cutoff

        # step 7: rowwise outliers
        raw_scores = self._raw_row_scores(self.standardized_residuals_)
        finite = np.isfinite(raw_scores)
        self._row_score_median = float(np.median(raw_scores[finite]))
        self._row_score_mad = float(
            1.482602218505602 * np.median(np.abs(raw_scores[finite] - self._row_score_median))
        )
        self.row_scores_ = self._standardize_row_scores(raw_scores)
        self.row_outliers_ = self._flag_rows(self.row_scores_)

        # step 8: unstandardize
        self.rescaled_predictions_ = predictions * self.scale_ + self.location_
        self.is_fitted_ = True
        return self

    def _fit_scaler(self, values: np.ndarray):
        """Robust location and scale of the target columns, mean and standard deviation of the
        predictor columns. With a fixed center, the scales are computed around it."""
        d = values.shape[1]
        self.location_, self.scale_ = np.zeros(d), np.zeros(d)
        if self.center is not None:
            center = np.asarray(self.center, dtype=float).ravel()
            if center.size != d or np.isnan(center).any():
                raise ValueError(f"center should be a vector of length {d} without nan.")
            estimator = copy.deepcopy(self.scale_estimator)
            if not hasattr(estimator, "center"):
                raise ValueError("With a fixed center, scale_estimator needs a center option.")
            estimator.center = False
        else:
            estimator = self.scale_estimator
        for i in range(d):
            x = values[:, i]
            x = x[np.isfinite(x)]
            if self.predictor_mask_[i]:
                # predictor columns are clean and may be discrete: classical standardization
                self.location_[i], self.scale_[i] = np.mean(x), np.std(x, ddof=1)
            elif self.center is not None:
                self.location_[i] = center[i]
                self.scale_[i] = estimator.fit(x - center[i]).scale
            else:
                est = estimator.fit(x)
                self.location_[i], self.scale_[i] = est.location, est.scale

    def _standardize_array(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "location_"):
            raise ValueError("Tried to standardize but scaler not fitted yet.")
        return (values - self.location_) / self.scale_

    def _standardize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Standardized data as a DataFrame with the index and columns of X."""
        values = X.to_numpy(dtype=float)
        return pd.DataFrame(self._standardize_array(values), index=X.index, columns=X.columns)

    def _univariate_outliers_array(self, Z: np.ndarray) -> np.ndarray:
        """Step 2: set standardized cells beyond the cutoff to nan (not in predictor columns)."""
        with np.errstate(invalid="ignore"):
            outlying = (np.abs(Z) > self.cutoff) & ~self.predictor_mask_
        return np.where(outlying, np.nan, Z)

    def _univariate_outliers(self, Z: pd.DataFrame) -> pd.DataFrame:
        """Step 2 for a DataFrame of standardized data."""
        return pd.DataFrame(
            self._univariate_outliers_array(Z.to_numpy(dtype=float)),
            index=Z.index,
            columns=Z.columns,
        )

    def _robust_correlation(self, U: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Robust correlations of all pairs of columns that involve a target column: the
        ``gkwls`` correlation of cellWise between target columns, and the Pearson correlation
        over the observed cells of the target column for a predictor column."""
        d = U.shape[1]
        q_corr = chi2.ppf(self.chi2_quantile, 2)
        correlation = np.eye(d)
        for i in range(d):
            if verbose and i % max(d // 10, 1) == 0:
                print(f"Correlation: Processing column {i + 1} of {d} ({((i + 1) / d):.1%})")
            for j in range(i + 1, d):
                if self.predictor_mask_[i] and self.predictor_mask_[j]:
                    # not needed: predictor columns are not predicted
                    correlation[i, j] = correlation[j, i] = np.nan
                    continue
                if self.predictor_mask_[i] or self.predictor_mask_[j]:
                    # a predictor column can be discrete, so its joint distribution with a target
                    # column is not elliptical: Pearson correlation on the cells of the target
                    # column that survived step 2 (point-biserial for a binary predictor)
                    rows = np.isfinite(U[:, i]) & np.isfinite(U[:, j])
                    r = np.corrcoef(U[rows, i], U[rows, j])[0, 1] if rows.sum() > 2 else np.nan
                    correlation[i, j] = correlation[j, i] = 0.0 if np.isnan(r) else r
                    continue
                correlation[i, j] = correlation[j, i] = _corr_gkwls(
                    U[:, i], U[:, j], q_corr, self._prec_scale
                )
        return correlation

    def _min_correlations(self) -> np.ndarray:
        """Per column: minimum absolute correlation for using it to predict another column."""
        predictor_min_correlation = (
            self.min_correlation
            if self.predictor_min_correlation is None
            else self.predictor_min_correlation
        )
        return np.where(self.predictor_mask_, predictor_min_correlation, self.min_correlation)

    def _select_neighbors(self, U: np.ndarray):
        """For each target column: the k columns with the highest absolute robust correlation
        (``kBestCorr``), their weights (absolute correlations below the minimum set to 0) and
        robust slopes (``compSlopes``). The column itself is then added as first neighbor with
        weight and slope 1 (``includeSelf``). Predictor columns get no neighbors."""
        d = U.shape[1]
        k = min(self.n_neighbors, d - 1)
        min_correlations = self._min_correlations()
        q_regr = self._quantile_cutoff
        neighbors = np.zeros((d, k), dtype=int)
        correlations = np.zeros((d, k))
        weights = np.zeros((d, k))
        slopes = np.zeros((d, k))
        for i in range(d):
            if self.predictor_mask_[i]:
                continue
            corrs = np.nan_to_num(self.robust_correlation_[:, i], nan=0.0)
            order = np.argsort(-np.abs(corrs), kind="stable")
            order = order[order != i][:k]
            neighbors[i], correlations[i] = order, corrs[order]
            w = np.abs(corrs[order])
            thresholds = min_correlations[order]
            w[(thresholds > 0) & (w < thresholds)] = 0.0
            weights[i] = w
            for m in np.flatnonzero(w > 0):
                slopes[i, m] = _slope_med_wls(U[:, order[m]], U[:, i], q_regr, self._prec_scale)
        target = ~self.predictor_mask_
        self._connected = np.flatnonzero(target & (weights.sum(axis=1) > 0))
        self._standalone = np.flatnonzero(target & (weights.sum(axis=1) == 0))
        self.neighbors_ = np.column_stack([np.arange(d), neighbors])
        self.neighbor_correlations_ = np.column_stack([np.ones(d), correlations])
        self.neighbor_slopes_ = np.column_stack([np.ones(d), slopes])
        self._neighbor_weights = np.column_stack([np.ones(d), weights])
        self.slopes_ = np.full((d, d), np.nan)
        for i in range(d):
            for m in np.flatnonzero(weights[i] > 0):
                self.slopes_[neighbors[i, m], i] = slopes[i, m]

    def _raw_predictions(self, U: np.ndarray) -> np.ndarray:
        """Step 4 (``predictCol``): predict each connected column by the weighted mean of the
        slope-times-neighbor predictions over its finite neighbor cells (the cell itself
        included); other columns keep their values of U."""
        estimates = U.copy()
        for i in self._connected:
            w = self._neighbor_weights[i]
            used = w > 0
            if used.sum() <= 1:
                estimates[:, i] = 0.0
                continue
            ngb = self.neighbors_[i, used]
            contributions = U[:, ngb] * self.neighbor_slopes_[i, used] * w[used]
            finite = np.isfinite(contributions)
            with np.errstate(invalid="ignore", divide="ignore"):
                estimates[:, i] = np.where(finite, contributions, 0.0).sum(axis=1) / (
                    finite * w[used]
                ).sum(axis=1)
        return estimates

    def _deshrink(self, raw: np.ndarray) -> np.ndarray:
        """Step 5: multiply the predictions of the connected columns by their deshrinkage
        factors; then replace all missing predictions by 0."""
        predictions = raw.copy()
        predictions[:, self._connected] *= self.deshrinkage_[self._connected]
        return np.where(np.isfinite(predictions), predictions, 0.0)

    def _raw_residuals(self, Z: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Standardized cells minus their predictions; Z itself for standalone columns and nan
        for predictor columns."""
        residuals = Z - predictions
        residuals[:, self._standalone] = Z[:, self._standalone]
        residuals[:, self.predictor_mask_] = np.nan
        return residuals

    def _standardize_residuals(self, raw_residuals: np.ndarray) -> np.ndarray:
        residuals = raw_residuals.copy()
        residuals[:, self._connected] /= self.residual_scales[self._connected]
        return residuals

    @staticmethod
    def _raw_row_scores(standardized_residuals: np.ndarray) -> np.ndarray:
        """Row outlyingness before standardization: the mean over the observed cells of the
        chi-squared(1) probability of the squared standardized residual, minus 0.5."""
        with np.errstate(invalid="ignore"):
            probabilities = erf(np.sqrt(standardized_residuals**2 / 2))
        finite = np.isfinite(probabilities)
        counts = finite.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(finite, probabilities, 0.0).sum(axis=1) / counts - 0.5

    def _standardize_row_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        return (raw_scores - self._row_score_median) / self._row_score_mad

    def _flag_rows(self, row_scores: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return (np.abs(row_scores) > self._quantile_cutoff) & (row_scores > 0)

    # ------------------------------------------------------------------------------ new data

    def predict(self, X: pd.DataFrame, rowwise: bool = False) -> np.ndarray:
        """Flag the cellwise (or rowwise) outliers of new data with the fitted model, as
        ``cellWise::DDCpredict``.

        Args:
            X (pd.DataFrame): New data to predict outliers for.
            rowwise (bool, optional): Whether to predict rowwise instead of cellwise outliers.
                Defaults to False.

        Raises:
            ValueError: Model not fitted.
            ValueError: Data shape mismatch.

        Returns:
            np.ndarray:
            - If rowwise is True: A boolean array of shape (n_samples,) with rowwise outliers.
            - If rowwise is False: A boolean array of shape (n_samples, n_features) with cellwise
              outliers. Use `standardized_residuals` to get the standardized residuals.
        """
        flags, residuals, _ = self._predict_cells(X)
        if rowwise:
            return self._flag_rows(self._standardize_row_scores(self._raw_row_scores(residuals)))
        return flags

    def standardized_residuals(self, X: pd.DataFrame) -> np.ndarray:
        """Standardized residuals of the cells of (new) data, i.e. the difference between each
        standardized cell and its prediction from the cells of the row, divided by the robust
        scale of the training residuals. Cells with an absolute standardized residual above the
        cutoff are flagged by `predict`.

        Args:
            X (pd.DataFrame): Data with the same columns as the data used in fit.

        Returns:
            np.ndarray: Array of shape (n_samples, n_features); nan for missing cells. Can be
                passed to `cellmap` to visualize new data.
        """
        return self._predict_cells(X)[1]

    def impute(self, X: pd.DataFrame, impute_outliers: bool = True) -> pd.DataFrame:
        """Replace missing (and, optionally, flagged) cells by their DDC predictions.

        Args:
            X (pd.DataFrame): Data with the same columns as the data used in fit.
            impute_outliers (bool, optional): Whether to also impute the cells flagged by
                `predict`. Defaults to True.

        Returns:
            pd.DataFrame: X with imputed cells (``Ximp`` in cellWise). Predictor columns are
                returned unchanged.
        """
        X = self._check_new_data(X, "Impute")
        flags, _, estimates = self._predict_cells(X)
        values = X.to_numpy(dtype=float)
        replace = np.isnan(values) | (flags if impute_outliers else False)
        return pd.DataFrame(np.where(replace, estimates, values), index=X.index, columns=X.columns)

    def _predict_cells(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cellwise outlier flags, standardized residuals and predictions (on the original
        scale) for (new) data, with the fitted neighbors, slopes and scales."""
        X = self._check_new_data(X, "Predict")
        Z = self._standardize_array(X.to_numpy(dtype=float))
        U = self._univariate_outliers_array(Z)
        predictions = self._deshrink(self._raw_predictions(U))
        residuals = self._standardize_residuals(self._raw_residuals(Z, predictions))
        with np.errstate(invalid="ignore"):
            flags = np.abs(residuals) > self.cutoff
        return flags, residuals, predictions * self.scale_ + self.location_

    def _check_new_data(self, X: pd.DataFrame, method: str) -> pd.DataFrame:
        """Check new data passed to predict / impute and replace infinite values by nan."""
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model not fitted yet.")
        if not X.shape[1] == self.cellwise_outliers_.shape[1]:
            raise ValueError(
                f"{method} can only be called with the same data as fit. "
                f"Received {X.shape[1]} columns, expected {self.cellwise_outliers_.shape[1]}"
            )
        if X.select_dtypes(include=np.number).shape != X.shape:
            raise ValueError("Only numerical data is supported.")
        X = X.replace([np.inf, -np.inf], np.nan)
        if self.predictor_mask_.any():
            if list(X.columns) != list(self.feature_names_in_):
                raise ValueError(
                    "With predictor_columns, X must have the same columns as in fit, in the "
                    "same order."
                )
            missing = X.columns[self.predictor_mask_ & X.isna().any(axis=0).to_numpy()]
            if len(missing) > 0:
                raise ValueError(
                    f"Predictor columns cannot contain missing values: {list(missing)}."
                )
        return X

    # ------------------------------------------------------------------------------ plots

    def cellmap(
        self,
        X: pd.DataFrame,
        standardized_residuals: np.ndarray | None = None,
        annotate: bool = False,
        fmt: str = ".1f",
        figsize: tuple[int, int] = (7, 10),
        row_zoom: tuple[int, int] | pd.Index | None = None,
        col_zoom: tuple[int, int] | pd.Index | None = None,
        vmax_clip: float = float(np.sqrt(stats.chi2.ppf(0.999, df=1))),
        cmap: str | matplotlib.colors.Colormap = "custom",
    ) -> Axes:
        """
        Visualize the standardized residuals of the DDC model as a heatmap. Predictor columns
        (see `predictor_columns`) have no residuals and are shown blank.

        Args:
            X (pd.DataFrame): The data used to predict the residuals.
            standardized_residuals (np.ndarray | None, optional): if X is not the original data used
                to fit the model, the standardized residuals of the cells predicted on the new X
                data should be passed (see `standardized_residuals`).
            annotate (bool, optional): Whether to annotate the heatmap cells
                with the original values. Defaults to False.
            fmt (str, optional): Format to use for annotations. Defaults to ".1f".
            figsize (tuple[int, int], optional): Figure size. Defaults to (7, 10).
            row_zoom (tuple[int, int] | pd.Index | None, optional):
                If not None, a subset of the rows is selected for visualization.
                A tuple is interpreted as a slice, a pd.Index as a selection.
                Defaults to None.
            col_zoom (tuple[int, int] | pd.Index | None, optional):
                Similar to row_zoom but for columns. Defaults to None.
            vmax_clip (float): standardized absolute residuals larger than vmax will get the darkest
                color and hence get clipped
            cmap (str | matplotlib.colors.Colormap, optional): matplotlib colormap or string, maps
                the data to the color space.

        Returns:
            Axes: The matplotlib axes with the heatmap.
        """
        if not getattr(self, "is_fitted_", False):
            raise ValueError("Model not fitted yet.")
        if cmap == "custom":
            cmap = get_custom_cmap(vmax_clip)
        if standardized_residuals is None:
            standardized_residuals = self.standardized_residuals_
        plot_data = pd.DataFrame(standardized_residuals, index=X.index, columns=X.columns)
        X_annot = X
        if row_zoom is not None:
            if isinstance(row_zoom, tuple):
                plot_data = plot_data.iloc[row_zoom[0] : row_zoom[1], :]
                X_annot = X_annot.iloc[row_zoom[0] : row_zoom[1], :]
            else:
                plot_data = plot_data.loc[row_zoom, :]
                X_annot = X.loc[row_zoom, :]
        if col_zoom is not None:
            if isinstance(col_zoom, tuple):
                plot_data = plot_data.iloc[:, col_zoom[0] : col_zoom[1]]
                X_annot = X_annot.iloc[:, col_zoom[0] : col_zoom[1]]
            else:
                plot_data = plot_data.loc[:, col_zoom]
                X_annot = X_annot.loc[:, col_zoom]
        if annotate:
            annotate = X_annot
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax = sns.heatmap(
            plot_data,
            cmap=cmap,
            annot=annotate,
            cbar_kws={"label": "Standardized residuals"},
            fmt=fmt,
            ax=ax,
            vmin=-vmax_clip,
            vmax=vmax_clip,
            linewidths=1 / len(plot_data) * figsize[1] * 0.1,
            linecolor="white",
        )
        return ax

    def _check_data_conditions(self, X: pd.DataFrame):
        if X.select_dtypes(include=np.number).shape != X.shape:
            raise ValueError("Only numerical data is supported.")
        self.predictor_mask_ = self._get_predictor_mask(X)
        if self.predictor_mask_.all():
            raise ValueError("At least one column must not be a predictor column.")
        predictors = X.loc[:, self.predictor_mask_]
        with_nan = predictors.columns[predictors.isna().any(axis=0)]
        if len(with_nan) > 0:
            raise ValueError(f"Predictor columns cannot contain missing values: {list(with_nan)}.")
        constant = predictors.columns[predictors.nunique() <= 1]
        if len(constant) > 0:
            raise ValueError(f"Predictor columns cannot be constant: {list(constant)}.")
        targets = X.loc[:, ~self.predictor_mask_]
        discrete = targets.columns[targets.nunique() <= self.min_unique_values]
        if len(discrete) > 0:
            raise ValueError(
                f"Columns with {self.min_unique_values} or fewer unique values are not supported "
                f"(see min_unique_values), except as predictor columns: {list(discrete)}."
            )

    def _get_predictor_mask(self, X: pd.DataFrame) -> np.ndarray:
        """Boolean array (n_features,) marking the predictor columns."""
        names = self.predictor_columns
        if names is None:
            return np.zeros(X.shape[1], dtype=bool)
        if isinstance(names, str) or not np.iterable(names):
            names = [names]
        unknown = [name for name in names if name not in X.columns]
        if unknown:
            raise ValueError(f"Unknown predictor columns: {unknown}.")
        return np.asarray(X.columns.isin(names))
