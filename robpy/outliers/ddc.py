import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats as stats

from matplotlib.axes import Axes
from sklearn.base import OutlierMixin
from scipy.stats import chi2

from robpy.univariate import CellwiseOneStepM
from robpy.utils.distance import mahalanobis_distance
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


class DDC(OutlierMixin):
    def __init__(
        self,
        chi2_quantile: float = 0.99,
        min_correlation: float = 0.5,
        scale_estimator: RobustScale = CellwiseOneStepM(),
    ):
        """
        Implementation of the Detecting Deviating Cells (DDC) algorithm. Based on the R
        implementation in the package cellWise.

        Args:
            chi2_quantile (float, optional): Quantile of the chi-squared distribution to use as
              threshold for univariate outlier detection in step 2.
              Default is 0.99.
            min_correlation (float, optional): Minimum correlation between variables to consider
              them. Default is 0.5.
            scale_estimator (RobustScale, optional) : robust scale estimator to scale the
              initial data with. Defaults to CellwiseOneStepM().

        References:
            - Rousseeuw, P. J., & Van den Bossche, W. (2018). Detecting deviating data cells.
              Technometrics, 60(2), 135-145.

        """
        self.chi2_quantile = chi2_quantile
        self.cutoff = np.sqrt(chi2.ppf(chi2_quantile, df=1))
        self.min_correlation = min_correlation
        self.scale_estimator = scale_estimator

    def fit(self, X: pd.DataFrame, y=None, verbose: bool = False):
        self._check_data_conditions(X)
        X = X.replace([np.inf, -np.inf], np.nan)
        # step 1: standardization
        self._fit_scaler(X)
        Z = self._standardize(X)
        # step 2: univariate outlier detection
        U = pd.DataFrame(
            data=np.where(np.abs(Z) > self.cutoff, np.nan, Z),
            columns=X.columns,
            index=X.index,
        )
        # step 3: correlation matrix and slopes
        self.robust_correlation_ = self._robust_correlation(U, verbose=verbose)
        self.slopes_ = self._get_slopes(U)

        # step 4: predicted values
        self.raw_predictions_ = self._get_predicted_values(U)

        # step 5: deshrinkage
        self.deshrinkage_ = np.array(
            [
                self._robust_slope(
                    self.raw_predictions_[~np.isnan(Z.iloc[:, i]), i],
                    Z.to_numpy()[~np.isnan(Z.iloc[:, i]), i],
                )
                for i in range(X.shape[1])
            ]
        )
        self.predictions_ = self.raw_predictions_ * self.deshrinkage_

        # step 6: cellwise outliers
        self.cellwise_outliers_, self.standardized_residuals_ = self._cellwise_outliers(
            Z, self.predictions_, fit=True
        )
        # step 7: rowwise outliers
        self.row_outliers_ = self._rowwise_outliers(self.standardized_residuals_, fit=True)
        self.is_fitted_ = True
        # step 8: rescale
        self.rescaled_predictions_ = self.predictions_ * self.scale_ + self.location_
        return self

    def predict(self, X: pd.DataFrame, rowwise: bool = False) -> np.ndarray:
        """Predict outliers in the data.

        Args:
            X (pd.DataFrame): New data to predict outliers for.
            rowwise (bool, optional): Whether to predict rowwise instead of cellwise outliers.
                Defaults to False.

        Raises:
            ValueError: Model not fitted.
            ValueError: Data shape mismatch.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - If rowwise is True: A 1D array of shape (n_samples,) with rowwise outliers.
            - If rowwise is False: A matrix of shape (n_samples, n_features) with cellwise outliers
              and an array containing the standardized residuals of the cells.
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        if not X.shape[1] == self.cellwise_outliers_.shape[1]:
            raise ValueError(
                f"Predict can only be called with the same data as fit. "
                f"Received {X.shape[1]} columns, expected {self.cellwise_outliers_.shape[1]}"
            )
        if X.select_dtypes(include=np.number).shape != X.shape:
            raise ValueError("Only numerical data is supported.")
        X = X.replace([np.inf, -np.inf], np.nan)
        # step 1: standardization
        Z = self._standardize(X)
        U = pd.DataFrame(
            data=np.where(np.abs(Z) > self.cutoff, np.nan, Z),
            columns=X.columns,
            index=X.index,
        )
        predictions = self._get_predicted_values(U) * self.deshrinkage_
        cellwise_outliers, standardized_residuals = self._cellwise_outliers(Z, predictions)

        if rowwise:
            return self._rowwise_outliers(standardized_residuals)

        return cellwise_outliers, standardized_residuals

    def _cellwise_outliers(
        self, Z: pd.DataFrame, predictions: np.ndarray, fit: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get cellwise outliers boolean indicator matrix.

        Args:
            Z (pd.DataFrame): Standardized input data.
            predictions (np.ndarray): Predicted standardized values (deshrinked).
            fit (bool, optional): Whether to fit the scale estimator. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - cellwise_outliers (np.ndarray): Boolean indicator matrix of cellwise outliers.
            - standardized_residuals (np.ndarray): Standardized residuals.
        """
        raw_residuals = Z.values - predictions
        if fit:
            self.residual_scales = np.array(
                [
                    CellwiseOneStepM()
                    .fit(raw_residuals[:, i][~np.isnan(raw_residuals[:, i])])
                    .scale
                    for i in range(raw_residuals.shape[1])
                ]
            )
        standardized_residuals = raw_residuals / self.residual_scales
        cellwise_outliers = np.abs(standardized_residuals) > self.cutoff

        return cellwise_outliers, standardized_residuals

    def _rowwise_outliers(
        self, standardized_residuals: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """Calculate rowwise outliers based on standardized residuals.

        Args:
            standardized_residuals (np.ndarray)
            fit (bool, optional): Whether to fit the scale estimator. Defaults to False.

        Returns:
            np.ndarray: array of shape (n_samples,) with boolean rowwise outliers
        """
        raw_t_values_ = np.nanmean(chi2.cdf(standardized_residuals**2, df=1), axis=1)
        if fit:
            self.t_scale_est = CellwiseOneStepM().fit(raw_t_values_)
        standardized_t_values_ = (
            raw_t_values_ - self.t_scale_est.location
        ) / self.t_scale_est.scale
        row_outliers = standardized_t_values_ > self.cutoff
        return row_outliers

    def impute(self, X: pd.DataFrame, impute_outliers: bool = True) -> pd.DataFrame:
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        if X.shape[1] != self.cellwise_outliers_.shape[1]:
            raise ValueError(
                f"Impute can only be called with the same data as fit. "
                f"Received {X.shape[1]} columns, expected {self.cellwise_outliers_.shape[1]}"
            )
        X = X.replace([np.inf, -np.inf], np.nan)
        Z = self._standardize(X)
        # step 2: univariate outlier detection
        U = pd.DataFrame(
            data=np.where(np.abs(Z) > self.cutoff, np.nan, Z),
            columns=X.columns,
            index=X.index,
        )
        raw_predictions = self._get_predicted_values(U)
        predictions = raw_predictions * self.deshrinkage_
        rescaled_predictions = (predictions * self.scale_ + self.location_).round(3)

        if impute_outliers:
            cellwise_outliers, _ = self.predict(X)
            results = np.where(
                cellwise_outliers | np.isnan(X.replace([-np.inf, np.inf], np.nan)),
                rescaled_predictions,
                X,
            )
        else:
            results = np.where(
                np.isnan(X.replace([-np.inf, np.inf], np.nan)), rescaled_predictions, X
            )

        return pd.DataFrame(results, index=X.index, columns=X.columns)

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
        Visualize the standardized residuals of the DDC model as a heatmap.

        Args:
            X (pd.DataFrame): The data used to predict the residuals.
            standardized_residuals (np.ndarray | None, optional): if X is not the original data used
                to fit the model, the standardized residuals of the cells predicted on the new X
                data should be passed.
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
        if not self.is_fitted_:
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
        if any(X.nunique() <= 3):
            raise ValueError("Columns with less than 3 unique values are not supported.")

    def _fit_scaler(self, X: pd.DataFrame):
        self.location_, self.scale_ = [], []
        for i in range(X.shape[1]):
            est = self.scale_estimator.fit(X.iloc[:, i].to_numpy()[~np.isnan(X.iloc[:, i])])
            self.location_.append(est.location)
            self.scale_.append(est.scale)

    def _standardize(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "location_"):
            raise ValueError("Tried to standardize but scaler not fitted yet.")
        return (X - self.location_) / self.scale_

    def _robust_correlation(self, X: pd.DataFrame, verbose: bool = False) -> np.ndarray:
        """Calculate a correlation matrix with shape (n_feat, n_feat)
        using a robust scale estimator.

        Args:
            X (pd.DataFrame): The standardized data.
            verbose (bool, optional): Whether to print progress. Defaults to False.

        Returns:
            np.ndarray: correlation matrix (shape = (X.shape[1], X.shape[1])).
        """
        correlation = np.ones((X.shape[1], X.shape[1]))
        for i in range(X.shape[1]):
            if verbose and i % (X.shape[1] // 10) == 0:
                print(
                    f"Correlation: Processing column {i + 1} of {X.shape[1]} "
                    f"({((i + 1) / X.shape[1]):.1%})"
                )
            for j in range(i + 1, X.shape[1]):
                xy = X.iloc[:, [i, j]].values
                xy = xy[~(np.isnan(xy).any(axis=1)), :]
                if xy.size == 0:
                    correlation[i, j] = correlation[j, i] = 0
                    continue
                x, y = xy.T
                xy_corr = np.clip(
                    (
                        (
                            CellwiseOneStepM().fit(x + y).scale ** 2
                            - CellwiseOneStepM().fit(x - y).scale ** 2
                        )
                        / 4
                    ),
                    -0.99,
                    0.99,
                )
                distances = mahalanobis_distance(
                    xy, np.array([0, 0]), np.array([[1, xy_corr], [xy_corr, 1]])
                )
                xy_subset = xy[distances < self.cutoff, :]
                correlation[i, j] = correlation[j, i] = np.corrcoef(xy_subset, rowvar=False)[0, 1]
        return correlation

    def _robust_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        if np.all(x == 0):
            return 0
        init_slope = np.median(y[x != 0] / x[x != 0])
        residuals = y - init_slope * x
        r_cutoff = self.cutoff * CellwiseOneStepM().fit(residuals).scale
        mask = np.abs(residuals) <= r_cutoff

        return np.linalg.lstsq(x[mask].reshape(-1, 1), y[mask], rcond=None)[0][0]

    def _get_slopes(self, X: pd.DataFrame) -> np.ndarray:
        slopes = np.full((X.shape[1], X.shape[1]), np.nan)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if i == j:
                    continue
                xy = X.iloc[:, [i, j]].values
                xy = xy[(~np.isnan(xy).any(axis=1)), :].T
                x, y = xy
                xy_corr = self.robust_correlation_[i, j]
                if np.abs(xy_corr) < self.min_correlation:
                    continue
                slopes[i, j] = self._robust_slope(x, y)
        return slopes

    def _get_predicted_values(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(X.shape)
        for i in range(X.shape[1]):
            weights = np.repeat(
                [
                    [
                        np.abs(r) * (np.abs(r) >= 0.5) * (idx != i)
                        for idx, r in enumerate(self.robust_correlation_[:, i])
                    ]
                ],
                X.shape[0],
                axis=0,
            ) * ~np.isnan(X.values)

            weights_sum = weights.sum(axis=1).reshape(-1, 1)
            weights = np.divide(
                weights, weights_sum, out=np.zeros_like(weights), where=weights_sum != 0
            )
            for j in range(X.shape[1]):
                if np.isnan(self.slopes_[j, i]):
                    continue
                predictions[:, i] += weights[:, j] * self.slopes_[j, i] * X.iloc[:, j].fillna(0)
        return predictions
