import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.stats as stats

from matplotlib.axes import Axes
from sklearn.base import OutlierMixin
from scipy.stats import chi2

from robpy.univariate import CellwiseOneStepMEstimator
from robpy.utils.distance import mahalanobis_distance
from robpy.univariate.base import RobustScaleEstimator


def get_custom_cmap(vmax_clip: int):
    norm = matplotlib.colors.Normalize(-vmax_clip, vmax_clip)
    colors = [
        [norm(-vmax_clip), "#4652a3"],
        [norm(-(vmax_clip + 2.5) / 2), "#8d6d8c"],
        [norm(-2.5), "#f6eb15"],
        [norm(2.5), "#f6eb15"],
        [norm(vmax_clip), "#ec2123"],
    ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


class DDCEstimator(OutlierMixin):
    def __init__(
        self,
        chi2_quantile: float = 0.99,
        min_correlation: float = 0.5,
        scale_estimator: RobustScaleEstimator = CellwiseOneStepMEstimator(),
    ):
        """Implementation of the Detecting Deviating Cells (DDC) algorithm.

        Args:
            chi2_quantile (float, optional): Quantile of the chi-squared distribution to use as
              threshold for univariate outlier detection in step 2.
              Default is 0.99.
            min_correlation (float, optional): Minimum correlation between variables to consider
              them
            scale_estimator (RobustScaleEstimator, optional) : robust scale estimator to scale the
              initial data with. Defaults to CellwiseOneStepMEstimator().

        References:
            Rousseeuw, P. J., & Bossche, W. V. D. (2018). Detecting Deviating Data Cells.
            Technometrics, 60(2), 135–145. https://doi.org/10.1080/00401706.2017.1340909

            R Implementation:
            https://www.rdocumentation.org/packages/cellWise/versions/2.5.3/topics/DDC

        """
        self.chi2_quantile = chi2_quantile
        self.cutoff = np.sqrt(chi2.ppf(chi2_quantile, df=1))
        self.min_correlation = min_correlation
        self.scale_estimator = scale_estimator

    def fit(self, X: pd.DataFrame, y=None, verbose: bool = False):
        self._check_data_conditions(X)
        X = X.replace([np.inf, -np.inf], np.nan)
        # step 1: standardization
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
        self.raw_residuals_ = Z.values - self.predictions_
        self.residual_scales_ = np.array(
            [
                CellwiseOneStepMEstimator()
                .fit(self.raw_residuals_[:, i][~np.isnan(self.raw_residuals_[:, i])])
                .scale
                for i in range(X.shape[1])
            ]
        )
        self.standardized_residuals_ = self.raw_residuals_ / self.residual_scales_
        self.cellwise_outliers_ = np.abs(self.standardized_residuals_) > self.cutoff
        # step 7: rowwise outliers
        self.raw_t_values_ = np.nanmean(chi2.cdf(self.standardized_residuals_**2, df=1), axis=1)
        est = CellwiseOneStepMEstimator().fit(self.raw_t_values_)
        self.standardized_t_values_ = (self.raw_t_values_ - est.location) / est.scale
        self.row_outliers_ = self.standardized_t_values_ > self.cutoff
        self.is_fitted_ = True
        # step 8: rescale
        self.rescaled_predictions_ = self.predictions_ * self.scale_ + self.location_
        return self

    def predict(self, X, rowwise: bool = False):
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        if X.shape != self.cellwise_outliers_.shape:
            raise ValueError("Predict can only be called with the same data as fit.")
        if rowwise:
            return self.row_outliers_

        return self.cellwise_outliers_

    def impute(self, X, impute_outliers: bool = True):
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        if X.shape[1] != self.cellwise_outliers_.shape[1]:
            raise ValueError("Impute can only be called with the same data as fit.")
        if impute_outliers:
            results = np.where(
                self.cellwise_outliers_ | np.isnan(X.replace([-np.inf, np.inf], np.nan)),
                self.rescaled_predictions_,
                X,
            )
        else:
            results = np.where(
                np.isnan(X.replace([-np.inf, np.inf], np.nan)), self.rescaled_predictions_, X
            )

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(results, index=X.index, columns=X.columns)
        else:
            return results

    def cellmap(
        self,
        X: pd.DataFrame,
        annotate: bool = False,
        fmt: str = ".1f",
        figsize: tuple[int, int] = (7, 10),
        row_zoom: tuple[int, int] | pd.Index | None = None,
        col_zoom: tuple[int, int] | pd.Index | None = None,
        vmax_clip: float = np.sqrt(stats.chi2.ppf(0.999, df=1)),
        cmap: str | matplotlib.colors.Colormap = "custom",
    ) -> Axes:
        """Visualize the standardized residuals of the DDC model as a heatmap.

        Args:
            X (pd.DataFrame): The original data used to fit the model.
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
            Axes: the matplotlib axes with the heatmap
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted yet.")
        if cmap == "custom":
            cmap = get_custom_cmap(vmax_clip)
        plot_data = pd.DataFrame(self.standardized_residuals_, index=X.index, columns=X.columns)
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

    def _standardize(self, X: pd.DataFrame):
        self.location_, self.scale_ = [], []
        for i in range(X.shape[1]):
            est = self.scale_estimator.fit(X.iloc[:, i].to_numpy()[~np.isnan(X.iloc[:, i])])
            self.location_.append(est.location)
            self.scale_.append(est.scale)
        return (X - self.location_) / self.scale_

    def _robust_correlation(self, X: pd.DataFrame, verbose: bool = False) -> np.ndarray:
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
                            CellwiseOneStepMEstimator().fit(x + y).scale ** 2
                            - CellwiseOneStepMEstimator().fit(x - y).scale ** 2
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
        r_cutoff = self.cutoff * CellwiseOneStepMEstimator().fit(residuals).scale
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
