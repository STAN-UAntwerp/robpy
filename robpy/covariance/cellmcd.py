import logging
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2, median_abs_deviation
from typing import Literal, Optional
from robpy.covariance.utils.alter_covariance import truncated_covariance
from robpy.covariance.base import RobustCovarianceEstimator
from robpy.utils.logging import get_logger
from robpy.covariance.utils.cellmcd_utils import objective_function
from robpy.utils.general import inverse_submatrix
from robpy.covariance.utils.cellmcd_visualization_utils import (
    annote_outliers,
    annote_outliers_ellipse,
    draw_ellipse,
    draw_threshold_lines,
    get_thresholds,
)
from robpy.preprocessing.scaling import RobustScaler
from robpy.univariate.onestep_m import OneStepWrappingEstimator
from robpy.covariance.initial_ddcw import InitialDDCWEstimator
from sklearn.exceptions import NotFittedError


class CellMCDEstimator(RobustCovarianceEstimator):
    def __init__(
        self,
        *,
        alpha: float = 0.75,
        quantile: float = 0.99,
        crit: float = 1e-4,
        max_c_steps: int = 100,
        min_eigenvalue: float = 1e-4,
        verbosity: int = logging.WARNING,
    ):
        """
        Cell MCD estimator based on the algorithm proposed in Raymaekers and Rousseeuw (2023).


        Args:
            alpha (float, optional):
                Percentage indicating how much cells must remain unflagged in each column.
                Defaults to 0.75.
            quantile (float, optional):
                Cutoff value to flag cells.
                Defaults to 0.99.
            crit (float, optional):
                Stop iterating when successive covariance matrices of the standardized data
                differ by less than crit.
                Defaults to 1e-4
            max_c_steps (int, optional):
                Maximum number of C-steps used in the algorithm.
                Defaults to 100.
            min_eigenvalue: (float, optional):
                Lower bound on the minimum eigenvalue of the covariance estimator
                on the standardized data. Should be at least 1e-6.
                Defaults to 1e-4.

        References:
            - Raymaekers and Rousseeuw, The Cellwise Minimum Covariance Determinant Estimator, 2023,
              Journal of the American Statistical Association.
        """
        if min_eigenvalue < 1e-6:
            raise ValueError("The lower bound on the eigenvalues should be at least 1e-6.")
        super().__init__(store_precision=False, assume_centered=False, nans_allowed=True)
        self.alpha = alpha
        self.quantile = quantile
        self.crit = crit
        self.max_c_steps = max_c_steps
        self.min_eigenvalue = min_eigenvalue
        self.logger = get_logger("CellMCDEstimator", level=verbosity)
        self.verbosity = verbosity

    def calculate_covariance(self, X: np.ndarray) -> np.ndarray:
        # Step 0: robustly standardize the data
        mads = median_abs_deviation(X, nan_policy="omit", axis=0)
        if np.min(mads) < 1e-8:
            raise ValueError("At least one variable has an almost zero median absolute deviation.")
        scaler = RobustScaler(scale_estimator=OneStepWrappingEstimator())
        scaler.fit(X, ignore_nan=True)
        X_scaled = scaler.transform(X)

        # Step 1: Check that there aren't too many marginal outliers and too many nan's.
        #         Check that there are enough observations compared to variables.
        self._check_data(X_scaled)

        # Step 2: calculate the initial estimates
        ddcw = InitialDDCWEstimator(alpha=self.alpha, min_eigenvalue=self.min_eigenvalue)
        ddcw.fit(X_scaled)
        initial_location = ddcw.location_
        initial_cov = ddcw.covariance_
        initial_cov_inv = ddcw.precision_
        X_scaled[np.abs(X_scaled) > 3] = np.nan

        # Step 3: MCD iterations (we work with X_scaled)
        location, cov, cov_inv, W = self._C_steps_until_convergence(
            X_scaled, initial_location, initial_cov, initial_cov_inv
        )
        # Step 4: make prediction
        predictions, conditional_variances = self._make_predictions(
            X_scaled, cov, cov_inv, location, W
        )

        # Step 5: transform back
        cov = np.diag(scaler.scales_) @ cov @ np.diag(scaler.scales_)
        location = scaler.locations_ + location * scaler.scales_
        predictions = scaler.locations_ + predictions * scaler.scales_
        conditional_stds = np.sqrt(conditional_variances) * scaler.scales_
        X_imputed = X.copy()
        X_imputed[W == 0] = predictions[W == 0]
        residuals = (X - predictions) / conditional_stds
        scaler_residuals = RobustScaler(scale_estimator=OneStepWrappingEstimator())
        scaler_residuals.fit(residuals, ignore_nan=True)
        residuals = residuals / scaler_residuals.scales_

        self.W = W
        self.location_ = location
        self.predictions = predictions
        self.conditional_stds = conditional_stds
        self.X_imputed = X_imputed
        self.residuals = residuals
        self.covariance_ = cov
        self.X = X

        return cov

    def cell_MCD_plot(
        self,
        variable: int,
        variable_name: str = "variable",
        row_names: Optional[list] = None,
        second_variable: Optional[int] = None,
        second_variable_name: str = "second variable",
        plottype: Literal[
            "indexplot",
            "residuals_vs_variable",
            "residuals_vs_predictions",
            "variable_vs_predictions",
            "bivariate",
        ] = "indexplot",
        figsize: tuple[int, int] = (8, 8),
    ):
        """
        Function to plot the results of a cellMCD analysis: 5 types of diagnostic plots.

        Arguments:
            plottype (Literal string, optional):
                 "indexplot": plots the residuals of a variable,
                 "residuals_vs_variable": plots a variable versus its residuals,
                 "residuals_vs_predictions": plots the predictions of a variable versus its
                 residuals,
                 "variable_vs_predictions": plots a variable against its predictions,
                 "bivariate": plots two variables against each other,

                 Defaults to "indexplot".
            variable (int): Index of the variable under consideration.
            variable_name (str, optional): Name of the variable of interest for the axis label.
              Defaults to "variable".
            second_variable (int): Index of the second variable under consideration,
              only needed for plottype "bivariate".
            second_variable_name (str, optional): Name of the second variable for the axis label,
              only relevant for plottype "bivariate". Defaults to "second variable".
            row_names (list of strings, optional): Row_names of the observations if you want
              the outliers annoted with their name.
            figsize (tuple[int,int], optional): Size of the figure. Defaults to (8,8).
        """

        if not hasattr(self, "covariance_"):
            raise NotFittedError()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        cutoff = np.sqrt(chi2.ppf(self.quantile, 1))

        if plottype == "indexplot":
            x, y = np.arange(self.W.shape[0]), self.residuals[:, variable]
            h_thresholds, v_thresholds = (-cutoff, cutoff), None
            xlabel, ylabel = "index", f"standardized residuals of {variable_name}"
            title = "Indexplot: standardized residuals"

        elif plottype == "residuals_vs_variable":
            x, y = self.X[:, variable], self.residuals[:, variable]
            h_thresholds, v_thresholds = (-cutoff, cutoff), get_thresholds(cutoff, x)
            xlabel, ylabel = variable_name, f"standardized residuals of {variable_name}"
            title = f"Standardized residuals versus the {variable_name}"

        elif plottype == "residuals_vs_predictions":
            x, y = self.predictions[:, variable], self.residuals[:, variable]
            h_thresholds, v_thresholds = (-cutoff, cutoff), get_thresholds(cutoff, x)
            xlabel, ylabel = (
                f"predictions of {variable_name}",
                f"standardized residuals of {variable_name}",
            )
            title = f"Standardized residuals versus the predictions of the {variable_name}"

        elif plottype == "variable_vs_predictions":
            x, y = self.predictions[:, variable], self.X[:, variable]
            h_thresholds, v_thresholds = get_thresholds(cutoff, y), get_thresholds(cutoff, x)
            xlabel, ylabel = f"predictions of {variable_name}", f"observed {variable_name}"
            ax.axline((x[0], x[0]), slope=1, color="grey", linestyle="-.")
            title = f"{variable_name} versus its predictions"

        elif plottype == "bivariate":
            if second_variable is None:
                raise ValueError("second_variable must be provided for bivariate plot.")
            x, y = self.X[:, variable], self.X[:, second_variable]
            h_thresholds, v_thresholds = get_thresholds(cutoff, y), get_thresholds(cutoff, x)
            xlabel, ylabel = variable_name, second_variable_name
            draw_ellipse(
                self.covariance_[np.ix_([variable, second_variable], [variable, second_variable])],
                self.location_[[variable, second_variable]],
                ax,
                self.quantile,
            )
            title = f"{variable_name} versus {second_variable_name}"

        ax.scatter(x=x, y=y)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

        if row_names is not None:
            if plottype != "bivariate":
                annote_outliers(ax, row_names, x, y, h_thresholds, v_thresholds)
            else:
                annote_outliers_ellipse(
                    ax,
                    row_names,
                    self.location_,
                    self.covariance_,
                    variable,
                    second_variable,
                    x,
                    y,
                    self.quantile,
                )

        draw_threshold_lines(ax, h_thresholds, v_thresholds)

        return fig

    def _make_predictions(
        self, X: np.ndarray, sigma: np.ndarray, sigma_inv: np.ndarray, mu: np.ndarray, W: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the predictions of the cells given the other clean cells in the same row"""

        n, p = X.shape
        predictions = np.zeros([n, p])
        conditional_variances = np.zeros([n, p])
        unique_rows = np.unique(W, axis=0)

        for w in unique_rows:  # iterate over all w patterns
            row_idx_w = np.where(np.all(W == w, axis=1))[0]  # rows with the corresponding w pattern
            missing_col_idx = np.where(w == 0)[0]
            observed_col_idx = np.where(w == 1)[0]

            if len(missing_col_idx) < p and len(missing_col_idx) > 0:
                # there are both observed and missing values

                # predict the missing values
                predictions, conditional_variances = self._predict_missing_values(
                    X,
                    predictions,
                    conditional_variances,
                    row_idx_w,
                    missing_col_idx,
                    observed_col_idx,
                    mu,
                    sigma,
                    sigma_inv,
                )

                # predict the observed values
                predictions, conditional_variances = self._predict_observed_values(
                    X,
                    predictions,
                    conditional_variances,
                    row_idx_w,
                    observed_col_idx,
                    mu,
                    sigma,
                    sigma_inv,
                )

            elif len(missing_col_idx) == 0:  # no missings in the w-pattern
                predictions, conditional_variances = self._predict_no_missing(
                    X, predictions, conditional_variances, row_idx_w, mu, sigma, sigma_inv
                )

            else:  # all missings in the w-pattern
                predictions, conditional_variances = self._predict_all_missing(
                    predictions, conditional_variances, row_idx_w, mu, sigma_inv
                )

        return predictions, conditional_variances

    def _C_steps_until_convergence(
        self,
        X_scaled: np.ndarray,
        initial_location: np.array,
        initial_cov: np.ndarray,
        initial_cov_inv: np.ndarray,
    ):
        """Perform C-steps (update W, location and covariance) until convergence.

        Arguments:
        - X_scaled (np.ndarray): scaled data
        - initial_location (np.array): initial location estimate
        - initial_cov (np.ndarray): initial covariance estimate
        - initial_cov_inv (np.ndarray): inverse of the initial covariance estimate
        """
        n, p = X_scaled.shape

        h = round(self.alpha * n)
        initial_W = np.ones([n, p])
        initial_W[np.isnan(X_scaled)] = 0
        objective_values = np.full(self.max_c_steps + 1, np.nan)
        step = 0
        q = (
            chi2.ppf(0.99, 1) - np.log(np.diag(initial_cov_inv)) + np.log(2 * np.pi)
        )  # equation (21)
        penalty = np.sum(q * np.sum(1 - initial_W, axis=0))
        objective_values[step] = (
            objective_function(X_scaled, initial_W, initial_location, initial_cov, initial_cov_inv)
            + penalty
        )
        convergence_criteria = 1
        W_old = initial_W
        location_old = initial_location
        cov_old = initial_cov
        cov_inv_old = initial_cov_inv
        while convergence_criteria > self.crit and step < self.max_c_steps:
            W, location, cov = self._C_step(
                X_scaled, W_old, location_old, cov_old, cov_inv_old, q, h
            )
            convergence_criteria = np.max(np.abs(cov - cov_old))
            cov = truncated_covariance(cov, self.min_eigenvalue)
            cov_inv = np.linalg.inv(cov)
            penalty = np.sum(q * np.sum(1 - W, axis=0))
            objective_value = objective_function(X_scaled, W, location, cov, cov_inv) + penalty
            if objective_value > objective_values[step]:
                return location_old, cov_old, cov_inv_old, W_old
            step = step + 1
            objective_values[step] = objective_value
            W_old = W
            location_old = location
            cov_old = cov
            cov_inv_old = cov_inv

        return location, cov, cov_inv, W

    def _C_step(
        self,
        X: np.ndarray,
        W: np.ndarray,
        mu: np.array,
        sigma: np.ndarray,
        sigma_inv: np.ndarray,
        q: np.array,
        h: int,
    ):
        n, p = X.shape

        # first update W
        W = self._update_W(X, W, mu, sigma, sigma_inv, q, h)

        # next: update mu & sigma
        X_imputed = np.copy(X)
        bias = np.zeros([p, p])
        for i in range(n):
            missing_col_idx = np.where(W[i, :] == 0)[0]
            observed_col_idx = np.where(W[i, :] == 1)[0]
            if len(missing_col_idx) > 0:
                if len(missing_col_idx) == p:  # if all are missing
                    X_imputed[i, :] = mu
                    bias = bias + sigma
                else:
                    sigma_inv_mis_inv = np.linalg.inv(
                        sigma_inv[np.ix_(missing_col_idx, missing_col_idx)]
                    )
                    X_imputed[np.ix_([i], missing_col_idx)] = (
                        mu[missing_col_idx]
                        - (
                            sigma_inv_mis_inv
                            @ sigma_inv[np.ix_(missing_col_idx, observed_col_idx)]
                            @ (X[np.ix_([i], observed_col_idx)] - mu[observed_col_idx]).T
                        ).flatten()
                    )
                    bias[np.ix_(missing_col_idx, missing_col_idx)] = (
                        bias[np.ix_(missing_col_idx, missing_col_idx)] + sigma_inv_mis_inv
                    )
        mu = np.mean(X_imputed, axis=0)
        bias = bias / n
        sigma = np.cov(X_imputed, rowvar=False) * (n - 1) / n + bias

        return W, mu, sigma

    def _update_W(
        self,
        X: np.ndarray,
        W: np.ndarray,
        mu: np.array,
        sigma: np.ndarray,
        sigma_inv: np.ndarray,
        q: np.array,
        h: int,
    ) -> np.ndarray:
        """Part (a) of the C-step: updating the matrix W while keeping mu and sigma unchanged.
        This is done per column."""

        n = W.shape[0]
        ordering = np.argsort(np.sum(W, axis=0), kind="stable")
        for j in ordering:
            delta = self._calculate_delta(X, W, sigma, sigma_inv, mu, j)
            good_cells = np.where(delta <= q[j])[0]
            if len(good_cells) < h:  # cannot set less than h cells to 1
                delta_rank = np.argsort(delta, kind="stable")
                good_cells = delta_rank[:h]
            w_new = np.zeros(n)
            w_new[good_cells] = 1
            W[:, j] = w_new

        return W

    def _calculate_delta(
        self,
        X: np.ndarray,
        W: np.ndarray,
        sigma: np.ndarray,
        sigma_inv: np.ndarray,
        mu: np.ndarray,
        j,
    ):
        """Calculates the  delta's from equation (20) for the column j (without q_j)"""

        n = X.shape[0]
        delta = np.full(n, np.inf)
        unique_rows = np.unique(W, axis=0)
        for w in unique_rows:
            x = X[:, j]
            row_idx_w = np.where(np.all(W == w, axis=1))[0]  # rows with current w pattern
            finite_rows = row_idx_w[np.isfinite(x[row_idx_w])]  # row_idx_w with finite/not-nan x_j
            if len(finite_rows) > 0:  # if there are rows that aren't nans or infs
                x = x[finite_rows]  # the relevant entries
                w1 = w.copy()  # current W pattern
                w0 = w.copy()
                w0[j] = 0  # current W pattern, but W_ij is set to zero
                if np.any(w0):  # current W pattern contains nonzeros (without looking at j)
                    w1[j] = 1  # current W pattern, but W_ij is set to one
                    index_0 = np.nonzero(w0)[0]  # ones in current W pattern (without j)
                    index_1 = np.nonzero(w1)[0]
                    mu_0 = mu[index_0]
                    mu_1 = mu[index_1]
                    sigma_1 = sigma[np.ix_(index_1, index_1)]
                    sigma_inv_0 = inverse_submatrix(sigma, sigma_inv, index_0)

                    jtemp = np.where(index_1 == j)[0]  # current j in index 1
                    jtemp_n = np.where(index_1 != j)[0]  # ones in index 1 (without j)
                    Xtemp = X[np.ix_(finite_rows, index_0)] - mu_0
                    x_hat = mu_1[jtemp] + (Xtemp @ sigma_inv_0 @ sigma_1[np.ix_(jtemp_n, jtemp)]).T

                    c = (
                        sigma_1[np.ix_(jtemp, jtemp)]
                        - sigma_1[np.ix_(jtemp, jtemp_n)]
                        @ sigma_inv_0
                        @ sigma_1[np.ix_(jtemp_n, jtemp)]
                    )

                    delta[finite_rows] = (x - x_hat) ** 2 / c + np.log(c) + np.log(2 * np.pi)

                else:  # current W pattern has zeros everywhere (except on j)
                    delta[finite_rows] = (
                        (x - mu[j]) ** 2 / sigma[j, j] + np.log(sigma[j, j]) + np.log(2 * np.pi)
                    )

        return delta

    def _predict_missing_values(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        conditional_variances: np.ndarray,
        row_idx_w: np.array,
        missing_col_idx: np.array,
        observed_col_idx: np.array,
        mu: np.array,
        sigma: np.ndarray,
        sigma_inv: np.ndarray,
    ):
        """Predict the missing values when there are both observed and missing values."""

        sigma_observed_inv = inverse_submatrix(sigma, sigma_inv, observed_col_idx)
        conditional_variance_missing = np.diag(
            sigma[np.ix_(missing_col_idx, missing_col_idx)]
            - sigma[np.ix_(missing_col_idx, observed_col_idx)]
            @ sigma_observed_inv
            @ sigma[np.ix_(observed_col_idx, missing_col_idx)]
        )

        # predict the missing values
        for index in row_idx_w:
            conditional_variances[np.ix_([index], missing_col_idx)] = conditional_variance_missing
            predictions[np.ix_([index], missing_col_idx)] = (
                mu[missing_col_idx]
                + (X[np.ix_([index], observed_col_idx)] - mu[observed_col_idx])
                @ sigma_observed_inv
                @ sigma[np.ix_(observed_col_idx, missing_col_idx)]
            )

        return predictions, conditional_variances

    def _predict_observed_values(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        conditional_variances: np.ndarray,
        row_idx_w: np.array,
        observed_col_idx: np.array,
        mu: np.array,
        sigma: np.ndarray,
        sigma_inv: np.ndarray,
    ):
        """Predict the observed values when there are both observed and missing values"""

        if len(observed_col_idx) == 1:  # only 1 observed value
            for index in row_idx_w:
                predictions[np.ix_([index], observed_col_idx)] = mu[observed_col_idx]
                conditional_variances[np.ix_([index], observed_col_idx)] = sigma[
                    observed_col_idx, observed_col_idx
                ]
        else:  # multiple observed values
            for obs in observed_col_idx:
                other_observations = observed_col_idx[observed_col_idx != obs]
                sigma_others_inv = inverse_submatrix(sigma, sigma_inv, other_observations)
                conditional_variance_others = (
                    sigma[obs, obs]
                    - sigma[np.ix_([obs], other_observations)]
                    @ sigma_others_inv
                    @ sigma[np.ix_(other_observations, [obs])]
                )

                for index in row_idx_w:
                    predictions[index, obs] = (
                        mu[obs]
                        + (X[np.ix_([index], other_observations)] - mu[other_observations])
                        @ sigma_others_inv
                        @ sigma[np.ix_(other_observations, [obs])]
                    )
                    conditional_variances[index, obs] = conditional_variance_others

        return predictions, conditional_variances

    def _predict_no_missing(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        conditional_variances: np.ndarray,
        row_idx_w: np.array,
        mu: np.array,
        sigma: np.ndarray,
        sigma_inv: np.ndarray,
    ):
        """Make predictions when no values in the w-pattern are missing."""

        p = X.shape[1]
        conditional_variance_nomissings = 1.0 / np.diag(sigma_inv)
        for index in row_idx_w:
            conditional_variances[index, :] = conditional_variance_nomissings
        for j in range(p):
            j_neg = np.arange(p)
            j_neg = j_neg[j_neg != j]
            sigma_jneg_inv = inverse_submatrix(sigma, sigma_inv, j_neg)
            Xtemp = X[np.ix_(row_idx_w, j_neg)] - mu[j_neg]
            predictions[np.ix_(row_idx_w, [j])] = (
                mu[j] + Xtemp @ sigma_jneg_inv @ sigma[np.ix_(j_neg, [j])]
            )

        return predictions, conditional_variances

    def _predict_all_missing(
        self,
        predictions: np.ndarray,
        conditional_variances: np.ndarray,
        row_idx_w: np.array,
        mu: np.array,
        sigma_inv: np.ndarray,
    ):
        """Make predictions when all values in the w-pattern are missing."""
        for index in row_idx_w:
            predictions[index, :] = mu
            conditional_variances[index, :] = np.diag(sigma_inv)

        return predictions, conditional_variances

    def _check_data(self, X_scaled: np.ndarray):
        """Checks that there aren't too many nans or marginal outliers per column. Checks
        that there are enough observations.

        Arguments:
            X_scaled (np.ndarray): robustly standardized data set."""

        X = np.copy(X_scaled)

        cutoff = np.sqrt(chi2.ppf(self.quantile, 1))
        n_marginal_outliers = np.sum(np.abs(X) > cutoff, axis=0) / X.shape[0]
        if np.max(n_marginal_outliers) > 1 - self.alpha:
            raise ValueError(
                f"\nAt least one variable has more than {100 * (1 - self.alpha)}% of "
                "marginal  outliers."
            )

        X[np.abs(X) > cutoff] = np.nan
        n_bad_values = np.isnan(X).sum(axis=0) / X.shape[0]
        if np.max(n_bad_values) > 1 - self.alpha:
            raise ValueError(
                f"\nAt least one variable has more than {100 * (1 - self.alpha)}% of nan's or "
                "marginal outliers."
            )

        if X.shape[0] < 5 * X.shape[1]:
            raise ValueError(
                "There are not enough observations compared to the number of "
                f"variables, n/p = {X.shape[0]/X.shape[1]} < 5."
            )
