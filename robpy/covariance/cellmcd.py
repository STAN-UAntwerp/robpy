import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2, median_abs_deviation
from typing import Literal
from robpy.covariance.base import RobustCovarianceEstimator
from robpy.utils.distance import mahalanobis_distance
from robpy.utils.logging import get_logger
from robpy.preprocessing.scaling import RobustScaler
from robpy.univariate.onestep_m import OneStepWrappingEstimator
from robpy.outliers.ddc import DDCEstimator
from robpy.covariance.wrapping import WrappingCovarianceEstimator
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
        """Cell MCD estimator based on the algorithm proposed in
            Raymaekers and Rousseeuw, The Cellwise Minimum Covariance Determinant Estimator, 2023,
            Journal of the American Statistical Association.

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
        """
        self.alpha = alpha
        self.quantile = quantile
        self.crit = crit
        self.max_c_steps = max_c_steps
        self.min_eigenvalue = min_eigenvalue
        self.logger = get_logger("CellMCDEstimator", level=verbosity)
        self.verbosity = verbosity

    def calculate_covariance(self, X: np.ndarray) -> np.ndarray:

        n, p = X.shape
        if self.min_eigenvalue < 1e-6:
            raise ValueError("The lower bound on the eigenvalues should be at least 1e-6.")

        # Step 0: robustly standardize the data
        mads = np.apply_along_axis(
            lambda x: median_abs_deviation(x, nan_policy="omit"), axis=0, arr=X
        )
        if np.min(mads) < 1e-8:
            raise ValueError("At least one variable has an almost zero median absolute deviation.")
        scaler = RobustScaler(scale_estimator=OneStepWrappingEstimator(omit_nans=True))
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        # Step 1: Check that there aren't too many marginal outliers and too many nan's.
        #         Check that there are enough observations compared to variables.
        self._check_data(np.copy(X_scaled))

        # Step 2: calculate the initial estimates
        initial_location, initial_cov = self._DDCW(X_scaled)
        initial_cov_inv = np.linalg.inv(initial_cov)
        q = (
            chi2.ppf(0.99, 1) - np.log(np.diag(initial_cov_inv)) + np.log(2 * np.pi)
        )  # equation (21)
        X_scaled[np.abs(X_scaled) > 3] = np.nan

        # Step 3: MCD iterations (we work with X_scaled)
        h = round(self.alpha * n)
        initial_W = np.ones([n, p])
        initial_W[np.isnan(X_scaled)] = 0
        objective_values = np.full(self.max_c_steps + 1, np.nan)
        step = 0
        penalty = np.sum(q * np.sum(1 - initial_W, axis=0))
        objective_values[step] = (
            self._objective_function(
                X_scaled, initial_W, initial_location, initial_cov, initial_cov_inv
            )
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
            cov = self._truncated_covariance(cov)
            cov_inv = np.linalg.inv(cov)
            penalty = np.sum(q * np.sum(1 - W, axis=0))
            objective_value = (
                self._objective_function(X_scaled, W, location, cov, cov_inv) + penalty
            )
            if objective_value > objective_values[step]:
                W = W_old
                location = location_old
                cov = cov_old
                cov_inv = cov_inv_old
                break
            step = step + 1
            objective_values[step] = objective_value
            W_old = W
            location_old = location
            cov_old = cov
            cov_inv_old = cov_inv

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
        scaler_residuals = RobustScaler(scale_estimator=OneStepWrappingEstimator(omit_nans=True))
        scaler_residuals.fit(residuals)
        residuals = residuals / scaler_residuals.scales_

        self.W = W
        self.location_ = location
        self.predictions = predictions
        self.conditional_stds = conditional_stds
        self.X_imputed = X_imputed
        self.residuals = residuals
        self.steps = step
        self.covariance_ = cov
        self.X = X

        return cov

    def cell_MCD_plot(
        self,
        variable: int,
        variable_name: str = "variable",
        row_names: list = None,
        second_variable: int = None,
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
        """Function to plot the results of a cellMCD analysis: 5 types of diagnostic plots.

        Arguments:
        plottype (Literal string, optional):
            - "indexplot": plots the residuals of a variable
            - "residuals_vs_variable": plots a variable versus its residuals
            - "residuals_vs_predictions": plots the predictions of a variable versus its residuals
            - "variable_vs_predictions": plots a variable against its predictions
            - "bivariate": plots two variables against each other
            Defaults to "indexplot".
        variable (int):
            Index of the variable under consideration.
        variable_name (str, optional):
            Name of the variable of interest for the axis label.
            Defaults to "variable".
        second_variable (int):
            Index of the second variable under consideration, only needed for plottype "bivariate".
        second_variable_name (str, optional):
            Name of the second variable for the axis label, only relevant for plottype "bivariate".
            Defaults to "second variable".
        row_names (list of strings, optional):
            Row_names of the observations if you want the outliers annoted with their name.
        figsize (tuple[int,int], optional):
            Size of the figure.
            Defaults to (8,8).
        """

        if not hasattr(self, "covariance_"):
            raise NotFittedError()

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        cutoff = np.sqrt(chi2.ppf(self.quantile, 1))

        if plottype == "indexplot":
            x, y = np.arange(self.W.shape[0]), self.residuals[:, variable]
            h_thresholds, v_thresholds = [-cutoff, cutoff], None
            xlabel, ylabel = "index", f"standardized residuals of {variable_name}"
            title = "Indexplot: standardized residuals"

        elif plottype == "residuals_vs_variable":
            x, y = self.X[:, variable], self.residuals[:, variable]
            h_thresholds, v_thresholds = self._get_thresholds(cutoff, x)
            xlabel, ylabel = variable_name, f"standardized residuals of {variable_name}"
            title = f"Standardized residuals versus the {variable_name}"

        elif plottype == "residuals_vs_predictions":
            x, y = self.predictions[:, variable], self.residuals[:, variable]
            h_thresholds, v_thresholds = self._get_thresholds(cutoff, x)
            xlabel, ylabel = (
                f"predictions of {variable_name}",
                f"standardized residuals of {variable_name}",
            )
            title = f"Standardized residuals versus the predictions of the {variable_name}"

        elif plottype == "variable_vs_predictions":
            x, y = self.predictions[:, variable], self.X[:, variable]
            h_thresholds, v_thresholds = self._get_thresholds(cutoff, x, y)
            xlabel, ylabel = f"predictions of {variable_name}", f"observed {variable_name}"
            ax.axline((x[0], x[0]), slope=1, color="grey", linestyle="-.")
            title = f"{variable_name} versus its predictions"

        elif plottype == "bivariate":
            if second_variable is None:
                raise ValueError("second_variable must be provided for bivariate plot.")
            x, y = self.X[:, variable], self.X[:, second_variable]
            h_thresholds, v_thresholds = self._get_thresholds(cutoff, x, y)
            xlabel, ylabel = variable_name, second_variable_name
            self._draw_ellipse(
                self.covariance_[np.ix_([variable, second_variable], [variable, second_variable])],
                self.location_[[variable, second_variable]],
                ax,
            )
            title = f"{variable_name} versus {second_variable_name}"

        ax.scatter(x=x, y=y)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)

        if row_names:
            if plottype != "bivariate":
                self._annote_outliers(ax, row_names, x, y, h_thresholds, v_thresholds)
            else:
                self._annote_outliers_ellipse(ax, row_names, variable, second_variable, x, y)

        self._draw_threshold_lines(ax, h_thresholds, v_thresholds)

        return fig

    def _annote_outliers(
        self,
        ax,
        row_names,
        x: np.array,
        y: np.array,
        h_thresholds: list[float, float],
        v_thresholds: list[float, float] = None,
    ):
        """Label outliers with their name if they exceed a threshold"""
        if h_thresholds is None and v_thresholds is not None:
            for i, (xi, yi) in enumerate(zip(x, y)):
                if (
                    yi < h_thresholds[0]
                    or yi > h_thresholds[1]
                    or xi < v_thresholds[0]
                    or xi > v_thresholds[1]
                ):
                    ax.text(xi, yi, row_names[i], fontsize=12, ha="center", va="bottom")
        else:
            for i, (xi, yi) in enumerate(zip(x, y)):
                if yi < h_thresholds[0] or yi > h_thresholds[1]:
                    ax.text(xi, yi, row_names[i], fontsize=9, ha="center", va="bottom")

    def _annote_outliers_ellipse(
        self,
        ax,
        row_names,
        variable,
        second_variable,
        x: np.array,
        y: np.array,
    ):
        """Label outliers with their name if they are outside the tolerance ellipse"""
        mask = mahalanobis_distance(
            np.column_stack((x, y)),
            self.location_[[variable, second_variable]],
            covariance=self.covariance_[
                np.ix_([variable, second_variable], [variable, second_variable])
            ],
        ) > np.sqrt(chi2.ppf(self.quantile, 2))
        for xi, yi, name in zip(x[mask], y[mask], [row_names[i] for i in np.where(mask)[0]]):
            ax.text(xi, yi, name, fontsize=9, ha="center", va="bottom")

    def _draw_ellipse(self, cov: np.ndarray, center: np.array, ax):
        """Get the ellipse for bivariate data given the covariance matrix (for the shape) and the
        location (for the center)."""

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        shape = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T  # orthogonalize
        angles = np.linspace(0, 2 * np.pi, 200 + 1)
        xy = np.column_stack((np.cos(angles), np.sin(angles)))
        radius = np.sqrt(chi2.ppf(self.quantile, 2))
        ellipse = radius * xy @ shape + center
        ax.plot(ellipse[:, 0], ellipse[:, 1], linewidth=3, color="darkgray")

    def _get_thresholds(self, cutoff: float, x: np.array, y: np.array = None):
        scaler = OneStepWrappingEstimator(omit_nans=True).fit(x)
        threshold_x_plus = scaler.location_ + cutoff * scaler.scale
        threshold_x_min = scaler.location_ - cutoff * scaler.scale
        if y is not None:
            scaler = OneStepWrappingEstimator(omit_nans=True).fit(y)
            threshold_y_plus = scaler.location + cutoff * scaler.scale
            threshold_y_min = scaler.location - cutoff * scaler.scale
            return [threshold_y_min, threshold_y_plus], [threshold_x_min, threshold_x_plus]
        else:
            return [-cutoff, cutoff], [threshold_x_min, threshold_x_plus]

    def _draw_threshold_lines(
        self,
        ax,
        h_thresholds: list[float, float],
        v_thresholds: list[float, float] | None,
    ):

        for h in h_thresholds:
            ax.axhline(h, color="grey", linestyle="--")
        if v_thresholds is not None:
            for v in v_thresholds:
                ax.axvline(v, color="grey", linestyle="--")

    def _make_predictions(
        self, X: np.ndarray, sigma: np.ndarray, sigma_inv: np.ndarray, mu: np.array, W: np.ndarray
    ):
        """Calculate the predictions of the cells given the other clean cells in the same row"""

        n, p = X.shape
        predictions = np.zeros([n, p])
        conditional_variances = np.zeros([n, p])
        unique_rows = np.unique(W, axis=0)

        for w in unique_rows:  # iterate over all w patterns
            w_rows = np.where(np.all(W == w, axis=1))[0]  # rows with the corresponding w pattern
            missing = np.where(w == 0)[0]
            observed = np.where(w == 1)[0]

            if len(missing) < p:
                if len(missing) > 0:
                    sigma_observed_inv = self._inverse_submatrix(sigma, sigma_inv, observed)
                    conditional_variance_missing = np.diag(
                        sigma[np.ix_(missing, missing)]
                        - sigma[np.ix_(missing, observed)]
                        @ sigma_observed_inv
                        @ sigma[np.ix_(observed, missing)]
                    )

                    # predict the missing values
                    for index in w_rows:
                        conditional_variances[np.ix_([index], missing)] = (
                            conditional_variance_missing
                        )
                        predictions[np.ix_([index], missing)] = (
                            mu[missing]
                            + (X[np.ix_([index], observed)] - mu[observed])
                            @ sigma_observed_inv
                            @ sigma[np.ix_(observed, missing)]
                        )

                    # predict the observed values
                    if len(observed) == 1:  # only 1 observed value
                        for index in w_rows:
                            predictions[np.ix_([index], observed)] = mu[observed]
                            conditional_variances[np.ix_([index], observed)] = sigma[
                                observed, observed
                            ]
                    else:  # multiple observed values
                        for obs in observed:
                            other_observations = observed[observed != obs]
                            sigma_others_inv = self._inverse_submatrix(
                                sigma, sigma_inv, other_observations
                            )
                            conditional_variance_others = (
                                sigma[obs, obs]
                                - sigma[np.ix_([obs], other_observations)]
                                @ sigma_others_inv
                                @ sigma[np.ix_(other_observations, [obs])]
                            )

                            for index in w_rows:
                                predictions[index, obs] = (
                                    mu[obs]
                                    + (
                                        X[np.ix_([index], other_observations)]
                                        - mu[other_observations]
                                    )
                                    @ sigma_others_inv
                                    @ sigma[np.ix_(other_observations, [obs])]
                                )
                                conditional_variances[index, obs] = conditional_variance_others

                else:  # no missings in the pattern
                    conditional_variance_nomissings = 1.0 / np.diag(sigma_inv)
                    for index in w_rows:
                        conditional_variances[index, :] = conditional_variance_nomissings
                    for j in range(p):
                        j_neg = np.arange(p)
                        j_neg = j_neg[j_neg != j]
                        sigma_jneg_inv = self._inverse_submatrix(sigma, sigma_inv, j_neg)
                        Xtemp = X[np.ix_(w_rows, j_neg)] - mu[j_neg]
                        predictions[np.ix_(w_rows, [j])] = (
                            mu[j] + Xtemp @ sigma_jneg_inv @ sigma[np.ix_(j_neg, [j])]
                        )
            else:  # all missings in the pattern
                for index in w_rows:
                    predictions[index, :] = mu
                    conditional_variances[index, :] = np.diag(sigma_inv)

        return predictions, conditional_variances

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
            missing = np.where(W[i, :] == 0)[0]
            observed = np.where(W[i, :] == 1)[0]
            if len(missing) > 0:
                if len(missing) == p:  # if all are missing
                    X_imputed[i, :] = mu
                    bias = bias + sigma
                else:
                    sigma_inv_mis_inv = np.linalg.inv(sigma_inv[np.ix_(missing, missing)])
                    X_imputed[np.ix_([i], missing)] = (
                        mu[missing]
                        - (
                            sigma_inv_mis_inv
                            @ sigma_inv[np.ix_(missing, observed)]
                            @ (X[np.ix_([i], observed)] - mu[observed]).T
                        ).flatten()
                    )
                    bias[np.ix_(missing, missing)] = (
                        bias[np.ix_(missing, missing)] + sigma_inv_mis_inv
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
            w_rows = np.where(np.all(W == w, axis=1))[0]  # rows with current w pattern
            finite_rows = w_rows[np.isfinite(x[w_rows])]  # w_rows with finite/not-nan x_j
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
                    sigma_inv_0 = self._inverse_submatrix(sigma, sigma_inv, index_0)

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

    def _objective_function(
        self, X: np.ndarray, W: np.ndarray, location: np.array, cov: np.ndarray, cov_inv: np.ndarray
    ) -> float:
        """Calculates the value of the objective function in equation (9) without the penalty
        for a certain X, W, location, covariance and the inverse covariance"""

        objective = 0
        unique_rows = np.unique(W, axis=0)
        for w in unique_rows:
            w_rows = np.where(np.all(W == w, axis=1))[0]
            w_ones = np.where(w)[0]

            subset_location = np.array(location)[w_ones]
            subset_cov = cov[np.ix_(w_ones, w_ones)]
            subset_cov_inv = self._inverse_submatrix(cov, cov_inv, w_ones)

            subset_X = X[w_rows][:, w_ones] - subset_location
            partial_MD = np.sum(subset_X.dot(subset_cov_inv) * subset_X, axis=1)
            objective = objective + np.sum(
                partial_MD + np.linalg.slogdet(subset_cov)[1] + np.log(2 * np.pi) * len(w_ones)
            )
        return objective

    def _inverse_submatrix(self, A: np.ndarray, A_inv: np.ndarray, indices: np.array) -> np.ndarray:
        """Given a matrix A and its inverse A_inv, this function calculates the inverse
        of the submatrix of A consisting of the rows and columns in indices."""

        p = A.shape[1]
        n_submatrix = len(indices)
        indices_neg = np.setdiff1d(np.arange(p), indices)
        result = np.zeros([n_submatrix, n_submatrix])

        if n_submatrix < p and n_submatrix > p / 2.0:  # in this scenario it useful to use the trick
            result = (
                A_inv[np.ix_(indices, indices)]
                - A_inv[np.ix_(indices, indices_neg)]
                @ np.linalg.inv(A_inv[np.ix_(indices_neg, indices_neg)])
                @ A_inv[np.ix_(indices_neg, indices)]
            )
        elif n_submatrix < p and n_submatrix <= p / 2.0:  # don't use the trick
            result = np.linalg.inv(A[np.ix_(indices, indices)])
        else:  # submatrix is the original matrix
            result = A_inv

        return result

    def _DDCW(self, X: np.ndarray):
        """Calculates the initial cellwise robust estimates of location and scatter using an
        adaptation of DDC.

        Arguments:
            X (np.ndarray): scaled data set

        [based on cellWise:::DDCWcov]"""

        n, p = X.shape

        # DDC with constraint -> imputed and rescaled Zimp:
        DDC = DDCEstimator(chi2_quantile=0.9, scale_estimator=OneStepWrappingEstimator()).fit(
            pd.DataFrame(X)
        )
        W = np.copy(DDC.cellwise_outliers_)
        flagged_too_many = np.where(np.sum(W, axis=0) / X.shape[0] > 1 - self.alpha)[0]
        for i in flagged_too_many:
            ordering = np.argsort(np.abs(DDC.standardized_residuals_[:, i]))[::-1]
            W[:, i] = [False for _ in range(n)]
            W[ordering[0 : int(n * 0.25)], i] = True
        Zimp = np.copy(X)
        Zimp[np.logical_or(W, np.isnan(X))] = DDC.impute(pd.DataFrame(X), impute_outliers=True)[
            np.logical_or(W, np.isnan(X))
        ]
        Z = (X - DDC.location_) / DDC.scale_
        Zimp = (Zimp - DDC.location_) / DDC.scale_
        Zimp_original = np.copy(Zimp)
        Zimp = np.delete(Zimp, np.where(DDC.row_outliers_)[0], axis=0)

        # project data on eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(Zimp, rowvar=False))
        eigenvectors = eigvecs[:, np.where(eigvals > self.min_eigenvalue)[0][::-1]]
        Zimp_proj = Zimp @ eigenvectors

        # wrapped location and covariance
        Zimp_proj_scaler = RobustScaler(
            scale_estimator=OneStepWrappingEstimator(omit_nans=True)
        ).fit(Zimp_proj)
        Zimp_proj_scaler.scales_[Zimp_proj_scaler.scales_ < self.min_eigenvalue] = (
            self.min_eigenvalue
        )
        Zimp_proj_wrapped_cov = WrappingCovarianceEstimator(
            locations=Zimp_proj_scaler.locations_, scales=Zimp_proj_scaler.scales_, rescale=True
        ).fit(Zimp_proj)
        cov = (
            eigenvectors @ Zimp_proj_wrapped_cov.covariance_ @ eigenvectors.T
        )  # back to original axis system
        cov = self._truncated_covariance(cov)
        cov = (self._covariance_to_correlation(cov) * DDC.scale_).T * DDC.scale_

        # temporary points: delete casewise outliers
        U = np.minimum(np.maximum(Z, -2), 2)
        RD = mahalanobis_distance(U, np.zeros(p), cov) ** 2
        U_outlying_cases = np.where(RD / np.median(RD) * chi2.ppf(0.5, p) > chi2.ppf(0.99, p))[0]
        Z = np.delete(Z, U_outlying_cases, axis=0)
        Zimp = np.delete(Zimp_original, U_outlying_cases, axis=0)

        # project data on eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(Zimp, rowvar=False))
        eigenvectors = eigvecs[:, np.where(eigvals > self.min_eigenvalue)[0][::-1]]
        Zimp_proj = Zimp @ eigenvectors

        # wrapped location and covariance
        Zimp_proj_scaler = RobustScaler(
            scale_estimator=OneStepWrappingEstimator(omit_nans=True)
        ).fit(Zimp_proj)
        Zimp_proj_scaler.scales_[Zimp_proj_scaler.scales_ < self.min_eigenvalue] = (
            self.min_eigenvalue
        )
        Zimp_proj_wrapped_cov = WrappingCovarianceEstimator(
            locations=Zimp_proj_scaler.locations_, scales=Zimp_proj_scaler.scales_, rescale=True
        ).fit(Zimp_proj)
        cov = (
            eigenvectors @ Zimp_proj_wrapped_cov.covariance_ @ eigenvectors.T
        )  # back to original axis system
        cov = self._covariance_to_correlation(cov)
        cov = (self._truncated_covariance(cov) * DDC.scale_).T * DDC.scale_

        location = np.array(DDC.location_)

        return location, cov

    def _truncated_covariance(self, cov: np.ndarray) -> np.ndarray:
        """Modifies the covariance such that all eigenvalues are at least as large as the
        given minimum."""
        eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
        eigvals_idx = np.argsort(eigvals)[::-1]
        eigvals[eigvals < self.min_eigenvalue] = self.min_eigenvalue
        return eigvecs[:, eigvals_idx] @ np.diag(eigvals[eigvals_idx]) @ eigvecs[:, eigvals_idx].T

    def _covariance_to_correlation(self, cov: np.ndarray) -> np.ndarray:
        """Converts a covariance matrix to a correlation matrix"""
        stds = np.sqrt(np.diag(cov))
        return cov / np.outer(stds, stds)

    def _check_data(self, X: np.ndarray):
        """Checks that there aren't too many nans or marginal outliers per column. Checks
        that there are enough observations.

        Arguments:
            X (np.ndarray): robustly standardized data set."""

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
