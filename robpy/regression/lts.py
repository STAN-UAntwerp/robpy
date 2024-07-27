from __future__ import annotations
import logging

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from scipy.stats import norm

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression

from robpy.regression.base import RobustRegressor, _convert_input_to_array


class FastLTSRegressor(RobustRegressor):
    """
    Implementation of FAST-LTS model based on R implementation of the
    ltsReg method in the robustbase R package
    (cfr. https://www.rdocumentation.org/packages/robustbase/versions/0.93-8/topics/ltsReg)
    and the python implementation `Reweighted-FastLTS`
    (cfr. https://github.com/GiuseppeCannata/Reweighted-FastLTS/blob/master/Reweighted_FastLTS.py)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        n_initial_subsets: int = 500,
        n_initial_c_steps: int = 2,
        n_best_models: int = 10,
        reweighting: bool = True,
        tolerance: float = 1e-15,
        random_state: int = 42,
    ):
        """Initialize a FAST LTS regressor

        Args:
            alpha (float): percentage of data to consider as subset for
                           calculating the trimmed squared error.
                           Must be between 0.5 and 1, with 1 being equal to normal LS regression.
                           Defaults to 0.5.
            n_initial_subset (int): number of initial subsets to apply C-steps on
                                    (cfr `m` in original R implementatino). Defaults to 500.
            n_initial_c_steps (int): number of c-steps to apply on n_initial_subsets
                                     before final c-steps until convergenge . Defaults to 2.
            n_best_models (int): number of best models after initial c-steps to consider
                                 until convergence. Defaults to 10.
            reweighting (bool): Whether to apply reweighting to the raw estimates. Defaults to True.
            tolerance (float): Acceptable delta in loss value between C-steps.
                               If current loss  -  previous loss <= tolerance, model is converged.
                               Defaults to 1e-15.
        """
        self.alpha = alpha
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_c_steps = n_initial_c_steps
        self.n_best_models = n_best_models
        self.reweighting = reweighting
        self.tolerance = tolerance
        self.random_state = random_state
        self.logger = logging.getLogger("FastLTS")

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        initial_weights: np.ndarray | None = None,
        verbosity: int = logging.INFO,
    ) -> FastLTSRegressor:
        """Fit the model to the data

        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            initial_weights (Optional[np.ndarray], optional): Optionally pass fixed initial weights,
                            in case of n_initial_subsets > 1, this means all models start
                            from the same initial weights.
                            There is therefore no benefit from setting n_initial_subsets > 1
                            Defaults to None.
            verbosity (int, optional): [description]. Defaults to logging.INFO.

        Returns:
            The fitted FastLTS object
        """
        self.logger.setLevel(verbosity)

        X, y = _convert_input_to_array(X, y)
        y = y.reshape(-1, 1)
        if self.alpha < 0.5 or self.alpha > 1:
            raise ValueError(f"alpha must be between 0.5 and 1, but received {self.alpha}")
        h = int(X.shape[0] * self.alpha)
        self.logger.info(
            f"Applying {self.n_initial_c_steps} initial c-steps "
            f"on {self.n_initial_subsets} initial subsets"
        )

        lr_models, losses, h_subsets = self._apply_initial_C_steps(
            X, y, h, initial_weights=initial_weights, verbosity=verbosity
        )
        best_model_idxs = np.argsort(losses)[: self.n_best_models]
        best_model, best_loss, best_h_subset = (
            lr_models[best_model_idxs[0]],
            losses[best_model_idxs[0]],
            h_subsets[best_model_idxs[0]],
        )
        self.logger.info(f"Performing final C-steps on {self.n_best_models} best models")
        for model_idx in tqdm(best_model_idxs, disable=verbosity > logging.INFO):
            (
                current_model,
                current_h_subset,
                current_loss,
                _,
            ) = self._apply_C_steps_untill_convergence(
                lr_models[model_idx], losses[model_idx], X, y, h, self.tolerance, self.logger
            )

            if current_loss < best_loss:
                best_loss = current_loss
                best_model = current_model
                best_h_subset = current_h_subset
        self.model = best_model
        self.best_h_subset = best_h_subset
        self._scale = get_correction_factor(p=X.shape[1], n=X.shape[0], alpha=self.alpha) * np.sqrt(
            best_loss
        )
        if self.reweighting:
            residuals = y.reshape(-1, 1) - self.predict(X)
            mask = (np.abs(residuals / np.std(residuals)) <= norm.ppf(0.9875)).flatten()
            self.model = LinearRegression().fit(X[mask, :], y[mask])
            new_subset = np.arange(X.shape[0])[mask]
            best_loss = self._get_loss_value(X, y, new_subset, self.model)
            self._scale = get_correction_factor_reweighting(
                p=X.shape[1], n=X.shape[0], alpha=self.alpha
            ) * np.sqrt(best_loss)

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "model"):
            raise NotFittedError

        X, _ = _convert_input_to_array(X)

        return self.model.predict(X).reshape(-1, 1)

    def _apply_initial_C_steps(
        self,
        X: np.ndarray,
        y: np.ndarray,
        h: int,
        initial_weights: np.ndarray | None,
        verbosity: int = logging.DEBUG,
    ) -> tuple[list[LinearRegression], list[float], list[np.ndarray]]:
        """
        Perform initial c_steps on n_initial_subsets of size n_features + 1

        Returns:
            List of models, List of losses and List of h subsets
        """
        np.random.seed(self.random_state)
        lr_models = []
        losses = []
        h_subsets = []
        for seed, _ in tqdm(
            enumerate(range(self.n_initial_subsets), start=self.random_state),
            disable=verbosity > logging.INFO,
            total=self.n_initial_subsets,
        ):
            lr_model = self._get_initial_model(X, y, seed)
            if initial_weights is not None:
                self.logger.warning(
                    f"Initializing models with fixed weights {initial_weights} "
                    f"instead of random initializations."
                )
                lr_model.intercept_ = initial_weights[[0]]
                lr_model.coef_ = initial_weights[None, 1:]
            h_subset_idx = self._get_h_subset(lr_model, X, y, h)
            for _ in range(self.n_initial_c_steps):
                h_subset_idx, lr_model = self._apply_C_step(lr_model, X, y, h)
            # get final residuals
            losses.append(self._get_loss_value(X, y, h_subset_idx, lr_model))
            lr_models.append(lr_model)
            h_subsets.append(h_subset_idx)
        return lr_models, losses, h_subsets

    @staticmethod
    def _get_initial_model(
        X: np.ndarray,
        y: np.ndarray,
        random_state: int = 42,
    ) -> LinearRegression:
        """Get a Linear Regression model that is fitted on
        a random subset of the data of size n_features + 1

        Args:
            X (np.ndarray): Feature data
            y (np.ndarray): Labels
            random_state (int, optional): Random seed, will determine the random subset.
                Defaults to 42.

        Returns:
            lr_model: A Linear Regression model fitted on a random subset
        """
        n_obs, n_features = X.shape  # n, p
        np.random.seed(random_state)
        subset_idx = np.random.choice(n_obs, n_features + 1, replace=False)
        lr_model = LinearRegression().fit(X[subset_idx], y[subset_idx])
        return lr_model

    @staticmethod
    def _get_loss_value(
        X: np.ndarray,
        y: np.ndarray,
        h_subset_idx: np.ndarray | list[int],
        model: LinearRegression,
    ) -> float:
        """Get the Least Trimmed Squared loss for a specific model and h subset

        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            h_subset_idx (np.ndarray): Indices of h subset
            model (LinearRegression): A trained Linear Regression model

        Returns:
            mean squared residual of h_subset
        """
        y_true = y[h_subset_idx].reshape(-1, 1)
        y_pred = model.predict(X[h_subset_idx]).reshape(-1, 1)
        residuals = y_true - y_pred
        return np.sum(np.power(residuals, 2)) / len(h_subset_idx)

    @staticmethod
    def _apply_C_steps_untill_convergence(
        current_model: LinearRegression,
        previous_loss: float,
        X: np.ndarray,
        y: np.ndarray,
        h: int,
        tolerance: float = 1e-15,
        logger: logging.Logger = logging.getLogger("FastLTS"),
    ) -> tuple[LinearRegression, np.ndarray, float, int]:
        """Apply c-steps until convergence

        Args:
            current_model (LinearRegression): model to start from
            previous_loss (float): reference loss value
            X (np.ndarray): Training data features
            y (np.ndarray): Training data targets
            h (int): Number of samples to consider in subset
            tolerance (float, optional): min delta in loss between iterations. Defaults to 1e-15.
            logger (logging.Logger, optional): logger. Defaults to logging.getLogger('FastLTS').

        Returns:
            Tuple[LinearRegression, np.ndarray, float, int]:
                updated model, h subset indices, final loss value, final iteration idx
        """
        iteration = 0
        while True:
            current_h_subset, current_model = FastLTSRegressor._apply_C_step(current_model, X, y, h)
            current_loss = FastLTSRegressor._get_loss_value(X, y, current_h_subset, current_model)
            logger.debug(
                f"Iteration {iteration}: current loss = {current_loss:.3f}, "
                f"previous loss = {previous_loss:.3f}"
            )
            if (previous_loss - current_loss) <= tolerance:
                break
            previous_loss = current_loss
            iteration += 1
        return current_model, current_h_subset, current_loss, iteration

    @staticmethod
    def _get_h_subset(
        lr_model: LinearRegression, X: np.ndarray, y: np.ndarray, h: int
    ) -> np.ndarray:
        """Get the indices of the h observations with the smallest residuals for a given model

        Args:
            lr_model (LinearRegression): A fitted Linear Regression Model
            X (np.ndarray): Features
            y (np.ndarray): Labels
            h (int): Number of observations to include in the subset

        Returns:
            np.ndarray: Array of indices for the h subset
        """
        residuals = y - lr_model.predict(X).reshape(-1, 1)
        return np.argsort(np.abs(residuals).flatten())[:h]

    @staticmethod
    def _apply_C_step(
        lr_model: LinearRegression, X: np.ndarray, y: np.ndarray, h: int
    ) -> tuple[np.ndarray, LinearRegression]:
        """
        Apply a single C-step

        Returns:
            h subset indices, fitted lr model
        """
        h_subset_idx = FastLTSRegressor._get_h_subset(lr_model, X, y, h)
        lr_model = LinearRegression().fit(X[h_subset_idx], y[h_subset_idx])
        return h_subset_idx, lr_model


def get_correction_factor(p: int, n: int, alpha: float) -> float:
    """
    Calculate the small sample correction factor for the scale resulting from LTS regression.

    References:
        Pison, G., Van Aelst, S. & Willems, G. Small sample corrections for LTS and MCD.
        Metrika 55, 111–123 (2002). https://doi.org/10.1007/s001840200191

        https://github.com/cran/robustbase/blob/c4b9d21cfc4beb64653bb2ffba9e549e2dbb98ed/R/ltsReg.R
    """
    if alpha < 0.5 or alpha > 1:
        raise ValueError(f"alpha must be between 0.5 and 1, but received {alpha}")
    if p == 0:  # intercept only
        fp_500_n = 1 - np.exp(0.262024211897096) / n**0.604756680630497
        fp_875_n = 1 - np.exp(-0.351584646688712) / n**1.01646567502486
        if 0.5 <= alpha <= 0.875:
            fp_alpha_n = fp_500_n + (fp_875_n - fp_500_n) / 0.375 * (alpha - 0.5)
        else:  # 0.875 < alpha < 1
            fp_alpha_n = fp_875_n + (1 - fp_875_n) / 0.125 * (alpha - 0.875)
        fp_alpha_n = np.sqrt(fp_alpha_n)
    else:
        if p == 1:
            fp_500_n = 1 - np.exp(0.630869217886906) / n**0.650789250442946
            fp_875_n = 1 - np.exp(0.565065391014791) / n**1.03044199012509
        else:
            coefgqpkwad875 = np.array(
                [
                    [-0.458580153984614, 1.12236071104403, 3],
                    [-0.267178168108996, 1.1022478781154, 5],
                ]
            )
            coefeqpkwad500 = np.array(
                [
                    [-0.746945886714663, 0.56264937192689, 3],
                    [-0.535478048924724, 0.543323462033445, 5],
                ]
            )
            y_500 = np.log(-coefeqpkwad500[:, 0] / (p ** coefeqpkwad500[:, 1]))
            y_875 = np.log(-coefgqpkwad875[:, 0] / (p ** coefgqpkwad875[:, 1]))
            A_500 = np.column_stack((np.ones(2), -np.log(coefeqpkwad500[:, 2] * p**2)))
            coeffic_500 = np.linalg.solve(A_500, y_500)
            A_875 = np.column_stack((np.ones(2), -np.log(coefgqpkwad875[:, 2] * p**2)))
            coeffic_875 = np.linalg.solve(A_875, y_875)

            fp_500_n = 1 - np.exp(coeffic_500[0]) / n ** coeffic_500[1]
            fp_875_n = 1 - np.exp(coeffic_875[0]) / n ** coeffic_875[1]

        if alpha <= 0.875:
            fp_alpha_n = fp_500_n + (fp_875_n - fp_500_n) / 0.375 * (alpha - 0.5)
        else:
            fp_alpha_n = fp_875_n + (1 - fp_875_n) / 0.125 * (alpha - 0.875)
    return 1 / fp_alpha_n


def get_correction_factor_reweighting(p: int, n: int, alpha: float) -> float:
    """
    Calculate the small sample correction factor for the scale resulting from LTS regression.

    References:
        Pison, G., Van Aelst, S. & Willems, G. Small sample corrections for LTS and MCD.
        Metrika 55, 111–123 (2002). https://doi.org/10.1007/s001840200191

        https://github.com/cran/robustbase/blob/c4b9d21cfc4beb64653bb2ffba9e549e2dbb98ed/R/ltsReg.R
    """
    if alpha < 0.5 or alpha > 1:
        raise ValueError(f"alpha must be between 0.5 and 1, but received {alpha}")
    if p == 0:  # intercept only
        fp_500_n = 1 - np.exp(1.11098143415027) / n**1.5182890270453
        fp_875_n = 1 - np.exp(-0.66046776772861) / n**0.88939595831888
        if 0.5 <= alpha <= 0.875:
            fp_alpha_n = fp_500_n + (fp_875_n - fp_500_n) / 0.375 * (alpha - 0.5)
        else:  # 0.875 < alpha < 1
            fp_alpha_n = fp_875_n + (1 - fp_875_n) / 0.125 * (alpha - 0.875)
        fp_alpha_n = np.sqrt(fp_alpha_n)
    else:
        if p == 1:
            fp_500_n = 1 - np.exp(1.58609654199605) / n**1.46340162526468
            fp_875_n = 1 - np.exp(0.391653958727332) / n**1.03167487483316
        else:
            coefgqpkwad875 = np.array(
                [
                    [-0.474174840843602, 1.39681715704956, 3],
                    [-0.276640353112907, 1.42543242287677, 5],
                ]
            )
            coefeqpkwad500 = np.array(
                [
                    [-0.773365715932083, 2.02013996406346, 3],
                    [-0.337571678986723, 2.02037467454833, 5],
                ]
            )
            y_500 = np.log(-coefeqpkwad500[:, 0] / (p ** coefeqpkwad500[:, 1]))
            y_875 = np.log(-coefgqpkwad875[:, 0] / (p ** coefgqpkwad875[:, 1]))
            A_500 = np.column_stack((np.ones(2), -np.log(coefeqpkwad500[:, 2] * p**2)))
            coeffic_500 = np.linalg.solve(A_500, y_500)
            A_875 = np.column_stack((np.ones(2), -np.log(coefgqpkwad875[:, 2] * p**2)))
            coeffic_875 = np.linalg.solve(A_875, y_875)

            fp_500_n = 1 - np.exp(coeffic_500[0]) / n ** coeffic_500[1]
            fp_875_n = 1 - np.exp(coeffic_875[0]) / n ** coeffic_875[1]

        if alpha <= 0.875:
            fp_alpha_n = fp_500_n + (fp_875_n - fp_500_n) / 0.375 * (alpha - 0.5)
        else:
            fp_alpha_n = fp_875_n + (1 - fp_875_n) / 0.125 * (alpha - 0.875)
    return 1 / fp_alpha_n
