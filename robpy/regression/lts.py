from __future__ import annotations
import logging

import numpy as np
from sklearn.exceptions import NotFittedError

from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression

from robpy.regression.base import RobustRegressor


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
            tolerance (float): Acceptable delta in loss value between C-steps.
                               If current loss  -  previous loss <= tolerance, model is converged.
                               Defaults to 1e-15.
        """
        self.alpha = alpha
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_c_steps = n_initial_c_steps
        self.n_best_models = n_best_models
        self.tolerance = tolerance
        self.random_state = random_state
        self.logger = logging.getLogger("FastLTS")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
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
        h = int(X.shape[0] * self.alpha)
        self.logger.info(
            f"Applying {self.n_initial_c_steps} initial c-steps "
            f"on {self.n_initial_subsets} initial subsets"
        )
        y = y.reshape(-1, 1)
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

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model"):
            raise NotFittedError
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