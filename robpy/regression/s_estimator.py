import logging
import numpy as np

from sklearn.exceptions import NotFittedError
from statsmodels.api import WLS, add_constant
from statsmodels.regression.linear_model import RegressionResultsWrapper

from robpy.regression.base import RobustRegressor, _convert_input_to_array
from robpy.utils.rho import BaseRho, TukeyBisquare


class SEstimator(RobustRegressor):
    """Fast S algorithm similar to lmrob.S on robustbase
    (https://search.r-project.org/CRAN/refmans/robustbase/html/lmrob.S.html).
    S-estimation was initially described in

    Rousseeuw, P. J., and Yohai, V. J. (1984), "Robust Regression by Means of S-Estimators,"
    in Robust and Nonlinear Time Series,
    eds. J. Franke, W. Hardie, and D. Martin, Lecture Notes in Statistics, 26, Berlin:
    Springer-Verlag, pp. 256-272.

    This code is an implementation of the Fast S algorithm described in

    Salibian-Barrera, M., & Yohai, V. J. (2006).
    A Fast Algorithm for S-Regression Estimates.
    Journal of Computational and Graphical Statistics, 15(2), 414–427.
    http://www.jstor.org/stable/27594186


    """

    def __init__(
        self,
        rho: BaseRho = TukeyBisquare(c=1.547),
        n_initial_subsets: int = 500,
        n_initial_i_steps: int = 2,
        n_best_subsets: int = 5,
        max_scale_iterations: int = 2,
        b: float = 0.5,
        fit_intercept: bool = True,
        relative_tolerance: float = 1e-7,
        scale_tolerance: float = 1e-10,
        random_state: int = 101,
    ):
        """Fast S estimator.

        Args:
            rho (BaseRho, optional):
                score function to use on the residuals. Defaults to bisquare.
            n_initial_subsets (int, optional):
                Number of initial subsets to sample (`N` in the original paper).
                Defaults to 500.
            n_initial_i_steps (int, optional):
                Number of i-steps to take on the initial subsets (`k` in the original paper).
                Defaults to 2.
            n_best_subsets (int, optional):
                Number of subsets with the best M-scales (residuals transformered by score function)
                (`t` in the original paper).
                Defaults to 5.
            max_scale_iterations (int, optional):
                number of iterative steps to derive M-scale estimates (`r` in the original paper).
            b (float, optional):
                constant on the RHS of the M scale equation
            fit_intercept (bool, optional):
                Whether an intercept should be included in the linear regression
            relative_tolerance (float, optional):
                Determines the stopping criterium for the i-steps untill convergence
                (diff in beta norm should be higher then
                relative_tolerance * max(relative_tolerance, beta_norm))
            scale_tolerance (float, optional):
                If the difference between 2 subsequent scale estimates is below this threshold,
                the iterations are stopped and it is assumed the scale estimate converged.
        """
        self.rho = rho
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_i_steps = n_initial_i_steps
        self.n_best_subsets = n_best_subsets
        self.max_scale_iterations = max_scale_iterations
        self.b = b
        self.fit_intercept = fit_intercept
        self.model = None
        self.random_state = random_state
        self.relative_tolerance = relative_tolerance
        self.scale_tolerance = scale_tolerance
        self.logger = logging.getLogger("S-estimator")

    def fit(self, X, y, verbosity=logging.WARNING):
        self.logger.setLevel(verbosity)

        X, y = _convert_input_to_array(X, y)
        if self.fit_intercept:
            X = add_constant(X)
        models, scales = self._get_initial_models(X, y)
        self.logger.info(f"Fitted {len(models)} initial models.")
        best_idx = np.argsort(scales)[: self.n_best_subsets]
        best_models = []
        best_scales = []
        for i, idx in enumerate(best_idx, start=1):
            self.logger.info(
                f"Applying i-steps untill convergence for model {i} out of {self.n_best_subsets}"
            )
            previous_m = models[idx]
            previous_s = scales[idx]
            n_iter = 0
            while True:
                try:
                    m, s = self._i_step(X, y, (previous_m.predict(X) - y).flatten())
                    n_iter += 1
                except np.linalg.LinAlgError:
                    self.logger.warning(
                        f"Failed to aply i-steps untill convergence. "
                        f"Starting scale was: {previous_s}. Failed iteration = {n_iter}."
                        f"Storing model after last successfull i-step."
                    )
                    m.n_iter = n_iter
                    best_models.append(previous_m)
                    best_scales.append(previous_s)
                    break
                if self._converged(previous_m.params, m.params):
                    m.n_iter = n_iter
                    best_models.append(m)
                    best_scales.append(s)
                    break
                previous_m = m
                previous_s = s

        best_idx = np.argmin(best_scales)
        self.model = best_models[best_idx]
        self._scale = best_scales[best_idx]
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise NotFittedError
        X, _ = _convert_input_to_array(X)
        if self.fit_intercept:
            X = add_constant(X)
        return self.model.predict(X)

    def _get_scale(self, s0: float, residuals: np.ndarray):
        for i in range(self.max_scale_iterations):
            new_s = s0 * np.sqrt(np.mean(self.rho.rho(residuals / s0)) / self.b)
            if np.abs(s0 - new_s) < self.scale_tolerance:
                break
            s0 = new_s
        return new_s

    def _converged(self, previous_beta: np.ndarray, current_beta: np.ndarray) -> bool:
        norm_diff = float(np.linalg.norm(current_beta - previous_beta))
        norm_beta = float(np.linalg.norm(current_beta))

        return norm_diff <= self.relative_tolerance * max(self.relative_tolerance, norm_beta)

    def _i_step(
        self,
        X,
        y,
        residuals: np.ndarray,
        initial_scale: float | None = None,
    ) -> tuple[RegressionResultsWrapper, float]:
        if initial_scale is None:
            initial_scale = np.median(np.abs(residuals)) / 0.6745
        new_scale = self._get_scale(s0=initial_scale, residuals=residuals)
        residuals[residuals == 0] = np.finfo(np.float64).eps  # avoid divide by 0 errors
        weights = self.rho.psi(residuals / new_scale) / (residuals / new_scale)
        model = WLS(endog=y, exog=X, weights=weights).fit()
        model.weights = weights
        return model, new_scale

    def _get_initial_models(self, X, y) -> tuple[list[RegressionResultsWrapper], list[float]]:
        models = []
        scales = []
        for seed in range(self.random_state, self.random_state + self.n_initial_subsets):
            np.random.seed(seed)
            sample_idx = np.random.choice(X.shape[0], X.shape[1] + 1, replace=False)
            model = WLS(y[sample_idx], X[sample_idx]).fit()
            residuals = model.predict(X) - y
            scale = np.median(np.abs(residuals)) / 0.6745
            try:
                for _ in range(self.n_initial_i_steps):
                    model, scale = self._i_step(X, y, residuals)
                    residuals = model.resid
                models.append(model)
                scales.append(scale)
            except np.linalg.LinAlgError as e:
                self.logger.debug(f"Failed to apply i-steps on initial model with seed {seed}: {e}")

        return models, scales
