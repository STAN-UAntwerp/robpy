import logging
import numpy as np
import math as math
import warnings

from dataclasses import dataclass
from scipy.linalg import sqrtm
from scipy.stats import chi2, gamma, rankdata, norm

from robpy.covariance.base import RobustCovariance
from robpy.utils.distance import mahalanobis_distance
from robpy.utils.logging import get_logger
from robpy.univariate.qn import Qn
from robpy.covariance.ogk import OGK
from robpy.univariate.tau import Tau


@dataclass
class HSubset:
    indices: np.ndarray
    location: np.ndarray
    scale: np.ndarray
    determinant: float
    n_c_steps: int = 0


class FastMCD(RobustCovariance):
    def __init__(
        self,
        *,
        alpha: float | int | None = None,
        n_initial_subsets: int = 500,
        n_initial_c_steps: int = 2,
        n_best_subsets: int = 10,
        n_partitions: int | None = None,
        tolerance: float = 1e-8,
        correct_covariance: bool = True,
        reweighting: bool = True,
        verbosity: int = logging.WARNING,
        store_precision=True,
        assume_centered=False,
        random_seed: int | None = None,
    ):
        """
        Fast MCD estimator based on the algorithm proposed in Rousseeuw, P. J., & Van Driessen, K.
        (1999).

        Args:
            alpha (float | int | None, optional):
              Size of the h subset.
              If an integer between n/2 and n is passed, it is interpreted as h.
              If a float between 0.5 and 1 is passed, it is interpreted as a proportion of n
              (the training set size).
              If None or an integer below [(n+p+1)/2], h is set to [(n+p+1)/2].
              Defaults to None.
            n_initial_subsets (int, optional):
              Number of initial random subsets of size p+1.
              Defaults to 500.
            n_initial_c_steps (int, optional):
              Number of initial c steps to perform on all initial subsets.
              Defaults to 2.
            n_best_subsets (int, optional):
              Number of best subsets to keep and perform c steps on until convergence.
              Defaults to 10.
            n_partitions (int, optional):
              Number of partitions to split the data into.
              This can speed up the algorithm for large datasets (n > 600 suggested in paper).
              If None, 5 partitions are used if n > 600, otherwise 1 partition is used.
            tolerance (float, optional):
              Minimum difference in determinant between two iterations to stop the C-step.
              Defaults to 1e-8.
            correct_covariance (bool, optional):
              Whether to apply a consistency correction to the raw covariance estimate.
              Defaults to True.
            reweighting (bool, optional):
              Whether to apply reweighting to the raw covariance estimate.
              Defaults to True.
            random_seed (int | None, optional):
              Can be used to provide a random seed. Defaults to None.

        References:
            - Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum
              covariance determinant estimator. Technometrics, 41(3), 212-223.

        """
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.alpha = alpha
        self.n_initial_subsets = n_initial_subsets
        self.n_initial_c_steps = n_initial_c_steps
        self.n_best_subsets = n_best_subsets
        self.n_partitions = n_partitions
        self.tolerance = tolerance
        self.correct_covariance = correct_covariance
        self.reweighting = reweighting
        self.logger = get_logger("FastMCD", level=verbosity)
        self.verbosity = verbosity
        self.random_seed = random_seed

    def calculate_covariance(self, X) -> np.ndarray:
        if self.alpha == 1 or self.alpha == X.shape[0]:
            self.logger.warning(f"Default covariance is returned as alpha is {self.alpha}.")
            self.location_ = X.mean(0)
            return np.cov(X, rowvar=False)
        self.rng = np.random.default_rng(self.random_seed)
        # partition data (n_partitions > 1 can speed up algorithm for large datasets)
        partitions = self._partition_data(X)
        self.logger.info(f"Partitioned data into {len(partitions)} partitions")
        n_initial_subsets = self.n_initial_subsets // len(partitions)
        best_subsets = []
        # perform initial c steps on all initial subsets
        for data in partitions:
            subsets = self._get_initial_subsets(data, n_initial_subsets)
            subsets = [
                self._perform_multiple_c_steps(subset, data, n_iterations=self.n_initial_c_steps)
                for subset in subsets
            ]
            best_subsets.extend(sorted(subsets, key=lambda x: x.determinant)[: self.n_best_subsets])
        self.logger.info(f"Selecting {self.n_best_subsets} best subsets from {len(best_subsets)}")
        # perform additional c-steps on the best subsets
        best_subsets = [
            self._perform_multiple_c_steps(subset, X, n_iterations=self.n_initial_c_steps)
            for subset in best_subsets
        ]
        best_subsets = sorted(best_subsets, key=lambda x: x.determinant)[: self.n_best_subsets]
        best_subset = best_subsets[0]  # reference subset
        # perform c-steps until convergence
        for subset in best_subsets:
            while True:
                new_subset = self._perform_c_step(subset, X)
                if (
                    np.all(new_subset.indices == subset.indices)
                    or (new_subset.determinant - subset.determinant) < self.tolerance
                ):
                    break
                subset = new_subset
            if subset.determinant < best_subset.determinant:
                best_subset = subset
        self.best_subset = best_subset

        # post processing
        self.location_ = best_subset.location
        scale = best_subset.scale
        if self.correct_covariance:
            self.logger.debug("Applying consistency correction to the raw covariance estimate")
            scale = self._correct_covariance(X, best_subset)
        if self.reweighting:
            self.logger.debug("Applying reweighting to the raw covariance estimate")
            distances = mahalanobis_distance(X, best_subset.location, scale)
            self._mask = distances < np.sqrt(chi2.ppf(0.975, X.shape[1]))
            self.location_ = X[self._mask].mean(axis=0)
            scale = np.cov(X[self._mask], rowvar=False)
            if self.correct_covariance:
                # See Croux & Haesbroeck 1999 and R code for robustbase>covMcd>MCDCons
                # (https://rdrr.io/cran/robustbase/src/R/covMcd.R)
                self.logger.debug("Applying consistency correction after reweighting.")
                factor = 1 / (
                    gamma.cdf((chi2.ppf(0.975, X.shape[1]) / 2), X.shape[1] / 2 + 1) / 0.975
                )
                scale *= factor

        return scale

    def _correct_covariance(self, X: np.ndarray, best_subset: HSubset) -> np.ndarray:
        self._rescale_factor = np.median(
            np.square(mahalanobis_distance(X, best_subset.location, best_subset.scale))
        ) / chi2.ppf(0.5, X.shape[1])

        return self._rescale_factor * best_subset.scale

    def _perform_multiple_c_steps(
        self, subset: HSubset, X: np.ndarray, n_iterations: int
    ) -> HSubset:
        for _ in range(n_iterations):
            subset = self._perform_c_step(subset, X)
        return subset

    def _perform_c_step(self, subset: HSubset, X: np.ndarray) -> HSubset:
        """
        Perform a single C-step on the subset of the data

        Args:
            subset (HSubset): indices of the current subset.
            X (np.ndarray): data.

        Returns:
            HSubset: indices for new h subset.
        """
        # Calculate the Mahalanobis distances
        mahalanobis = mahalanobis_distance(X, subset.location, subset.scale)
        # Find the alpha (h_size) smallest distances
        h = self._get_h(X)
        idx = np.argsort(mahalanobis)[:h]

        return self._get_subset(indices=idx, X=X, n_c_steps=subset.n_c_steps + 1)

    def _get_h(self, X: np.ndarray) -> int:
        n, p = X.shape
        if self.alpha is None:
            return int((n + p + 1) / 2)
        elif isinstance(self.alpha, int) and (n / 2 <= self.alpha <= n):
            if self.alpha < int((n + p + 1) / 2):
                warnings.warn(
                    f"h is too small and therefore set to [(n+p+1)/2] ({int((n + p + 1) / 2)}).",
                    category=UserWarning,
                    stacklevel=2,
                )
            return np.max([self.alpha, int((n + p + 1) / 2)])
        elif (isinstance(self.alpha, float) and (0.5 <= self.alpha <= 1)) or self.alpha == 1:
            if int(self.alpha * n) < int((n + p + 1) / 2):
                warnings.warn(
                    f"h = alpha*n is too small and therefore set to [(n+p+1)/2]"
                    f" ({int((n + p + 1) / 2)}).",
                    category=UserWarning,
                    stacklevel=2,
                )
            return np.max([int(self.alpha * n), int((n + p + 1) / 2)])
        else:
            raise ValueError(
                f"alpha must be an integer between n/2 and n or"
                f" a float between 0.5 and 1, but received {self.alpha}."
            )

    def _partition_data(self, X: np.ndarray) -> list[np.ndarray]:
        if self.n_partitions is None:
            n_partitions = 5 if X.shape[0] > 600 else 1
        else:
            n_partitions = self.n_partitions

        return np.array_split(X, n_partitions)

    def _get_subset(
        self,
        indices: np.ndarray,
        X: np.ndarray,
        n_c_steps: int = 0,
        ensure_non_singular: bool = False,
    ) -> HSubset:
        """
        Construct an HSubset from a set of data indices and calculate location, scale and
        determinant.

        Args:
             - indices (np.ndarray): data indices.
             - X (np.ndarray): complete dataset.
             - n_c_steps (int): will be passed directly to the HSubset.
             - ensure_non_singular (bool): whether to resample in case the determinant is 0
                (relevant for sampling initial subsets).

        """
        mu = X[indices].mean(axis=0)
        cov = np.cov(X[indices], rowvar=False)
        det = np.linalg.det(cov)
        if ensure_non_singular:
            while math.isclose(det, 0):
                new_index = self.rng.choice(np.delete(np.arange(X.shape[0]), indices))
                indices = np.append(indices, new_index)
                mu = X[indices].mean(axis=0)
                cov = np.cov(X[indices], rowvar=False)
                det = np.linalg.det(cov)
        return HSubset(indices, mu, cov, det, n_c_steps=n_c_steps)

    def _get_initial_subsets(self, X: np.ndarray, n_subsets: int) -> list[HSubset]:
        return [
            self._get_subset(
                indices=self.rng.choice(X.shape[0], X.shape[1] + 1, replace=False),
                X=X,
                ensure_non_singular=True,
            )
            for _ in range(n_subsets)
        ]


class DetMCD(RobustCovariance):
    def __init__(
        self,
        *,
        alpha: float | int | None = None,
        tolerance: float = 1e-8,
        correct_covariance: bool = True,
        reweighting: bool = True,
        verbosity: int = logging.WARNING,
    ):
        """
        Deterministic MCD estimator (DetMCD) based on the algorithm proposed in Hubert, M.,
        Rousseeuw, P. J., & Verdonck, T. (2012).

        Args:
            alpha (float | int | None, optional):
                Size of the h subset.
                If an integer between n/2 and n is passed, it is interpreted as h.
                If a float between 0.5 and 1 is passed, it is interpreted as a proportion of n
                (the training set size).
                If None or an integer below [(n+p+1)/2], h is set to [(n+p+1)/2].
                Defaults to None.
            tolerance (float, optional):
                Minimum difference in determinant between two iterations to stop the C-step.
                Defaults to 1e-8.
            correct_covariance (bool, optional):
                Whether to apply a consistency correction to the raw covariance estimate.
                Defaults to True.
            reweighting (bool, optional):
                Whether to apply reweighting to the raw covariance estimate.
                Defaults to True.

        References:
            - Hubert, M., Rousseeuw, P. J., & Verdonck, T. (2012). A deterministic algorithm for
              robust location and scatter. Journal of Computational and Graphical Statistics, 21(3),
              618-637.

        """
        super().__init__()
        self.alpha = alpha
        self.tolerance = tolerance
        self.correct_covariance = correct_covariance
        self.reweighting = reweighting
        self.logger = get_logger("DetMCD", level=verbosity)
        self.verbosity = verbosity

    def calculate_covariance(self, X: np.ndarray) -> np.ndarray:
        if self.alpha == 1 or self.alpha == X.shape[0]:
            self.logger.warning(f"Default covariance is returned as alpha is {self.alpha}.")
            self.location_ = X.mean(0)
            return np.cov(X, rowvar=False)

        n, p = X.shape

        # Step 0: standardize X
        if n < 1000:
            Z = (X - np.median(X, axis=0)) / self._Qn_scale(X)
        else:
            Z = (X - np.median(X, axis=0)) / self._tau_scale(X)

        # Steps 1-3:
        best_subsets = self._get_initial_best_subsets(Z, X, n, p)

        # Step 4: C-steps until convergence
        best_subset = best_subsets[0]  # reference subset
        for subset in best_subsets:
            while True:
                new_subset = self._perform_c_step(subset, X)
                if (
                    np.all(new_subset.indices == subset.indices)
                    or (new_subset.determinant - subset.determinant) < self.tolerance
                ):
                    break
                subset = new_subset
            if subset.determinant < best_subset.determinant:
                best_subset = subset
        self.best_subset = best_subset

        # Step 5: post processing
        self.location_ = best_subset.location
        scale = best_subset.scale
        if self.correct_covariance:
            self.logger.debug("Applying consistency correction to the raw covariance estimate")
            scale = self._correct_covariance(X, best_subset)
        if self.reweighting:
            self.logger.debug("Applying reweighting to the raw covariance estimate")
            distances = mahalanobis_distance(X, best_subset.location, scale)
            self._mask = distances < np.sqrt(chi2.ppf(0.975, X.shape[1]))
            self.location_ = X[self._mask].mean(axis=0)
            scale = np.cov(X[self._mask], rowvar=False)
            if self.correct_covariance:
                # See Croux & Haesbroeck 1999 and R code for robustbase>covMcd>MCDCons
                # (https://rdrr.io/cran/robustbase/src/R/covMcd.R)
                self.logger.debug("Applying consistency correction after reweighting.")
                factor = 1 / (
                    gamma.cdf((chi2.ppf(0.975, X.shape[1]) / 2), X.shape[1] / 2 + 1) / 0.975
                )
                scale *= factor

        return scale

    def _Qn_scale(self, X: np.ndarray, axis=0):
        if X.ndim == 1:
            return Qn().fit(X).scale
        elif axis == 0:
            return [Qn().fit(col).scale for col in X.T]
        elif axis == 1:
            return [Qn().fit(col).scale for col in X]
        else:
            raise ValueError(f"axis {axis} not supported")

    def _tau_scale(self, X: np.ndarray, axis=0):
        if X.ndim == 1:
            return Tau().fit(X).scale
        elif axis == 0:
            return [Tau().fit(col).scale for col in X.T]
        elif axis == 1:
            return [Tau().fit(col).scale for col in X]
        else:
            raise ValueError(f"axis {axis} not supported")

    def _get_initial_best_subsets(self, Z: np.ndarray, X: np.ndarray, n: int, p: int):
        # Step 1: construct 6 preliminary estimates Sk of covariance or correlation
        Y = np.tanh(Z)
        S1 = np.corrcoef(Y, rowvar=False)
        R = rankdata(Z, axis=0)
        S2 = np.corrcoef(R, rowvar=False)
        S3 = np.corrcoef(norm.ppf((R - 1 / 3) / (n + 1 / 3)), rowvar=False)
        znorm = np.sqrt(np.sum(Z * Z, axis=1))
        w = 1 / znorm
        S4 = np.dot((Z * w[:, np.newaxis]).T, (Z * w[:, np.newaxis])) / n
        idx = np.argsort(znorm)[: math.ceil(n / 2)]
        S5 = np.cov(Z[idx, :], rowvar=False)
        S6 = OGK(
            location_estimator=np.median, scale_estimator=self._Qn_scale, reweighting=False
        ).calculate_covariance(Z)
        estimates_S = [S1, S2, S3, S4, S5, S6]

        # Step 2: construct 6 initial location and scatter estimates
        estimates_sigma = []
        estimates_mu = []
        for S in estimates_S:
            _, E = np.linalg.eigh(S)
            E = E[:, np.arange(p - 1, -1, -1)]
            B = Z @ E
            L = np.diag(np.power(self._Qn_scale(B), 2))
            cov = E @ L @ E.T
            estimates_sigma.append(cov)
            root_cov = sqrtm(cov)
            inv_root_cov = np.linalg.inv(root_cov)
            mu = root_cov @ np.median(Z @ inv_root_cov, axis=0)
            estimates_mu.append(mu)

        # Step 3: calculate statistical distances
        best_subsets = []
        for mu, cov in zip(estimates_mu, estimates_sigma):
            idx_h0 = np.argsort(mahalanobis_distance(Z, mu, cov))[: math.ceil(n / 2)]
            H = self._get_subset(idx_h0, X)  # h0
            H = self._perform_c_step(H, X)  # h
            best_subsets.append(H)

        return best_subsets

    # TODO: de functies hieronder moeten nog naar ergens anders.

    def _get_h(self, X: np.ndarray) -> int:
        n, p = X.shape
        if self.alpha is None:
            return int((n + p + 1) / 2)
        elif isinstance(self.alpha, int) and (n / 2 <= self.alpha <= n):
            if self.alpha < int((n + p + 1) / 2):
                warnings.warn(
                    f"h is too small and therefore set to [(n+p+1)/2] ({int((n + p + 1) / 2)}).",
                    category=UserWarning,
                    stacklevel=2,
                )
            return np.max([self.alpha, int((n + p + 1) / 2)])
        elif (isinstance(self.alpha, float) and (0.5 <= self.alpha <= 1)) or self.alpha == 1:
            if int(self.alpha * n) < int((n + p + 1) / 2):
                warnings.warn(
                    f"h = alpha*n is too small and therefore set to [(n+p+1)/2]"
                    f" ({int((n + p + 1) / 2)}).",
                    category=UserWarning,
                    stacklevel=2,
                )
            return np.max([int(self.alpha * n), int((n + p + 1) / 2)])
        else:
            raise ValueError(
                f"alpha must be an integer between n/2 and n or"
                f" a float between 0.5 and 1, but received {self.alpha}."
            )

    def _get_subset(
        self,
        indices: np.ndarray,
        X: np.ndarray,
        n_c_steps: int = 0,
        ensure_non_singular: bool = False,
    ) -> HSubset:
        """Construct an HSubset from a set of data indices and calculate location, scale and
         determinant.

        Args:
             - indices (np.ndarray): data indices.
             - X (np.ndarray): complete dataset.
             - n_c_steps (int): will be passed directly to the HSubset.
             - ensure_non_singular (bool): whether to resample in case the determinant is 0
             (relevant for sampling initial subsets).

        Returns:
            Hsubset.
        """
        mu = X[indices].mean(axis=0)
        cov = np.cov(X[indices], rowvar=False)
        det = np.linalg.det(cov)
        if ensure_non_singular:
            while math.isclose(det, 0):
                new_index = np.random.choice(np.delete(np.arange(X.shape[0]), indices))
                indices = np.append(indices, new_index)
                mu = X[indices].mean(axis=0)
                cov = np.cov(X[indices], rowvar=False)
                det = np.linalg.det(cov)
        return HSubset(indices, mu, cov, det, n_c_steps=n_c_steps)

    def _perform_c_step(self, subset: HSubset, X: np.ndarray) -> HSubset:
        """Perform a single C-step on the subset of the data

        Args:
            subset (HSubset): indices of the current subset.
            X (np.ndarray): data.

        Returns:
            HSubset: indices for new h subset.
        """
        # Calculate the Mahalanobis distances
        mahalanobis = mahalanobis_distance(X, subset.location, subset.scale)
        # Find the alpha (h_size) smallest distances
        h = self._get_h(X)
        idx = np.argsort(mahalanobis)[:h]

        return self._get_subset(indices=idx, X=X, n_c_steps=subset.n_c_steps + 1)

    def _correct_covariance(self, X: np.ndarray, best_subset: HSubset) -> np.ndarray:
        self._rescale_factor = np.median(
            np.square(mahalanobis_distance(X, best_subset.location, best_subset.scale))
        ) / chi2.ppf(0.5, X.shape[1])

        return self._rescale_factor * best_subset.scale
