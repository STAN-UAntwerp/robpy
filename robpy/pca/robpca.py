from __future__ import annotations

import numpy as np
import logging

from sklearn.decomposition import PCA
from robpy.pca.base import RobustPCA, get_od_cutoff
from robpy.utils.outlyingness import stahel_donoho
from robpy.utils.logging import get_logger
from robpy.covariance import FastMCD


class ROBPCA(RobustPCA):
    def __init__(
        self,
        *,
        n_components: int | None = None,
        k_min_var_explained: float = 0.8,
        alpha: float = 0.75,
        final_MCD_step: bool = True,
        random_seed: int | None = None,
        verbosity: int = logging.WARNING,
    ):
        """Implementation of ROBPCA algorithm as described in
        Hubert, M., Rousseeuw, P. J., & Vanden Branden, K. (2005).

        Args:
            n_components (int | None, optional):
                Number of components to select. If None, it is set during fit. Defaults to None.
            k_min_var_explained (float, optional): Minimum variance explained by the components.
                Only used if n_components is None. Defaults to 0.8.
            alpha (float, optional): Coverage parameter, determines the robustness and efficiency
                trade off of the estimator. Smaller alpha gives more robust but less accurate
                estimates. Must be a number between 0.5 and 1. Defaults to 0.75.
            final_MCD_step (bool, optional): Whether to apply the final MCD step to get maximally
                robust estimates. If False, the eigenvectors after projection onto V1 (subspace
                determined by points with OD < cutoff) are used as the final estimates.
                Defaults to True.
            random_seed (int | None, optional):
                Can be used to provide a random seed. Defaults to None.

        References:
            - Hubert, M., Rousseeuw, P. J., & Vanden Branden, K. (2005). ROBPCA: a new approach to
              robust principal component analysis. Technometrics, 47(1), 64-79.
        """
        super().__init__(n_components=n_components)
        self.k_min_var_explained = k_min_var_explained
        self.alpha = alpha
        self.final_MCD_step = final_MCD_step
        self.random_seed = random_seed
        self.logger = get_logger("ROBPCA", level=verbosity)
        self.verbosity = verbosity

    def fit(self, X: np.ndarray) -> ROBPCA:
        # step 1: singular value decomposition = applying standard PCA
        pca = PCA()
        X = pca.fit_transform(X)
        self.location_ = np.mean(X, axis=0)
        loadings = pca.components_.T
        # step 2: stahel donoho outlyingness --> h subset
        outlyingness = stahel_donoho(X, random_seed=self.random_seed)

        if not (isinstance(self.alpha, (int, float)) and 0.5 <= self.alpha <= 1):
            raise ValueError(
                f"alpha must be between 0.5 and 1 (inclusive), but received {self.alpha}."
            )
        elif self.n_components is not None:
            if int(self.alpha * X.shape[0]) < int((X.shape[0] + self.n_components + 1) / 2):
                self.logger.warning(
                    f"h is too small and therefore set to [(n+k+1)/2]"
                    f" ({int((X.shape[0] + self.n_components + 1) / 2)})."
                )
            h = np.max(
                [int(self.alpha * X.shape[0]), int((X.shape[0] + self.n_components + 1) / 2)]
            )
        else:
            if int(self.alpha * X.shape[0]) < int((X.shape[0] + 10 + 1) / 2):
                self.logger.warning(
                    f"h is too small and therefore set to [(n+10+1)/2]"
                    f" ({int((X.shape[0] + 10 + 1) / 2)})."
                )
            h = np.max([int(self.alpha * X.shape[0]), int((X.shape[0] + 10 + 1) / 2)])

        h_index = np.argsort(outlyingness)[:h]
        h_cov = np.cov(X[h_index], rowvar=False)
        # step 3: project on k-dimensional subspace
        eigvals_h, eigvecs_h = np.linalg.eigh(h_cov)
        sorted_eig_h_idx = np.argsort(eigvals_h)[::-1]
        var_explained_ratio = eigvals_h[sorted_eig_h_idx].cumsum() / eigvals_h.sum()
        if self.n_components is None:
            k = np.argmax(var_explained_ratio >= self.k_min_var_explained) + 1
        else:
            k = self.n_components
        self.explained_variance_ = eigvals_h[sorted_eig_h_idx[:k]]
        self.explained_variance_ratio = self.explained_variance_[:k].cumsum() / eigvals_h.sum()
        h_components = eigvecs_h[:, sorted_eig_h_idx[:k]]  # (p, k)
        X_proj = (X - self.location_) @ h_components @ h_components.T
        # step 4: orthogonal distance
        orth_dist = np.linalg.norm((X - self.location_) - X_proj, axis=1)
        v_index = np.argwhere(orth_dist < get_od_cutoff(orth_dist)).flatten()
        cov_v = np.cov(X[v_index], rowvar=False)
        eigvals_v, eigvecs_v = np.linalg.eigh(cov_v)
        k = min(k, np.linalg.matrix_rank(cov_v, hermitian=True))
        sorted_eig_v_idx = np.argsort(eigvals_v)[::-1]
        self.components_ = eigvecs_v[:, sorted_eig_v_idx[:k]]
        if self.final_MCD_step and k > 1:
            # step 5: final MCD step
            mcd = FastMCD().fit(self.transform(X))
            _, eigvecs_mcd = np.linalg.eigh(mcd.covariance)
            k = min(k, np.linalg.matrix_rank(mcd.covariance, hermitian=True))
            self.components_ = self.components_ @ np.flip(eigvecs_mcd[:, -k:], axis=1)
        # step 6: transform back to original X
        self.n_components = k
        X = pca.inverse_transform(X)
        self.components_ = loadings @ self.components_
        self.location_ = np.mean(X, axis=0)
        return self
