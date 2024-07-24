from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA
from robpy.pca.base import RobustPCAEstimator, get_od_cutoff
from robpy.utils.outlyingness import stahel_donoho
from robpy.covariance import FastMCDEstimator


class ROBPCAEstimator(RobustPCAEstimator):
    def __init__(
        self,
        *,
        n_components: int | None = None,
        k_min_var_explained: float = 0.8,
        alpha: float = 0.75,
        final_MCD_step: bool = True,
    ):
        """Implementation of ROBPCA algorithm as described in
        Hubert, Rousseeuw & Vanden Branden (2005) and Hubert, Rousseeuw & Verdonck (2009)


        Args:
            n_components (int | None, optional):
                Number of components to select. If None, it is set during fit to min (X.shape)
            k_min_var_explained (float, optional): minimum variance explained by the n_components
                Only used if n_components is None
            alpha (float, optional): coverage parameter, determines the robustness and efficiency
                trade off of the estimator.
                Smaller alpha gives more robust but less accurate estimates
            final_MCD_step (bool, optional): whether to apply the final MCD step to get maximally
                robust estimates. If False, the eigenvectors after projection onto V1 (subspace
                determined by points with OD < cutoff) are used as the final estimates.
                Defaults to True.

        References:
            - Hubert, Rousseeuw & Vanden Branden (2005),
              ROBPCA: A new approach to robust principal component analysis
            - Hubert, Rousseeuw & Verdonck (2009) Robust PCA for skewed data and its outlier map

        """
        super().__init__(n_components=n_components)
        self.k_min_var_explained = k_min_var_explained
        self.alpha = alpha
        self.final_MCD_step = final_MCD_step

    def fit(self, X: np.ndarray) -> ROBPCAEstimator:
        # step 1: singular value decomposition = applying standard PCA
        pca = PCA()
        X = pca.fit_transform(X)
        self.location_ = np.mean(X, axis=0)
        loadings = pca.components_.T
        # step 2: stahel donoho outlyingness --> h subset
        outlyingness = stahel_donoho(X)
        h_index = np.argsort(outlyingness)[: int(self.alpha * X.shape[0])]
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
            mcd = FastMCDEstimator().fit(self.transform(X))
            _, eigvecs_mcd = np.linalg.eigh(mcd.covariance)
            k = min(k, np.linalg.matrix_rank(mcd.covariance, hermitian=True))
            self.components_ = self.components_ @ np.flip(eigvecs_mcd[:, -k:], axis=1)
        # step 6: transform back to original X
        self.n_components = k
        X = pca.inverse_transform(X)
        self.components_ = loadings @ self.components_
        self.location_ = np.mean(X, axis=0)
        return self
