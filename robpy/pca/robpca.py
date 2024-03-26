from __future__ import annotations

from matplotlib.pylab import eig, f
import numpy as np

from sklearn.decomposition import PCA
from scipy.stats import median_abs_deviation, norm
from robpy.pca.base import RobustPCAEstimator, get_od_cutoff
from robpy.utils.outlyingness import stahel_donoho
from robpy.covariance import FastMCDEstimator


class ROBPCAEstimator(RobustPCAEstimator):
    def __init__(
        self,
        *,
        n_components: int | None = None,
        k: int | None = None,
        k_min_var_explained: float = 0.8,
        alpha: float = 0.75,
        final_MCD_step: bool = True,
    ):
        """Implementation of ROBPCA algoirthm as described in
        Hubert, Rousseeus & Vanden Branden (2005)
        ROBPCA: A new approach to robust principal component analysis
        and
        Hubert, Rousseeuw & Verdonck (2009) Robust PCA for skewed data and its outlier map

        Args:
            n_components (int | None, optional):
                Number of components to select. If None, it is set during fit to min (X.shape)
            k (int | None, optional): number of components to keep in
                the intermediate projection step. If None, it is determined by the number of
                components for which at least k_min_var_explained of the variance is explained
            k_min_var_explained (float, optional): minimum variance explained by the k components
                Only used if k is None
            alpha (float, optional): coverage parameter, determines the robustness and efficiency
                trade off of the estimator.
                Smaller alpha gives more robust but less accurate estimates
            final_MCD_step (bool, optional): whether to apply the final MCD step to get maximally
                robust estimates. If False, the eigenvectors after projection onto V1 (subspace
                determined by points with OD < cutoff) are used as the final estimates.
                Defaults to True.
        """
        super().__init__(n_components=n_components)
        self.k = k
        self.k_min_var_explained = k_min_var_explained
        self.alpha = alpha
        self.final_MCD_step = final_MCD_step

    def fit(self, X: np.ndarray) -> ROBPCAEstimator:
        # step 0: set n_components if not set and location
        if self.n_components is None:
            self.n_components = min(X.shape)
        self.location_ = np.mean(X, axis=0)

        # step 1: singular value decomposition = applying standard PCA
        X = PCA().fit_transform(X)
        # step 2: stahel donoho outlyingness --> h subset
        outlyingness = stahel_donoho(X)
        h_index = np.argsort(outlyingness)[: int(self.alpha * X.shape[0])]
        h_cov = np.cov(X[h_index], rowvar=False)
        # step 3: project on k-dimensional subspace
        eigvals_h, eigvecs_h = np.linalg.eigh(h_cov)
        sorted_eig_h_idx = np.argsort(eigvals_h)[::-1]
        var_explained = eigvals_h[sorted_eig_h_idx].cumsum() / eigvals_h.sum()
        if self.k is None:
            self.k = np.argmax(var_explained >= self.k_min_var_explained) + 1
        h_components = eigvecs_h[:, sorted_eig_h_idx[: self.k]]  # (p, k)
        X_proj = (X - self.location_) @ h_components @ h_components.T
        # step 4: orthogonal distance
        orth_dist = np.linalg.norm((X - self.location_) - X_proj, axis=1)
        v_index = np.argwhere(orth_dist < get_od_cutoff(orth_dist)).flatten()
        eigvals_v, eigvecs_v = np.linalg.eigh(np.cov(X[v_index], rowvar=False))
        sorted_eig_v_idx = np.argsort(eigvals_v)[::-1]
        self.components_ = eigvecs_v[:, sorted_eig_v_idx[: self.k]]
        self.explained_variance_ = eigvals_v[sorted_eig_v_idx[: self.k]]
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        if self.final_MCD_step:
            # step 5: final MCD step
            mcd = FastMCDEstimator().fit(self.project(X))
            eigvals_mcd, eigvecs_mcd = np.linalg.eigh(mcd.covariance)
            sorted_eig_mcd_idx = np.argsort(eigvals_mcd)[::-1]
            self.components_ = eigvecs_mcd[:, sorted_eig_mcd_idx[: self.k]]
            self.explained_variance_ = eigvals_mcd[sorted_eig_mcd_idx[: self.k]]
            self.explained_variance_ratio_ = (
                self.explained_variance_ / self.explained_variance_.sum()
            )
        return self
