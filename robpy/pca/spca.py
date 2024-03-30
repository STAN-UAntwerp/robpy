from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA
from robpy.pca.base import RobustPCAEstimator, get_od_cutoff
from robpy.utils.l1median import l1median
from scipy.stats import median_abs_deviation


class PCALocantoreEstimator(RobustPCAEstimator):
    def __init__(
        self,
        *,
        n_components: int | None = None,
        k_min_var_explained: float = 0.8,
    ):
        """Spherical PCA

        Args:
            n_components (int | None, optional):
                Number of components to select. If None, it is set during fit to explain the
                minimum variance.
            k_min_var_explained (float, optional):
                Minimum variance explained by the n_components
                Only used if n_components is None.
        """
        super().__init__(n_components=n_components)
        self.k_min_var_explained = k_min_var_explained

    def fit(self, X: np.ndarray) -> PCALocantoreEstimator:
        n = np.shape(X)[0]
        self.location_ = l1median(X)
        centered_X = X - self.location_
        d = np.sqrt(np.sum(centered_X * centered_X, axis=1))
        W = 1 / d
        SSCM = np.dot((centered_X * W[:, np.newaxis]).T, (centered_X * W[:, np.newaxis])) / n
        _, eigenvectors = np.linalg.eigh(SSCM)
        self.components_ = np.fliplr(eigenvectors)
        eigenvalues = np.square(
            np.apply_along_axis(
                median_abs_deviation,
                axis=0,
                arr=self.transform(X),
            )
            * 1.4826
        )
        var_explained_ratio = eigenvalues.cumsum() / eigenvalues.sum()
        if self.n_components is None:
            self.n_components = np.argmax(var_explained_ratio >= self.k_min_var_explained) + 1
        self.components_ = self.components_[:, : self.n_components]
        self.explained_variance_ = eigenvalues[: self.n_components]
        self.explained_variance_ratio_ = var_explained_ratio[: self.n_components]
        return self
