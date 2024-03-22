from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA
from robpy.pca.base import RobustPCAEstimator


class ROBPCAEstimator(RobustPCAEstimator):
    def __init__(self, *, n_components: int | None = None, alpha: float = 0.75):
        """Base class for robust PCA estimators

        Args:
            n_components (int | None, optional):
                Number of components to select. If None, it is set during fit to min (X.shape)
            alpha (float, optional): coverage parameter, determines the robustness and efficiency
                trade off of the estimator.
                Smaller alpha gives more robust but less accurate estimates
        """
        super().__init__(n_components=n_components)

    def fit(self, X: np.ndarray) -> ROBPCAEstimator:
        # step 1: singular value decomposition = applying standard PCA ()
        X = PCA().fit_transform(X)
        # step 2: stahel donoho outlyingness
