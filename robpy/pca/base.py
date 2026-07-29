import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import abstractmethod
from sklearn.decomposition._base import _BasePCA
from scipy.stats import chi2, norm
from robpy.univariate import UnivariateMCD


class RobustPCA(_BasePCA):
    def __init__(self, *, n_components: int | None = None):
        """Base class for robust PCA estimators.

        Args:
            n_components (int | None, optional):
                Number of components to select. If None, it is set during fit.

        """
        self.n_components = n_components

    @abstractmethod
    def fit(self, X: np.ndarray):
        """Fit the robust PCA model to the data.

        Args:
            X (np.ndarray): Data to fit the model to.
        """
        pass

    @property
    def _column_scales(self) -> np.ndarray | float:
        """Scales the columns were divided by before fitting.

        Estimators that robustly standardize the columns (e.g. :class:`~robpy.pca.macropca.
        MacroPCA` and :class:`~robpy.pca.cellpca.CellPCA`) override this so that
        :meth:`transform` and :meth:`project` can undo the standardization. The default is 1.0,
        i.e. no standardization.
        """
        return 1.0

    def _prepare_input(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Coerce X to a float array with the columns the model was fitted on.

        Estimators that clean the data during ``fit`` (dropping non-numeric, discrete or
        constant columns) record the retained columns in ``columns_kept_``. New data may then be
        passed either with the original columns (they are subsetted here) or with the retained
        columns only.

        Args:
            X (np.ndarray | pd.DataFrame): data matrix.

        Returns:
            np.ndarray: float array of shape (n_samples, n_features_kept).

        Raises:
            ValueError: if X has neither the original nor the retained number of columns.
        """
        X = np.asarray(X.values if isinstance(X, pd.DataFrame) else X, dtype=float)
        columns = getattr(self, "columns_kept_", None)
        if columns is None or X.shape[1] == len(columns):
            return X
        if X.shape[1] == getattr(self, "n_features_in_", None):
            return X[:, columns]
        raise ValueError(
            f"X has {X.shape[1]} columns, but the model was fitted on data with "
            f"{self.n_features_in_} columns, of which {len(columns)} were retained."
        )

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set. X is expected in the original units: any column standardization
        applied during ``fit`` is re-applied here.

        Args:
            X (np.ndarray | pd.DataFrame): new data of shape (n_samples, n_features), in the
                original units of the training data.

        Returns:
            np.ndarray: scores of shape (n_samples, n_components), i.e. the projection of X on
            the first principal components.
        """
        X = self._prepare_input(X)
        if self.location_ is None:
            self.location_ = np.mean(X, axis=0)
        return ((X - self.location_) / self._column_scales) @ self.components_

    def project(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Project the data onto the subspace spanned by the principal components.

        Args:
            X (np.ndarray | pd.DataFrame): data to project, in the original units.

        Returns:
            np.ndarray: the projection of the *centered* data onto the PCA subspace, in the
            original units and of shape (n_samples, n_features_kept). Add ``location_`` to obtain
            the reconstruction of X itself.
        """
        return (self.transform(X) @ self.components_.T) * self._column_scales

    def plot_outlier_map(
        self,
        X: np.ndarray | pd.DataFrame,
        figsize: tuple[int, int] = (4, 4),
        return_distances: bool = False,
    ) -> None | tuple[np.ndarray, np.ndarray, float, float]:
        """Plot Orthogonal distances vs Score distances to identify different types of outliers.

        The score distance (SD) measures how far a case lies from the center *within* the PCA
        subspace, the orthogonal distance (OD) how far it lies *from* that subspace. Cases beyond
        the SD cutoff only are good leverage points, cases beyond the OD cutoff only are
        orthogonal outliers, and cases beyond both are bad leverage points.

        Args:
            X (np.ndarray | pd.DataFrame): data matrix (n x p), in the original units.
            figsize (tuple[int, int], optional): Size of the plot. Defaults to (4, 4).
            return_distances (bool, optional):
                Whether to return the distances and cutoff values. Defaults to False.

        Returns:
            None | tuple: when ``return_distances`` is True, the tuple
            ``(score_distances, orthogonal_distances, score_cutoff, od_cutoff)``.
        """
        X = self._prepare_input(X)
        if self.explained_variance_ is None:
            raise ValueError(
                "The outlier map needs explained_variance_, which is not available on this "
                "fitted model."
            )
        orthogonal_distances = np.linalg.norm((X - self.location_) - self.project(X), axis=1)
        score_distances = np.sqrt(
            np.sum(np.square(self.transform(X)) / self.explained_variance_, axis=1)
        )
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(score_distances, orthogonal_distances)
        ax.set_xlabel("Score distance")
        ax.set_ylabel("Orthogonal distance")
        n_components = getattr(self, "n_components_", None) or self.n_components
        score_cutoff = np.sqrt(float(chi2.ppf(0.975, n_components)))
        od_cutoff = get_od_cutoff(orthogonal_distances)
        ax.axvline(score_cutoff, color="r", linestyle="--")
        ax.axhline(od_cutoff, color="r", linestyle="--")
        if return_distances:
            return (score_distances, orthogonal_distances, score_cutoff, od_cutoff)


def get_od_cutoff(orthogonal_distances: np.ndarray) -> float:
    mcd = UnivariateMCD().fit(orthogonal_distances ** (2 / 3))
    return float(mcd.location + mcd.scale * norm.ppf(0.975)) ** (3 / 2)
