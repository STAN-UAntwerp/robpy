import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from sklearn.decomposition._base import _BasePCA
from scipy.stats import chi2, norm, median_abs_deviation


class RobustPCAEstimator(_BasePCA):
    def __init__(self, *, n_components: int | None = None):
        """Base class for robust PCA estimators

        Args:
            n_components (int | None, optional):
                Number of components to select. If None, it is set during fit to min (X.shape)

        """
        self.n_components = n_components

    @abstractmethod
    def fit(self, X: np.ndarray):
        """Fit the robust PCA model to the data
        Args:
            X (np.ndarray): Data to fit the model to
        """
        pass

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """

        if self.location_ is not None:
            self.location_ = np.mean(X, axis=0)
        return (X - self.location_) @ self.components_

    def project(self, X: np.ndarray) -> np.ndarray:
        """Project the data onto the subspace spanned by the principal components
        Args:
            X (np.ndarray): Data to project

        Returns:
            np.ndarray: Projected data
        """
        return self.transform(X) @ self.components_.T

    def plot_outlier_map(
        self, X: np.ndarray, figsize: tuple[int, int] = (10, 4), return_distances: bool = False
    ) -> None | tuple[np.ndarray, np.ndarray, float, float]:
        """Plot Orthogonal distances vs Score distances to identify different types of outliers

        Args:
            X (np.ndarray): Data  matrix (n x p)
            figsize (tuple[int, int], optional): Size of the plot. Defaults to (10, 4).
            return_distances (bool, optional):
                Whether to return the distances and cutoff values. Defaults to False.
        """
        orthogonal_distances = np.linalg.norm((X - self.location_) - self.project(X), axis=1)
        score_distances = np.sqrt(
            np.sum(np.square(self.transform(X)) / self.explained_variance_, axis=1)
        )
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(score_distances, orthogonal_distances)
        ax.set_xlabel("Score distance")
        ax.set_ylabel("Orthogonal distance")
        score_cutoff = float(chi2.ppf(0.975, self.n_components))
        od_cutoff = get_od_cutoff(orthogonal_distances)
        ax.axvline(score_cutoff, color="r", linestyle="--")
        ax.axhline(od_cutoff, color="r", linestyle="--")
        if return_distances:
            return (score_distances, orthogonal_distances, score_cutoff, od_cutoff)


def get_od_cutoff(orthogonal_distances: np.ndarray) -> float:
    # TODO: replace median and mad by univariate MCD
    return float(
        np.median(orthogonal_distances)
        + (median_abs_deviation(orthogonal_distances) * norm.ppf(0.975))
    ) ** (3 / 2)
