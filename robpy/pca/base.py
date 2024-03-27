import numpy as np
import matplotlib.pyplot as plt
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
    
    def transform(self, X):
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
            X = X - self.mean_
        X_transformed = X @ self.components_.T
        return X_transformed
    
    def project(self, X: np.ndarray) -> np.ndarray:
        """Project the data onto the subspace spanned by the principal components
        Args:
            X (np.ndarray): Data to project

        Returns:
            np.ndarray: Projected data
        """
        scores = self.transform(X)
        
        return scores @ self.components_
    
    def plot_outlier_map(self, X: np.ndarray, figsize: tuple[int, int] = (10, 4)):
        orthogonal_distances = np.linalg.norm(X - self.location_ - self.project(X), self.axis=1)
        score_distances = np.sqrt(np.sum(np.square(self.transform(X)) / self.eigen_values_, axis=1))
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(score_distances, orthogonal_distances)
        ax.set_xlabel("Score distance")
        ax.set_ylabel("Orthogonal distance")
        
        ax.axvline(chi2.ppf(0.975, self.n_components_), color="r", linestyle="--")
        ax.axhline(
            (
                np.median(orthogonal_distances) 
                + (median_abs_deviation(orthogonal_distances) * norm.ppf(0.975))
            )**(3/2), 
        color="r", linestyle="--")
        