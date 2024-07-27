import numpy as np

from robpy.preprocessing.utils import wrapping_transformation
from robpy.covariance.base import RobustCovarianceEstimator


class WrappingCovarianceEstimator(RobustCovarianceEstimator):
    """Covariance estimator based on the wrapping function proposed in
    Jakob Raymaekers & Peter J. Rousseeuw (2021)

    The wrapping transformation is defined as follows:

    .. math::

        \\Psi_{b, c}(z) =
        \\begin{cases}
          z & if \\  0 \\leq |z| < b \\\\
          q_1 \\tanh\\left(q_2 (c - |z|)\\right) \\mathrm{sign}(z) & if \\  b \\leq |z| \\leq c \\\\
          0  & if \\   c < |z|
        \\end{cases}

    Data is first scaled using median and MAD before applying the transformation.

    The (standard) covariance is subsequently estimated on the rescaled data
    Cov(X) = Cov(Median(X) + MAD(X) * phi(X - Median(X) / MAD(X)))

    References:
        - Jakob Raymaekers & Peter J. Rousseeuw (2021)
          Fast Robust Correlation for High-Dimensional Data,
          Technometrics, 63:2, 184-198, DOI: 10.1080/00401706.2019.1677270
    """

    def __init__(
        self,
        b: float = 1.5,
        c: float = 4.0,
        q1: float = 1.540793,
        q2: float = 0.8622731,
        rescale: bool = True,
        store_precision: bool = True,
        assume_centered: bool = False,
    ):
        """
        Args:
            X: data to be transformed, must have shape (N, D)
            b: lower cutoff
            c: upper cutoff
            q1, q2: transformation parameters (see formula)
            rescale: whether to rescale the wrapped data so the robust location and scale
                    of the transformed data are the same as the original data
        """
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.b = b
        self.c = c
        self.q1 = q1
        self.q2 = q2
        self.rescale = rescale

    def calculate_covariance(self, X: np.ndarray) -> np.ndarray:
        """Calculate the covariance matrix of the data X after applying the wrapping transformation

        Args:
            X: data to calculate the covariance matrix of, must have shape (N, D)

        Returns:
            robust covariance matrix of X
        """
        X_transformed = wrapping_transformation(
            X, b=self.b, c=self.c, q1=self.q1, q2=self.q2, rescale=self.rescale
        )
        self.correlation_ = np.corrcoef(X_transformed, rowvar=False)
        return np.cov(X_transformed, rowvar=False)
