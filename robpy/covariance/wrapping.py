import numpy as np

from robpy.preprocessing.utils import wrapping_transformation
from robpy.covariance.base import RobustCovariance


class WrappingCovariance(RobustCovariance):
    def __init__(
        self,
        b: float = 1.5,
        c: float = 4.0,
        q1: float = 1.540793,
        q2: float = 0.8622731,
        store_precision: bool = True,
        assume_centered: bool = False,
    ):
        """
        Covariance estimator based on the wrapping function proposed in Raymaekers, J., &
        Rousseeuw, P. J. (2021).

        The wrapping transformation is defined as follows:

        .. math::

            \\Psi_{b, c}(z) =
            \\begin{cases}
            z & \\text{if } \\  0 \\leq |z| < b, \\\\
            q_1 \\tanh\\left(q_2 (c - |z|)\\right) \\mathrm{sign}(z) & \\text{if } \\  b \\leq |z|
            \\leq c,\\\\
            0  & \\text{if } \\   c < |z|.
            \\end{cases}

        The data is first scaled using the median and the MAD before applying the transformation.

        Next, the robust covariance of X is computed from the classical covariance on the
        transformed data:

        .. math::

            Cov(Median(X) + MAD(X) * \\Psi_{b, c}(X - Median(X) / MAD(X)))

        Args:
            b (float, optional):
                Lower cutoff. Defaults to 1.5.
            c (float, optional):
                Upper cutoff. Defaults to 4.0.
            q1 (float, optional):
                Transformation parameter (see formula). Defaults to 1.540793.
            q2 (float, optional):
                Transformation parameter (see formula). Defaults to 0.8622731.
            store_precision (bool, optional):
                Whether to store the precision matrix. Defaults to True.
            assume_centered (bool, optional):
                Whether the data is already centered. Defaults to False.

        References:
            - Raymaekers, J., & Rousseeuw, P. J. (2021). Fast robust correlation for
              high-dimensional data. Technometrics, 63(2), 184-198.

        """
        super().__init__(store_precision=store_precision, assume_centered=assume_centered)
        self.b = b
        self.c = c
        self.q1 = q1
        self.q2 = q2

    def calculate_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the covariance matrix of the data X after applying the wrapping transformation.

        Args:
            X (np.ndarray): Data to calculate the covariance matrix of, must have shape (n, p).

        Returns:
            np.ndarray: Robust covariance matrix of X.
        """
        X_transformed = wrapping_transformation(
            X, b=self.b, c=self.c, q1=self.q1, q2=self.q2, rescale=True
        )
        self.correlation_ = np.corrcoef(X_transformed, rowvar=False)
        return np.cov(X_transformed, rowvar=False)
