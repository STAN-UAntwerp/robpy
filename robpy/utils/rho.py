import numpy as np


class BaseRho:
    """
    Base class for robust loss functions.
    """

    def rho(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def psi(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class TukeyBisquare(BaseRho):
    def __init__(self, c: float = 1.56):
        """
        Tukey's bisquare loss function.

        Args:
            c (float, optional):
                Tuning constant controlling the cutoff. Defaults to 1.56.
        """
        self.c = c

    def rho(self, X: np.ndarray) -> np.ndarray:
        return np.where(
            np.abs(X) <= self.c,
            (X**2 / 2) - (X**4 / (2 * self.c**2)) + (X**6 / (6 * self.c**4)),
            self.c**2 / 6,
        )

    def psi(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X) <= self.c, X * (1 - (X / self.c) ** 2) ** 2, 0)


class Huber(BaseRho):
    def __init__(self, b: float = 1.5):
        """
        Huber's loss function.

        Args:
            b (float, optional):
                Threshold between the quadratic and linear regions of loss. Defaults to 1.5.
        """
        self.b = b

    def rho(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X) <= self.b, X**2 / 2, self.b * np.abs(X) - self.b**2 / 2)

    def psi(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.abs(X) <= self.b, X, self.b * np.sign(X))
