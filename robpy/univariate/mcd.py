import numpy as np

from robpy.univariate.base import RobustScaleEstimator
from scipy.stats import chi2, gamma


class UnivariateMCDEstimator(RobustScaleEstimator):
    def __init__(self, h_size: float | int | None = None, consistency_correction: bool = True):
        """
        Implementation of univariate MCD (Hubert & Debruyne, 2009)

        Args:
            h_size (float or int, optional): size of the h subset.
              If an integer between n/2 and n is passed, it is interpreted as an absolute value.
              If a float  between 0.5 and 1 is passed, it is interpreted as a proportation
              of n (the training set size).
              If None, it is set to floor(n/2) + 1.
              Defaults to None.
            consistency_correction (boolean, optional):
              whether the estimates should be consistent at the normal model.
              Defaults to True.

        References:
            Hubert, M., & Debruyne, M. (2010). Minimum covariance determinant.
              Wiley interdisciplinary reviews: Computational statistics, 2(1), 36-43.
        """
        super().__init__()
        self.h_size = h_size
        self.consistency_correction = consistency_correction

    def _calculate(self, X: np.ndarray):
        self._set_h_size(X)
        if self.h_size == X.shape[0]:
            self.raw_location_ = self.location_ = X.mean()
            self.raw_scale_ = self.scale_ = X.std()
            self.raw_variance_ = self.variance_ = np.square(self.raw_scale_)
            return

        self.raw_variance_, self.raw_location_ = self._get_raw_estimates(X)
        self.variance_, self.location_ = self._reweighting(X)
        self.raw_scale_ = np.sqrt(self.raw_variance_)
        self.scale_ = np.sqrt(self.variance_)

    def _get_raw_estimates(self, X: np.ndarray):
        n = len(X)
        X = np.sort(X)
        var_best = np.inf
        index_best = 1
        for i in range(n - self.h_size + 1):
            var_new = np.var(X[i : (i + self.h_size)])
            if var_new < var_best:
                var_best = var_new
                index_best = i
        raw_var = var_best
        raw_loc = np.mean(X[index_best : (index_best + self.h_size)])
        if self.consistency_correction:
            # [Minimum covariance determinant, Mia Hubert & Michiel Debruyne (2009)]
            raw_var = raw_var * (self.h_size / n) / chi2.cdf(chi2.ppf(self.h_size / n, df=1), df=3)
        return raw_var, raw_loc

    def _reweighting(self, X: np.ndarray):
        distances = (X - self.raw_location_) ** 2 / self.raw_variance_
        mask = distances < chi2.ppf(0.975, df=1)
        loc = np.mean(X[mask])
        var = np.var(X[mask])
        if self.consistency_correction:
            # Croux & Haesbroeck (1999)
            delta = np.sum(mask) / len(X)
            var = var * delta * np.reciprocal(gamma.cdf(chi2.ppf(delta, df=1) / 2, a=3 / 2))
        return var, loc

    def _set_h_size(self, X: np.ndarray):
        n = len(X)
        if self.h_size is None:
            self.h_size = n // 2 + 1
        elif isinstance(self.h_size, int) and (n / 2 <= self.h_size <= n):
            pass
        elif (isinstance(self.h_size, float) and (0.5 <= self.h_size <= 1)) or self.h_size == 1:
            self.h_size = int(self.h_size * n)
        else:
            raise ValueError(
                f"h_size must be an integer between n/2 and n or a float between 0.5 and 1 "
                f"but received {self.h_size}"
            )
