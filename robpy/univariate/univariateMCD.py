import numpy as np

from robpy.univariate.base import RobustScaleEstimator
from scipy.stats import chi2, gamma


class UnivariateMCDEstimator(RobustScaleEstimator):
    def calculate_univariateMCD(
        self, h_size: float | int | None = None, consistency_correction=True
    ) -> np.array:
        """
        Implementation of univariate MCD

        [Minimum covariance determinant, Mia Hubert & Michiel Debruyne (2009)]

        Args:
            X: univariate data
            h_size: parameter determining the size of the h-subset. Defaults to floor(n/2) + 1.
            consistency_correction: whether the estimates should be consistent at the normal model

        Returns:
            raw_var: raw variance estimate
            raw_loc: raw location estimate
            var: reweigthed variance estimate
            loc: reweigthed location estimate
        """
        n = len(self.X)
        if h_size is None:
            h_size = n // 2 + 1
        elif h_size == 1:
            self.MCD_raw_location = self.MCD_location = np.mean(self.X)
            self.MCD_raw_variance = self.MCD_variance = np.var(self.X)
            return np.var(self.X)
        elif isinstance(h_size, int) and (1 < h_size <= n):
            pass
        elif isinstance(h_size, float) and (0 < h_size < 1):
            h_size = int(h_size * n)
        else:
            raise ValueError(
                f"h_size must be an integer > 1 and <= n or a float between 0 and 1 "
                f"but received {h_size}"
            )

        raw_var, raw_loc = self.get_raw_estimates(self.X, h_size, n, consistency_correction)
        var, loc = self.reweighting(self.X, raw_var, raw_loc, n, consistency_correction)

        self.MCD_raw_location = raw_loc
        self.MCD_raw_variance = raw_var
        self.MCD_location = loc
        self.MCD_variance = var

        return var

    def get_raw_estimates(self, X: np.array, h_size, n, consistency_correction):
        X = np.sort(self.X)
        var_best = np.inf
        index_best = 1
        for i in range(n - h_size + 1):
            var_new = np.var(X[i : (i + h_size)])
            if var_new < var_best:
                var_best = var_new
                index_best = i
        raw_var = var_best
        raw_loc = np.mean(X[index_best : (index_best + h_size)])
        if consistency_correction:
            """[Minimum covariance determinant, Mia Hubert & Michiel Debruyne (2009)]"""
            raw_var = raw_var * (h_size / n) / chi2.cdf(chi2.ppf(h_size / n, df=1), df=3)
        return raw_var, raw_loc

    def reweighting(self, X: np.array, raw_var, raw_loc, n, consistency_correction):
        distances = (X - raw_loc) ** 2 / raw_var
        mask = distances < chi2.ppf(0.975, df=1)
        loc = np.mean(X[mask])
        var = np.var(X[mask])
        if consistency_correction:
            """[Influence function and efficiency of the MCD scatter matrix estimator,
            Christophe Croux & Gentiane Haesbroeck (1999)]"""
            delta = np.sum(mask) / n
            var = var * delta * np.reciprocal(gamma.cdf(chi2.ppf(delta, df=1) / 2, a=3 / 2))
        return var, loc
