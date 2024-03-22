import numpy as np

from robpy.univariate.base import RobustScaleEstimator
from robpy.utils import weighted_median


class QnEstimator(RobustScaleEstimator):

    def calculate_Qn(self) -> float:
        """
        Implementation of Qn estimator

        [Time-efficient algorithms for two highly robust estimators of scale,
        Christophe Croux and Peter J. Rousseeuw (1992)]
        [Selecting the k^th element in X+Y and X1+...+Xm,
        Donald B. Johnson and Tetsuo Mizoguchi (1978)]

        Args:
            X: univariate data
        Returns:
            scale: robust Qn scale estimator
        """
        n = len(self.X)
        h = n // 2 + 1
        k = h * (h - 1) // 2
        c = 2.219144
        y = np.sort(self.X)
        left = n + 1 - np.arange(n)
        right = np.full(n, n)
        jhelp = n * (n + 1) // 2
        knew = k + jhelp
        nL = jhelp
        nR = n * n
        found = False

        while (nR - nL) > n and (not found):
            weight = right - left + 1
            jhelp = (left + weight // 2).astype(int)
            work = y - y[n - jhelp]
            trial = weighted_median(work, weight)
            P = np.searchsorted(-np.flip(y), trial - y, "left", np.arange(n))
            Q = np.searchsorted(-np.flip(y), trial - y, "right", np.arange(n)) + 1
            if knew <= np.sum(P):
                right = P
                nR = np.sum(P)
            elif knew > (np.sum(Q) - n):
                left = Q
                nL = np.sum(Q) - n
            else:
                Qn = trial
                found = True
        if not found:
            work = []
            for i in range(1, n):
                if left[i] <= right[i]:
                    for jj in range(int(left[i]), int(right[i] + 1)):
                        work.append(y[i] - y[n - jj - 1 + 1])
            k = int(knew - nL)
            Qn = np.partition(np.array(work), k)[:k].max()
        dn = self.get_small_sample_dn(n)
        Qn = dn * c * Qn

        self.Qn = Qn

        return Qn

    def get_small_sample_dn(self, n: int):
        """
        Calculates the correction factor for the Qn estimator
        at small samples [Time-efficient algorithms for two highly robust estimators of scale,
        Christophe Croux and Peter J. Rousseeuw (1992)].
        """
        DNDICT = {
            2: 0.399,
            3: 0.994,
            4: 0.512,
            5: 0.844,
            6: 0.611,
            7: 0.857,
            8: 0.669,
            9: 0.872,
        }
        if n <= 9:
            return DNDICT.get(n)
        elif n % 2 != 0:
            return n / (n + 1.4)
        return n / (n + 3.8)
