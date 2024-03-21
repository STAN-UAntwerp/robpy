import numpy as np
import math as math


class RobustScaleEstimator:
    def __init__(self, X: np.array, estimator: str):
        """
        Args:
            X: univariate data
            estimator: "Qn" or "univariateMCD"
        """

        self.X = X
        if estimator == "Qn":
            self.Qn = None
        elif estimator == "univariateMCD":
            self.MCD_variance = None
            self.MCD_location = None
            self.MCD_raw_variance = None
            self.MCD_raw_location = None
