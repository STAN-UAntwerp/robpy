import numpy as np
import math as math
from robpy.preprocessing import Qn, weighted_median, weighted_median2
from robpy import preprocessing


X = np.array([1, 2, 4, 7, 10, 15, 17])
w = np.array([1, 1, 1, 3, 3, 3, 3])

print(weighted_median(X, w))  # yours always returns 1 element, not mean of 2 elements
print(weighted_median2(X, w))
