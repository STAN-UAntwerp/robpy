import numpy as np
import math as math

from robpy.univariate.hubers_m_est import UnivariateHuberMEstimator
from scipy.stats import median_abs_deviation

X = np.random.normal(loc=0, scale=1, size=1000)
X[0] = 10000
print(X[:20])

print(UnivariateHuberMEstimator().fit(X).location)
print(np.mean(X))
print(np.median(X))

print(UnivariateHuberMEstimator().fit(X).scale)
print(np.std(X))
print(median_abs_deviation(X, scale="normal"))

# --------------------------------------------------------------------------------------------------

# data = np.loadtxt("/home/sarahleyder/robpy/robpy/data.txt")
# print(data.shape)
# print([UnivariateHuberMEstimator().fit(col).location for col in data.T])  # third variable: wrong
# print([UnivariateHuberMEstimator().fit(col).scale for col in data.T])  # third variable: wrong

# data = np.loadtxt("/home/sarahleyder/robpy/robpy/baseball.txt")
# print(data.shape)
# print([UnivariateHuberMEstimator().fit(col).location for col in data.T])  # third variable: wrong
# print([UnivariateHuberMEstimator().fit(col).scale for col in data.T])

# data = np.loadtxt("/home/sarahleyder/robpy/robpy/testmatrixalcohol.txt")
# print(data.shape)
# print([UnivariateHuberMEstimator().fit(col).location for col in data.T])  # third variable: wrong
# print([UnivariateHuberMEstimator().fit(col).scale for col in data.T])

### cannot seem to use np.where to calculate the wi, errors:
### https://stackoverflow.com/questions/49760092/python-numpy-where-returning-unexpected-warning
