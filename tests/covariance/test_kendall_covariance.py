import numpy as np
from robpy.covariance.kendall import KendallTauEstimator


def test_fit_sets_object_attributes():
    # given
    X = np.random.rand(100, 2)
    # when
    estimator = KendallTauEstimator().fit(X)
    # then
    assert hasattr(estimator, "covariance_")
    assert hasattr(estimator, "correlation_")
