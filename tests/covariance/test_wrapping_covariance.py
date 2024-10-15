import numpy as np
from robpy.covariance import WrappingCovariance


def test_fit_sets_object_attributes():
    # given
    X = np.random.rand(100, 2)
    # when
    estimator = WrappingCovariance().fit(X)
    # then
    assert hasattr(estimator, "covariance_")
