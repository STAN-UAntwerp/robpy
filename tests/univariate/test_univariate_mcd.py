import numpy as np
from robpy.univariate import UnivariateMCD


def test_mcd_sets_all_attributes():
    # given
    X = np.random.randn(1000)
    # when
    estimator = UnivariateMCD().fit(X)
    # then
    assert isinstance(estimator.location, float)
    assert isinstance(estimator.scale, float)
