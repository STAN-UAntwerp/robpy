import numpy as np
from robpy.univariate.univariateMCD import univariateMCDEstimator


def test_calculate_univariateMCD():
    # given
    X = np.random.randn(1000)
    # when
    estimator = univariateMCDEstimator(X, estimator="univariateMCD")
    estimator.calculate_univariateMCD()
    # then
    assert hasattr(estimator, "MCD_location")
    assert hasattr(estimator, "MCD_variance")
    assert hasattr(estimator, "MCD_raw_location")
    assert hasattr(estimator, "MCD_raw_variance")

    assert isinstance(estimator.MCD_location, float)
    assert isinstance(estimator.MCD_variance, float)
    assert isinstance(estimator.MCD_raw_location, float)
    assert isinstance(estimator.MCD_raw_variance, float)
