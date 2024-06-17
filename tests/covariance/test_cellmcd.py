import pytest
import numpy as np
from robpy.covariance.cellmcd import CellMCDEstimator


def test_fit_sets_object_attributes():
    # given
    X = np.random.normal(0, 1, [1000, 5])
    # when
    max_c_steps = 100
    estimator = CellMCDEstimator(max_c_steps=max_c_steps)
    estimator.calculate_covariance(X)
    # then
    assert hasattr(estimator, "covariance_")
    assert hasattr(estimator, "location_")
    assert hasattr(estimator, "predictions")
    assert hasattr(estimator, "residuals")
    assert estimator.steps < max_c_steps
