import pytest
import numpy as np
from robpy.covariance.mcd import DetMCDEstimator


def test_fit_sets_object_attributes():
    # given
    X = np.random.rand(100, 2)
    # when
    estimator = DetMCDEstimator().fit(X)
    # then
    assert hasattr(estimator, "covariance_")


@pytest.mark.parametrize(
    "h_size, n, p,  expected_h_subset_size",
    [
        (None, 100, 2, 51),
        (70, 100, 3, 70),
        (0.75, 1000, 5, 750),
        (None, 1500, 4, 752),
    ],
)
def test_estimator_can_handle_different_settings(h_size, n, p, expected_h_subset_size):
    # given
    X = np.random.rand(n, p)
    # when
    estimator = DetMCDEstimator(h_size=h_size).fit(X)
    # then
    assert hasattr(estimator, "covariance_")
    assert len(estimator.best_subset.indices) == expected_h_subset_size