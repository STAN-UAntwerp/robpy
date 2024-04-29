import pytest
import numpy as np
from robpy.covariance.mcd import FastMCDEstimator


def test_fit_sets_object_attributes():
    # given
    X = np.random.rand(100, 2)
    n_initial_c_steps = 2
    # when
    estimator = FastMCDEstimator(n_initial_c_steps=n_initial_c_steps).fit(X)
    # then
    assert hasattr(estimator, "covariance_")
    assert estimator.best_subset.n_c_steps >= n_initial_c_steps


@pytest.mark.parametrize(
    "h_size, n, p, n_partitions, expected_h_subset_size",
    [
        (None, 100, 2, None, 51),
        (70, 100, 3, None, 70),
        (0.75, 1000, 5, 4, 750),
        (None, 1500, 4, 1, 752),
    ],
)
def test_estimator_can_handle_different_settings(
    h_size, n, p, n_partitions, expected_h_subset_size
):
    # given
    X = np.random.rand(n, p)
    # when
    estimator = FastMCDEstimator(h_size=h_size, n_partitions=n_partitions).fit(X)
    # then
    assert hasattr(estimator, "covariance_")
    assert len(estimator.best_subset.indices) == expected_h_subset_size
