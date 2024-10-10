import numpy as np
import pytest
import math
from robpy.covariance.cellmcd import CellMCD


def test_fit_sets_object_attributes():
    # given
    X = np.random.normal(0, 1, [1000, 5])
    # when
    estimator = CellMCD()
    estimator.fit(X)
    eigenvalues = np.linalg.eigvals(estimator.covariance_)
    # then
    assert hasattr(estimator, "covariance_")
    assert hasattr(estimator, "location_")
    assert hasattr(estimator, "predictions")
    assert hasattr(estimator, "residuals")
    assert np.all(eigenvalues > estimator.min_eigenvalue)


def test_large_enough_eigenvalues():
    """Test positive definiteness and large enough eigenvalues."""
    # given
    X = np.random.normal(0, 1, [1000, 5])
    # when
    estimator = CellMCD(min_eigenvalue=1e-5)
    estimator.fit(X)
    eigenvalues = np.linalg.eigvals(estimator.covariance_)
    # then
    assert np.all(eigenvalues > estimator.min_eigenvalue)


@pytest.mark.parametrize(
    "alpha, quantile, max_c_steps, min_eigenvalue",
    [
        (0.75, 0.99, 100, 1e-4),
        (0.8, 0.95, 50, 1e-5),
        (0.75, 0.95, 100, 1e-6),
        (0.85, 0.98, 80, 1e-4),
    ],
)
def test_estimator_can_handle_different_settings(alpha, quantile, max_c_steps, min_eigenvalue):
    # given
    X = np.random.normal(0, 1, [1000, 5])
    # when
    estimator = CellMCD(
        alpha=alpha, quantile=quantile, max_c_steps=max_c_steps, min_eigenvalue=min_eigenvalue
    )
    estimator.fit(X)
    # then
    assert hasattr(estimator, "covariance_")
    assert np.all(np.sum(estimator.W, axis=0) > math.floor(X.shape[0] * alpha))
