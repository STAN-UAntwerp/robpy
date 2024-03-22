import pytest
import numpy as np
from robpy.covariance.ogk import OGKEstimator


@pytest.mark.parametrize("n_iterations, reweighting", [(1, False), (2, False), (2, True)])
def test_calculate_covariance(n_iterations, reweighting):
    # given
    X = np.random.normal(size=(100, 5))  # Example data
    estimator = OGKEstimator(n_iterations=n_iterations, reweighting=reweighting)
    expected_shape = (X.shape[1], X.shape[1])

    # when
    covariance_matrix = estimator.calculate_covariance(X)

    # then
    assert covariance_matrix.shape == expected_shape
    # Check if the covariance matrix is symmetric
    assert np.allclose(covariance_matrix, covariance_matrix.T)
    # Check if the calculated covariance matrix is positive semi-definite
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    assert np.all(eigenvalues >= 0)
