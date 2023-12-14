import numpy as np

from scipy.spatial.distance import mahalanobis
from robpy.utils.distance import mahalanobis_distance


def test_mahalanobis_distance_normal():
    # given
    np.random.seed(1)
    mean = np.random.rand(3)
    A = np.random.rand(3, 3)
    cov = A @ A.T
    data = np.random.multivariate_normal(mean=mean, cov=cov, size=20)

    expected_distances = [mahalanobis(x, mean, np.linalg.inv(cov)) for x in data]
    # when
    result_distances = mahalanobis_distance(data, location=mean, covariance=cov)
    # then
    np.testing.assert_allclose(result_distances, expected_distances)
