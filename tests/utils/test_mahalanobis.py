import numpy as np

from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
from robpy.utils import mahalanobis_distance


def test_mahalanobis_distance_normal():
    # given
    np.random.seed(1)
    true_mean = np.random.rand(3)
    A = np.random.rand(3, 3)
    true_cov = A @ A.T
    data = np.random.multivariate_normal(mean=true_mean, cov=true_cov, size=20)

    observed_mean = data.mean(axis=0)
    observed_inv_cov = np.linalg.inv(np.cov(data, rowvar=False))
    expected_distances = [mahalanobis(x, observed_mean, observed_inv_cov) for x in data]
    # when
    result_distances = mahalanobis_distance(data, robust=False)
    # then
    np.testing.assert_allclose(result_distances, expected_distances)


def test_mahalanobis_distance_robust():
    # given
    np.random.seed(1)
    true_mean = np.random.rand(3)
    A = np.random.rand(3, 3)
    true_cov = A @ A.T
    data = np.random.multivariate_normal(mean=true_mean, cov=true_cov, size=20)

    random_state = 101
    mcd = MinCovDet(random_state=random_state).fit(data)
    observed_mean = mcd.location_
    observed_inv_cov = np.linalg.inv(mcd.covariance_)
    expected_distances = [mahalanobis(x, observed_mean, observed_inv_cov) for x in data]
    # when
    result_distances = mahalanobis_distance(data, robust=True, random_state=random_state)
    # then
    np.testing.assert_allclose(result_distances, expected_distances)
