import numpy as np
from robpy.pca.spca import PCALocantoreEstimator


def test_spca_sets_components():
    # given
    n = 100
    p = 10
    X = np.random.randn(n, p)
    k = 4
    # when
    pca = PCALocantoreEstimator(n_components=k).fit(X)
    transformed = pca.transform(X)
    projected = pca.project(X)
    # then
    assert pca.components_.shape == (p, k)
    assert transformed.shape == (n, k)
    assert projected.shape == (n, p)
