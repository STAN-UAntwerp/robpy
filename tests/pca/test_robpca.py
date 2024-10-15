import numpy as np
from robpy.pca import ROBPCA


def test_robpca_sets_components():
    # given
    n = 100
    p = 10
    X = np.random.randn(n, p)
    k = 4
    # when
    pca = ROBPCA(n_components=k).fit(X)
    transformed = pca.transform(X)
    projected = pca.project(X)
    # then
    assert pca.components_.shape == (p, k)
    assert transformed.shape == (n, k)
    assert projected.shape == (n, p)
