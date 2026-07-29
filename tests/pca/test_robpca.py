import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robpy.pca import ROBPCA  # noqa: E402


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


def test_robpca_outlier_map():
    n, p, k = 100, 10, 4
    X = np.random.randn(n, p)
    pca = ROBPCA(n_components=k).fit(X)
    sd, od, sd_cutoff, od_cutoff = pca.plot_outlier_map(X, return_distances=True)
    assert sd.shape == od.shape == (n,)
    assert sd_cutoff > 0 and od_cutoff > 0
    np.testing.assert_allclose(od, np.linalg.norm((X - pca.location_) - pca.project(X), axis=1))
    plt.close("all")


def test_robpca_transform_returns_array_for_dataframe_input():
    X = np.random.randn(100, 10)
    pca = ROBPCA(n_components=4).fit(X)
    transformed = pca.transform(pd.DataFrame(X))
    assert isinstance(transformed, np.ndarray)
    np.testing.assert_allclose(transformed, pca.transform(X))
