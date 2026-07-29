import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robpy.pca.spca import PCALocantore  # noqa: E402


def test_spca_sets_components():
    # given
    n = 100
    p = 10
    X = np.random.randn(n, p)
    k = 4
    # when
    pca = PCALocantore(n_components=k).fit(X)
    transformed = pca.transform(X)
    projected = pca.project(X)
    # then
    assert pca.components_.shape == (p, k)
    assert transformed.shape == (n, k)
    assert projected.shape == (n, p)


def test_spca_outlier_map():
    n, p, k = 100, 10, 4
    X = np.random.randn(n, p)
    pca = PCALocantore(n_components=k).fit(X)
    sd, od, sd_cutoff, od_cutoff = pca.plot_outlier_map(X, return_distances=True)
    assert sd.shape == od.shape == (n,)
    assert sd_cutoff > 0 and od_cutoff > 0
    np.testing.assert_allclose(od, np.linalg.norm((X - pca.location_) - pca.project(X), axis=1))
    plt.close("all")
