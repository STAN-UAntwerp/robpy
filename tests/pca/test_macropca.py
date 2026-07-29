import matplotlib
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robpy.pca import MacroPCA  # noqa: E402


@pytest.fixture(scope="module")
def octane():
    return np.load(Path(__file__).parent / "data" / "octane_cellpca_ref.npz")["X"]


@pytest.fixture(scope="module")
def octane_fit(octane):
    return MacroPCA(n_components=2, kmax=3).fit(octane)


def test_macropca_sets_components(octane, octane_fit):
    n, p = octane.shape
    assert octane_fit.components_.shape == (p, 2)
    assert octane_fit.n_components_ == 2
    assert octane_fit.scores_.shape == (n, 2)
    assert octane_fit.full_scores_.shape == (n, 2)
    assert octane_fit.transform(octane).shape == (n, 2)
    assert octane_fit.project(octane).shape == (n, p)


@pytest.mark.parametrize("scale", [True, False])
def test_transform_matches_scores(octane, scale):
    """The base transform must undo the column standardization applied during fit, otherwise it
    disagrees with the scores computed internally."""
    model = MacroPCA(n_components=2, kmax=3, scale=scale).fit(octane)
    # without missing values the NA-imputed data is the original data
    np.testing.assert_allclose(model.X_NA_imputed_, octane, atol=1e-10)
    np.testing.assert_allclose(model.transform(octane), model.scores_, atol=1e-10)


def test_scale_equivariance(octane):
    """Fitting on X with given column scales equals fitting on the pre-scaled data."""
    rng = np.random.default_rng(1)
    s = np.exp(rng.uniform(-1, 1, octane.shape[1]))
    scaled = MacroPCA(n_components=2, kmax=3, scale=s).fit(octane)
    prescaled = MacroPCA(n_components=2, kmax=3, scale=False).fit(octane / s)
    np.testing.assert_allclose(scaled.components_, prescaled.components_, atol=1e-10)
    np.testing.assert_allclose(scaled.location_ / s, prescaled.location_, atol=1e-10)
    np.testing.assert_allclose(
        scaled.transform(octane), prescaled.transform(octane / s), atol=1e-10
    )


@pytest.mark.parametrize("scale", [True, False])
def test_plot_outlier_map(octane, scale):
    model = MacroPCA(n_components=2, kmax=3, scale=scale).fit(octane)
    sd, od, sd_cutoff, od_cutoff = model.plot_outlier_map(octane, return_distances=True)
    assert sd.shape == od.shape == (octane.shape[0],)
    assert sd_cutoff > 0 and od_cutoff > 0
    # the orthogonal distances are expressed in the units of the original data
    np.testing.assert_allclose(
        od, np.linalg.norm((octane - model.location_) - model.project(octane), axis=1)
    )
    # and the score distances are consistent with transform and explained_variance_
    np.testing.assert_allclose(
        sd, np.sqrt(np.sum(model.transform(octane) ** 2 / model.explained_variance_, axis=1))
    )
    plt.close("all")


def test_transform_after_dropped_columns():
    """DataCleaner may drop columns during fit; transform must still accept the original data."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 6))
    X[:, 3] = 1.0  # constant column, dropped as having a bad scale
    model = MacroPCA(n_components=2, kmax=3).fit(X)
    np.testing.assert_array_equal(model.columns_kept_, [0, 1, 2, 4, 5])
    assert model.n_features_in_ == 6
    assert model.components_.shape == (5, 2)
    # both the original and the cleaned data are accepted, and give the same scores
    np.testing.assert_allclose(model.transform(X), model.transform(X[:, model.columns_kept_]))
    assert model.project(X).shape == (60, 5)
    assert model.plot_outlier_map(X, return_distances=True)[0].shape == (60,)
    with pytest.raises(ValueError, match="fitted on data with 6 columns"):
        model.transform(X[:, :4])
    plt.close("all")


def test_transform_returns_array_for_dataframe_input(octane, octane_fit):
    scores = octane_fit.transform(pd.DataFrame(octane))
    assert isinstance(scores, np.ndarray)
    np.testing.assert_allclose(scores, octane_fit.transform(octane))


def test_fit_does_not_mutate_hyperparameters(octane):
    model = MacroPCA(n_components=2, kmax=3)
    model.fit(octane)
    assert model.n_components == 2
    assert model.get_params()["n_components"] == 2
