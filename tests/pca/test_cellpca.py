import matplotlib
import numpy as np
import pytest
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from robpy.pca import CellPCA, MacroPCA, select_k  # noqa: E402
from robpy.pca.cellpca import _up_uv_geninv, _normal_v  # noqa: E402


@pytest.fixture(scope="module")
def octane_ref():
    data = np.load(Path(__file__).parent / "data" / "octane_cellpca_ref.npz")
    return {k: data[k] for k in data.files}


def _align_columns(A, B):
    """Flip the sign of each column of A to agree with B (loadings are sign-ambiguous)."""
    A = A.copy()
    for j in range(A.shape[1]):
        if np.sign(A[np.argmax(np.abs(A[:, j])), j]) != np.sign(B[np.argmax(np.abs(B[:, j])), j]):
            A[:, j] *= -1
    return A


@pytest.fixture(scope="module")
def octane_fit(octane_ref):
    return CellPCA(n_components=2).fit(octane_ref["X"])


def test_cellpca_matches_r_on_octane(octane_fit, octane_ref):
    # Fitted values, residuals and the center match R closely; the small remaining differences
    # stem from robpy's DDC-based MacroPCA initialization (see the CellPCA docstring).
    fit, ref = octane_fit, octane_ref
    sd = ref["X"].std()
    assert np.median(np.abs(fit.fitted_ - ref["Xfit"])) < 1e-3 * sd
    assert np.abs(fit.location_ - ref["mu"]).max() < 1e-2 * sd
    assert np.abs(_align_columns(fit.components_, ref["V"]) - ref["V"]).max() < 5e-3
    assert np.abs(fit.row_residuals_scaled_ - ref["rowres"]).max() < 0.1
    # first eigenvalue matches R; see docstring re: the V_est MCD rotation for the rest
    assert abs(fit.explained_variance_[0] - ref["eig"][0]) < 0.05 * ref["eig"][0]


def test_macropca_initializes_cellpca(octane_ref):
    macro = MacroPCA(n_components=2, kmax=3, scale=False).fit(octane_ref["X"])
    assert macro.components_.shape == (octane_ref["X"].shape[1], 2)
    assert macro.full_scores_.shape == (octane_ref["X"].shape[0], 2)


def test_cellpca_fit_is_deterministic(octane_fit, octane_ref):
    refit = CellPCA(n_components=2).fit(octane_ref["X"])
    np.testing.assert_allclose(refit.fitted_, octane_fit.fitted_)


def test_cellpca_fit_sets_attributes(octane_fit):
    for attr in [
        "components_",
        "location_",
        "scores_",
        "fitted_",
        "imputed_",
        "residuals_scaled_",
        "row_residuals_scaled_",
        "explained_variance_",
    ]:
        assert hasattr(octane_fit, attr)
    assert octane_fit.components_.shape == (226, 2)
    assert octane_fit.scores_.shape == (39, 2)


def test_predict_is_consistent_on_training_data(octane_fit, octane_ref):
    pred = octane_fit.predict(octane_ref["X"])
    # clean cells reproduce the training fit
    assert np.median(np.abs(pred["Xfit"] - octane_fit.fitted_)) < 1e-6


def test_transform_matches_scores_projection(octane_fit, octane_ref):
    scores = octane_fit.transform(octane_ref["X"])
    assert scores.shape == (39, 2)


def test_up_uv_geninv_and_normal_v_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, 5))
    mu = rng.standard_normal(5)
    U = rng.standard_normal((8, 2))
    V = rng.standard_normal((5, 2))
    W = rng.uniform(0.2, 1, (8, 5))
    Uo, Vo = _up_uv_geninv(X, mu, U, V, W, W, update_V=True)
    assert Uo.shape == (8, 2) and Vo.shape == (5, 2)
    U2, V2 = _normal_v(U, V)
    # V2 should be orthonormal
    np.testing.assert_allclose(V2.T @ V2, np.eye(2), atol=1e-10)


def test_select_k_runs(octane_ref):
    # candidates include k=1, which exercises the univariate MCD path
    result = select_k(octane_ref["X"], [1, 2], selection="fev")
    assert result["k"] in (1, 2)
    assert len(result["objs"]) == 3  # v0 + 2 candidates


def test_row_residual_cutoffs(octane_fit):
    c1, c2 = octane_fit.row_residual_cutoffs()
    assert c1 > 0 and c2 == pytest.approx(1.5 * c1)


def test_distances_keys(octane_fit):
    d = octane_fit.distances()
    assert set(d) == {"d1", "d2", "d4", "sd_fit", "sd_x", "U_orig"}
    assert d["sd_fit"].shape == (39,)


@pytest.mark.parametrize(
    "plot_call",
    [
        lambda m: m.cell_map(),
        lambda m: m.total_deviation_plot(),
        lambda m: m.enhanced_outlier_map(),
        lambda m: m.drop_plot(),
        lambda m: m.functional_cell_map([2, 11, 14]),
        lambda m: m.silhouette_plot(),
    ],
)
def test_plots_render(octane_fit, plot_call):
    ax = plot_call(octane_fit)
    assert ax is not None
    plt.close("all")
