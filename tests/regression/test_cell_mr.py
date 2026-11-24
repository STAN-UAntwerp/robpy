"""Tests of cellPCA, cellRCov and cellMR against the authors' R code
(tests/reference/cellMR_reference.R)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

import robpy.pca.cellpca as cellpca
from robpy.covariance import CellRCov
from robpy.covariance.utils.robustbase_mcd import MCDResult
from robpy.pca import CellPCA
from robpy.pca.cellpca import _macropca, rho_tanh, scale_tanh, weight_tanh
from robpy.regression import CellMRRegression
from robpy.regression.cell_mr import cellmr_tuning

REFERENCE = Path(__file__).parents[1] / "reference"


def _reference(name: str) -> np.ndarray:
    return pd.read_csv(REFERENCE / "expected" / f"{name}.csv").to_numpy()


def _data(name: str = "cell_mr") -> np.ndarray:
    return pd.read_csv(REFERENCE / "data" / f"{name}.csv").to_numpy()


def _r_mcd(name: str) -> MCDResult:
    return MCDResult(
        center=_reference(f"{name}_center")[:, 0],
        cov=_reference(f"{name}_cov"),
        raw_center=_reference(f"{name}_raw_center")[:, 0],
        raw_cov=_reference(f"{name}_raw_cov"),
        crit=np.nan,
        best=np.array([]),
    )


def _use_r_mcd(monkeypatch, *names: str):
    """Use the MCD of the scores of the R code (rrcov::CovMcd is random), in call order."""
    results = [_r_mcd(name) for name in names]
    monkeypatch.setattr(cellpca, "_mcd_of_scores", lambda *args, **kwargs: results.pop(0))


def _standardized() -> tuple[np.ndarray, np.ndarray]:
    scales = _reference("cellpca_scales")[:, 0]
    return _data() / scales, _data("cell_mr_new") / scales


def test_tanh_functions():
    x = np.array([0.0, 1.0, -1.4, 2.0, -3.9, 4.5, np.nan])
    w = weight_tanh(x)
    np.testing.assert_allclose(w[:3], 1.0)
    assert w[5] == 0.0 and np.isnan(w[6])
    rho = rho_tanh(x)
    np.testing.assert_allclose(rho[1], 0.5)
    np.testing.assert_allclose(rho[5], 3.7622126668, rtol=1e-7)  # the constant of the R code
    rng = np.random.default_rng(0)
    assert abs(scale_tanh(rng.normal(size=20000)) - 1) < 0.03


def test_macropca_matches_r():
    Z, _ = _standardized()
    model = _macropca(Z, 2)
    np.testing.assert_allclose(model.center, _reference("macropca_center")[:, 0], atol=1e-8)
    np.testing.assert_allclose(model.loadings, _reference("macropca_loadings"), atol=1e-8)
    np.testing.assert_allclose(model.scores, _reference("macropca_scores"), atol=1e-7)


def test_cellpca_matches_r_given_its_mcd(monkeypatch):
    # given: the MCD of the scores of the R code
    _use_r_mcd(monkeypatch, "cellpca_mcd")
    Z, Z_new = _standardized()
    # when
    model = CellPCA(2).fit(Z)
    # then
    np.testing.assert_allclose(model.scores_, _reference("cellpca_U"), atol=1e-7)
    np.testing.assert_allclose(model.components_, _reference("cellpca_V"), atol=1e-8)
    np.testing.assert_allclose(model.location_, _reference("cellpca_mu")[:, 0], atol=1e-8)
    np.testing.assert_allclose(
        model.explained_variance_, _reference("cellpca_eigenvalues")[:, 0], rtol=1e-7
    )
    np.testing.assert_allclose(model.weights_cellwise_, _reference("cellpca_wc"), atol=1e-8)
    np.testing.assert_allclose(model.weights_rowwise_, _reference("cellpca_wr")[:, 0], atol=1e-8)
    np.testing.assert_allclose(model.X_imputed_, _reference("cellpca_Ximp"), atol=1e-7)
    np.testing.assert_allclose(
        model.rowwise_residuals_, _reference("cellpca_rowres")[:, 0], atol=1e-8
    )
    # new rows (cellPCA_predict)
    np.testing.assert_allclose(model.predict_scores(Z_new), _reference("cellpca_new_U"), atol=1e-7)
    np.testing.assert_allclose(model.impute(Z_new), _reference("cellpca_new_Ximp"), atol=1e-7)


def test_cellpca_with_its_own_mcd_spans_the_same_subspace():
    Z, _ = _standardized()
    model = CellPCA(2).fit(Z)
    V_r = _reference("cellpca_V")
    # same subspace and fit; the axes inside it come from the MCD of the scores
    np.testing.assert_allclose(model.components_ @ model.components_.T, V_r @ V_r.T, atol=1e-6)
    np.testing.assert_allclose(model.X_imputed_, _reference("cellpca_Ximp"), atol=1e-6)


def test_cellmr_matches_r_given_its_mcd(monkeypatch):
    # given: the MCDs of the scores of the R code (joint model, then the model of X)
    _use_r_mcd(monkeypatch, "cellmr_mcd_joint", "cellmr_mcd_X")
    D = _data()
    X, Y = D[:, :8], D[:, 8:]
    # when
    model = CellMRRegression(n_components=2, ridge=0.1, n_components_X=2, shrinkage_X=0.5)
    model.fit(X, Y)
    # then
    np.testing.assert_allclose(model.coef_, _reference("cellmr_B"), atol=1e-8)
    np.testing.assert_allclose(model.intercept_, _reference("cellmr_b0")[:, 0], atol=1e-8)
    np.testing.assert_allclose(model.covariance_, _reference("cellmr_S"), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(model.location_, _reference("cellmr_mu")[:, 0], atol=1e-8)
    np.testing.assert_allclose(model.error_covariance_reg_, _reference("cellmr_S_eps"), atol=1e-8)
    np.testing.assert_allclose(model.X_imputed_, _reference("cellmr_Ximp"), atol=1e-7)
    np.testing.assert_allclose(model.fitted_values_, _reference("cellmr_Yhat"), atol=1e-7)
    np.testing.assert_allclose(
        model.cellrcov_X_.covariance_, _reference("cellmr_S_x"), rtol=1e-7, atol=1e-8
    )
    # new rows: impute X with the cellPCA of X, then predict
    X_new = _data("cell_mr_new")[:, :8]
    scales = model.cellrcov_X_.scales_
    np.testing.assert_allclose(
        model.impute(X_new) / scales, _reference("cellmr_new_Ximp_std"), atol=1e-7
    )


def test_cellmr_with_its_own_mcd_is_close_to_r():
    D = _data()
    model = CellMRRegression(n_components=2, ridge=0.1).fit(D[:, :8], D[:, 8:])
    # B depends on the MCD only through the covariance of the subspace, which is invariant
    np.testing.assert_allclose(model.coef_, _reference("cellmr_B"), atol=1e-6)


def test_cellmr_is_robust_to_cellwise_outliers():
    rng = np.random.default_rng(3)
    n = 200
    F = rng.normal(size=(n, 2))
    X = F @ rng.normal(size=(2, 6)) + 0.3 * rng.normal(size=(n, 6))
    B = rng.normal(size=(6, 2))
    Y = X @ B + 0.3 * rng.normal(size=(n, 2))
    X_bad = X.copy()
    cells = rng.random(X.shape) < 0.05
    X_bad[cells] += 10.0
    model = CellMRRegression(n_components=3, ridge=1e-6, n_components_X=2).fit(X_bad, Y)
    ols = np.linalg.lstsq(np.column_stack([np.ones(n), X_bad]), Y, rcond=None)[0][1:]
    error_cellmr = np.abs(model.predict(X) - X @ B).mean()
    error_ols = np.abs(X @ ols - X @ B).mean()
    assert error_cellmr < error_ols / 2


def test_cellrcov_and_input_validation():
    D = _data()
    cov = CellRCov(n_components=2, shrinkage=0.5).fit(D)
    assert cov.covariance_.shape == (11, 11)
    assert np.all(np.linalg.eigvalsh(cov.covariance_) > 0)
    with pytest.raises(NotFittedError):
        CellMRRegression().predict(D[:, :8])
    model = CellMRRegression(n_components=2).fit(D[:, :8], D[:, 8:])
    with pytest.raises(ValueError, match="missing values"):
        model.predict(D[:5, :8])  # contains nan and there is no model of X
    with pytest.raises(ValueError, match="different number of rows"):
        CellMRRegression().fit(D[:, :8], D[:-1, 8:])


def test_cellmr_tuning_runs():
    D = _data()
    out = cellmr_tuning(
        D[:, :8], D[:, 8:], k_grid=[1, 2], ridge_grid=[0.01, 1.0], k_grid_X=[2], n_folds=4
    )
    assert out["best"]["n_components"] in (1, 2)
    assert out["best"]["ridge"] in (0.01, 1.0)
    assert out["n_components_X"] == 2
    assert out["cv_table"].shape == (4, 3)
