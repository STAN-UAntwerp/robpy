"""Tests of cellBoot against the authors' R code (tests/reference/cellBoot_reference.R)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import robpy.pca.cellpca as cellpca
import robpy.regression.cell_boot as cell_boot
from robpy.covariance import CellRCov
from robpy.covariance.utils.robustbase_mcd import MCDResult
from robpy.regression import cellboot

REFERENCE = Path(__file__).parents[1] / "reference"


def _reference(name: str) -> np.ndarray:
    return pd.read_csv(REFERENCE / "expected" / f"{name}.csv").to_numpy()


def _data():
    D = pd.read_csv(REFERENCE / "data" / "cell_mr.csv").to_numpy()
    return D[:, :8], D[:, 8:]


def test_weighted_estimate_of_the_observed_data_matches_r(monkeypatch):
    # given: the MCD of the cellPCA scores of the R code (rrcov::CovMcd is random)
    mcd = MCDResult(
        _reference("cellboot_mcd_center")[:, 0],
        _reference("cellboot_mcd_cov"),
        _reference("cellboot_mcd_raw_center")[:, 0],
        _reference("cellboot_mcd_raw_cov"),
        np.nan,
        np.array([]),
    )
    monkeypatch.setattr(cellpca, "_mcd_of_scores", lambda *args, **kwargs: mcd)
    X, Y = _data()
    Z = np.column_stack([X, Y])
    # when: the deterministic first steps of cellBoot
    cov = CellRCov(n_components=2, max_components=10, regularize=True, shrinkage=0.0).fit(Z)
    Zs = Z / cov.scales_
    model = cell_boot._build_model(cov, Zs, 11, 2)
    M = (~np.isnan(Z)).astype(float)
    imputed = cell_boot._ddc_weighted_impute(Zs, model)
    first = cell_boot._subspace_weights(Zs, imputed, model, M=M, scales="estimate")
    scales = (first["scale_1"], first["scale_2"])
    S, mu, out = cell_boot._weighted_covariance(Zs, model, M, scales, cov.scales_, 0.0)
    # then
    np.testing.assert_allclose(imputed, _reference("cellboot_Ximp"), atol=1e-8)
    np.testing.assert_allclose(out["weights_r"], _reference("cellboot_weights_r")[:, 0], atol=1e-8)
    np.testing.assert_allclose(S, _reference("cellboot_S_obs"), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(mu, _reference("cellboot_mu_obs")[:, 0], atol=1e-8)


@pytest.fixture(scope="module")
def bootstrap():
    X, Y = _data()
    return cellboot(X, Y, n_components=2, ridge=0.1, n_bootstrap=100, n_jobs=2)


def test_cellboot_is_close_to_r(bootstrap):
    # the bias correction and the bootstrap are Monte Carlo procedures, so only closeness is
    # expected (with 200 bootstrap samples the interval widths agree to a median ratio of 0.98)
    np.testing.assert_allclose(bootstrap.coef, _reference("cellboot_slopes"), atol=0.08)
    np.testing.assert_allclose(
        bootstrap.intercept, _reference("cellboot_intercepts")[:, 0], atol=0.08
    )
    ours = bootstrap.confidence_intervals
    theirs = _reference("cellboot_intervals")
    ratio = (ours[:, 1] - ours[:, 0]) / (theirs[:, 1] - theirs[:, 0])
    assert 0.8 < np.median(ratio) < 1.25
    assert np.corrcoef(ours[:, 0], theirs[:, 0])[0, 1] > 0.98


def test_cellboot_output(bootstrap):
    m = 3 + 8 * 3
    assert bootstrap.theta.shape == (100, m)
    assert bootstrap.confidence_intervals.shape == (m, 2)
    assert np.all(bootstrap.confidence_intervals[:, 0] <= bootstrap.confidence_intervals[:, 1])
    assert bootstrap.coef.shape == (8, 3) and bootstrap.intercept.shape == (3,)


def test_cellboot_linear_combinations():
    X, Y = _data()
    # the first slope of the first response minus that of the second
    combination = np.zeros((1, 3 + 24))
    combination[0, 3], combination[0, 3 + 8] = 1.0, -1.0
    result = cellboot(
        X, Y, n_components=2, ridge=0.1, linear_combinations=combination, n_bootstrap=5
    )
    assert result.theta.shape == (5, 1)
    assert result.confidence_intervals.shape == (1, 2)
