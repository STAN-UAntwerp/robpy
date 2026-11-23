"""The MCD with robustbase's corrections, and robustbase's Qn and scaleTau2, against R.

Reference values: rrcov::CovMcd (robustbase 0.99.6, rrcov 1.7.7) on the data below."""

import numpy as np
import pytest

from robpy.covariance.utils.robustbase_mcd import _qn, _scale_tau2, robustbase_mcd


def _data():
    rng = np.random.default_rng(3)
    X = np.sqrt(0.5) * rng.normal(size=(59, 1)) + np.sqrt(0.5) * rng.normal(size=(59, 3))
    X[:5] += 4
    x = np.concatenate([rng.normal(size=50), [8.0, 9.0, 10.0]])
    return X, x


@pytest.mark.parametrize(
    "alpha, deterministic, center, cov_diagonal, crit",
    [
        (
            0.5,
            False,
            [0.06017321202, -0.04293650087, -0.01820138089],
            [0.7589408268, 0.8679697058, 1.1415775],
            -3.527595701,
        ),
        (
            0.5,
            True,
            [0.08102619278, -0.0117750912, 0.03198167605],
            [0.7678922095, 0.9044941862, 1.258491777],
            -3.510552883,
        ),
        (
            0.75,
            False,
            [0.1050842466, 0.08520151132, 0.1169862326],
            [0.9112887944, 1.093630844, 1.445526486],
            -1.908777508,
        ),
        (
            0.75,
            True,
            [0.1050842466, 0.08520151132, 0.1169862326],
            [0.9112887944, 1.093630844, 1.445526486],
            -1.904172535,
        ),
    ],
)
def test_mcd_matches_rrcov(alpha, deterministic, center, cov_diagonal, crit):
    X, _ = _data()
    result = robustbase_mcd(X, alpha, deterministic=deterministic)
    np.testing.assert_allclose(result.center, center, atol=1e-9)
    np.testing.assert_allclose(np.diag(result.cov), cov_diagonal, rtol=1e-8)
    np.testing.assert_allclose(result.crit, crit, atol=1e-8)


def test_univariate_mcd_matches_rrcov():
    _, x = _data()
    result = robustbase_mcd(x, 0.5)
    np.testing.assert_allclose(result.center, [0.2723158045], atol=1e-9)
    np.testing.assert_allclose(result.cov, [[0.9788230168]], rtol=1e-8)
    np.testing.assert_allclose(result.raw_center, [0.2756559909], atol=1e-9)
    np.testing.assert_allclose(result.raw_cov, [[0.926605445]], rtol=1e-8)
    np.testing.assert_allclose(result.crit, -2.070134824, atol=1e-8)


def test_qn_and_scale_tau2_match_robustbase():
    _, x = _data()
    np.testing.assert_allclose(_qn(x), 1.09607366613916, rtol=1e-12)
    np.testing.assert_allclose(_qn(x[:12]), 1.01083441625026, rtol=1e-12)
    np.testing.assert_allclose(_scale_tau2(x), 0.9257073907174, rtol=1e-12)


def test_mcd_finds_the_optimum_regardless_of_the_seed():
    X, _ = _data()
    criteria = {robustbase_mcd(X, 0.5, random_seed=seed).crit for seed in range(3)}
    assert max(criteria) - min(criteria) < 1e-10
