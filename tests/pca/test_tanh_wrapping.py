import numpy as np

from robpy.utils.rho import TanhWrapping, calculate_q1q2, scale_tanh, get_c_scale_tanh

# Reference values produced with the R cellPCA code (rhoTanh / weightTanh / scaleTanh).
X = np.array([-6, -4, -3.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3.5, 4, 6.0])


def test_tanh_rho_default_matches_r():
    expected = np.array(
        [
            3.762213,
            3.762213,
            3.601043,
            1.863315,
            1.125,
            0.5,
            0.125,
            0.0,
            0.125,
            0.5,
            1.125,
            1.863315,
            3.601043,
            3.762213,
            3.762213,
        ]
    )
    np.testing.assert_allclose(TanhWrapping(b=1.5).rho(X), expected, atol=1e-6)


def test_tanh_weight_default_matches_r():
    expected = np.array(
        [0, 0, 0.1788512, 0.7229463, 1, 1, 1, 1, 1, 1, 1, 0.7229463, 0.1788512, 0, 0.0]
    )
    np.testing.assert_allclose(TanhWrapping(b=1.5).weight(X), expected, atol=1e-6)


def test_tanh_custom_parametrization_matches_r():
    rho2 = TanhWrapping(bb=0.7722239, cc=1.02778, q1=2.350072, q2=1.335335)
    rho_expected = np.array(
        [
            0.3987097,
            0.3987097,
            0.3987097,
            0.3987097,
            0.3987097,
            0.3974991,
            0.125,
            0.0,
            0.125,
            0.3974991,
            0.3987097,
            0.3987097,
            0.3987097,
            0.3987097,
            0.3987097,
        ]
    )
    weight_expected = np.array([0, 0, 0, 0, 0, 0.08713738, 1, 1, 1, 0.08713738, 0, 0, 0, 0, 0.0])
    np.testing.assert_allclose(rho2.rho(X), rho_expected, atol=1e-6)
    np.testing.assert_allclose(rho2.weight(X), weight_expected, atol=1e-6)


def test_calculate_q1q2_matches_r():
    q1, q2 = calculate_q1q2(0.7722239, 1.02778)
    np.testing.assert_allclose([q1, q2], [2.350074, 1.335335], atol=1e-4)


def test_scale_tanh_matches_r():
    assert abs(scale_tanh(np.array([1.0, 2, 3, 4, 5, 100])) - 5.106698) < 1e-4


def test_get_c_scale_tanh_default():
    assert get_c_scale_tanh() == 0.34308


def test_tanh_rho_preserves_nan():
    out = TanhWrapping(b=1.5).rho(np.array([1.0, np.nan, -2.0]))
    assert np.isnan(out[1]) and not np.isnan(out[0])
