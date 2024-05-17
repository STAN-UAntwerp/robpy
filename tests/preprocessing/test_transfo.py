import numpy as np
from robpy.preprocessing.transfo import RobustPowerTransformer


def test_fit_sets_object_attributes():
    # given
    X = np.abs(np.random.rand(100))
    # when "boxcox"
    estimator = RobustPowerTransformer(method="boxcox", standardize=True).fit(X)
    # then
    assert hasattr(estimator, "lambda_rew")
    assert hasattr(estimator, "scale_post")
    assert hasattr(estimator, "location_post")
    assert hasattr(estimator, "scale_pre")
    # when "yeojohnson"
    estimator = RobustPowerTransformer(method="yeojohnson", standardize=True).fit(X)
    # then
    assert hasattr(estimator, "lambda_rew")
    assert hasattr(estimator, "scale_post")
    assert hasattr(estimator, "location_post")
    assert hasattr(estimator, "scale_pre")
    assert hasattr(estimator, "location_pre")


def test_fit_transform():
    # given
    X = np.random.lognormal(size=1000)
    robust_transformation = RobustPowerTransformer().fit(X)
    # when
    Y = robust_transformation.transform(X)
    Z = robust_transformation.inverse_transform(Y)
    # then
    assert Y.shape == X.shape
    np.testing.assert_array_almost_equal(X, Z)


def test_reweighted_lambda():
    X = np.array(
        [
            2.08,
            0.96,
            3.33,
            4.35,
            1.14,
            1.68,
            0.58,
            0.31,
            0.32,
            3.19,
            0.99,
            1.73,
            0.28,
            0.9,
            0.77,
            1.87,
            0.85,
            1.96,
            4.65,
            0.79,
            1.15,
            1.43,
            0.68,
            0.66,
            3.51,
            0.23,
            0.9,
            1.02,
            0.87,
            0.6,
            0.28,
            4.26,
            0.17,
            5.5,
            1.05,
            5.35,
            0.75,
            0.28,
            11.45,
            0.69,
            0.73,
            2.05,
            1.3,
            1.08,
            2.4,
            0.6,
            1.14,
            1.17,
            0.16,
            0.55,
            5.85,
            0.82,
            2.09,
            2.4,
            0.78,
            1.01,
            0.39,
            4.1,
            1.62,
            2.82,
            2.19,
            2.11,
            1.94,
            3.36,
            1.79,
            0.92,
            0.77,
            0.44,
            1.18,
            1.91,
            0.36,
            3.78,
            0.14,
            2.82,
            0.95,
            0.97,
            0.82,
            0.46,
            0.2,
            1.94,
            0.42,
            1.88,
            4.57,
            0.45,
            0.72,
            6.42,
            1.29,
            3.52,
            8.91,
            1.13,
            0.26,
            1.22,
            1.17,
            1.39,
            0.1,
            0.76,
            0.5,
            0.85,
            0.58,
            0.66,
        ]
    )
    expected_results = np.array(
        [
            -0.7974395,
            -0.139671,
            0.04729415,
            0.04729415,
        ]
    )
    results = np.array(
        [
            RobustPowerTransformer("yeojohnson", False).fit(X).lambda_rew,
            RobustPowerTransformer("yeojohnson", True).fit(X).lambda_rew,
            RobustPowerTransformer("boxcox", False).fit(X).lambda_rew,
            RobustPowerTransformer("boxcox", True).fit(X).lambda_rew,
        ]
    )

    np.testing.assert_array_almost_equal(results[0], expected_results[0], decimal=2)
    np.testing.assert_array_almost_equal(results[1], expected_results[1], decimal=2)
    np.testing.assert_array_almost_equal(results[2], expected_results[2], decimal=2)
    np.testing.assert_array_almost_equal(results[3], expected_results[3], decimal=2)
