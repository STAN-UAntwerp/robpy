import numpy as np
import pytest
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


def _skewed_data_with_outliers():
    """Skewed clean column and two contaminated versions (5% of the values replaced).

    The reference values in the tests below are from R cellWise 2.5.7, ``transfo(x, type = "YJ")``
    and ``transfo(x, type = "BC")`` with default arguments, on these exact data.
    """
    rng = np.random.default_rng(14)
    x = rng.lognormal(0, 0.5, 2000)
    bad = rng.random(2000) < 0.05
    # +-8 sd outliers (some negative, so Yeo-Johnson only)
    x_bad = np.where(bad, x.mean() + 8 * x.std() * rng.choice([-1, 1], 2000), x)
    # outliers at +-8 sd on the log scale (positive, for Box-Cox)
    x_bad_pos = np.where(bad, x * np.exp(4 * rng.choice([-1, 1], 2000)), x)
    return x, x_bad, x_bad_pos


def test_yeojohnson_lambda_is_robust_to_outliers():
    # given
    x, x_bad, _ = _skewed_data_with_outliers()
    # when
    clean = RobustPowerTransformer(method="yeojohnson").fit(x)
    contaminated = RobustPowerTransformer(method="yeojohnson").fit(x_bad)
    # then
    # before the fix of the reweighting step: 0.34 and 0.05
    assert abs(contaminated.lambda_rew - clean.lambda_rew) < 0.1
    np.testing.assert_allclose(clean.lambda_rew, 0.3363838662, atol=0.05)
    np.testing.assert_allclose(contaminated.lambda_rew, 0.2808344897, atol=1e-4)
    np.testing.assert_allclose(contaminated.location_post, 0.0336316007, atol=1e-4)
    np.testing.assert_allclose(contaminated.scale_post, 0.9056829240, atol=1e-4)
    np.testing.assert_allclose(contaminated.transform(x_bad).max(), 3.9015491452, atol=1e-4)


def test_boxcox_lambda_is_robust_to_outliers():
    # given
    x, _, x_bad_pos = _skewed_data_with_outliers()
    # when
    clean = RobustPowerTransformer(method="boxcox").fit(x)
    contaminated = RobustPowerTransformer(method="boxcox").fit(x_bad_pos)
    # then
    # before the fix of the reweighting step: 0.012 and 0.037
    assert abs(contaminated.lambda_rew - clean.lambda_rew) < 0.1
    np.testing.assert_allclose(clean.lambda_rew, 0.0117986317, atol=1e-4)
    np.testing.assert_allclose(contaminated.lambda_rew, -0.0020372342, atol=1e-4)
    np.testing.assert_allclose(contaminated.location_post, 0.0247455303, atol=1e-4)
    np.testing.assert_allclose(contaminated.scale_post, 0.4823700269, atol=1e-4)


def test_lambda_range_is_widened_when_the_estimate_reaches_a_bound():
    # given: data with lambda about 0.34 and an initial range that excludes it
    x, _, _ = _skewed_data_with_outliers()
    # when
    estimator = RobustPowerTransformer(lambda_range=(0.5, 1.5)).fit(x)
    # then
    expected = RobustPowerTransformer().fit(x).lambda_rew
    np.testing.assert_allclose(estimator.lambda_rew, expected, atol=1e-3)
    np.testing.assert_allclose(estimator.inverse_transform(estimator.transform(x)), x)


def test_lambda_range_must_contain_one():
    with pytest.raises(ValueError, match="lambda_range"):
        RobustPowerTransformer(lambda_range=(-0.5, 0.5)).fit(np.arange(1.0, 50.0))


@pytest.mark.parametrize(
    "method, x",
    [
        ("yeojohnson", _skewed_data_with_outliers()[1]),  # lambda < 1
        ("yeojohnson", -_skewed_data_with_outliers()[1]),  # lambda > 1
        ("boxcox", _skewed_data_with_outliers()[2]),  # lambda close to 0
        ("boxcox", np.random.default_rng(4).chisquare(3, 500) ** 2),  # 0 < lambda < 1
        ("boxcox", 1 / np.random.default_rng(5).lognormal(0, 0.5, 500) ** 0.3 + 5),  # lambda > 1
    ],
)
def test_rectified_transform(method, x):
    # given
    classical = RobustPowerTransformer(method=method).fit(x)
    # when
    rectified = RobustPowerTransformer(method=method, rectified=True).fit(x)
    y = rectified.transform(x)
    # then
    assert rectified.lambda_rew == classical.lambda_rew
    assert rectified.changepoint_ is not None
    assert not hasattr(classical, "changepoint_")
    order = np.argsort(x)
    assert np.all(np.diff(y[order]) >= 0)  # monotone
    np.testing.assert_allclose(rectified.inverse_transform(y), x)
    # equal to the classical transformation up to the change point (up to standardization)
    y_classical = classical.transform(x)
    y_std = (y * rectified.scale_post + rectified.location_post - classical.location_post) / (
        classical.scale_post
    )
    x_std = rectified._standardize_pre(x)
    inside = (
        x_std <= rectified.changepoint_
        if rectified.lambda_rew < 1
        else (x_std >= rectified.changepoint_)
    )
    np.testing.assert_allclose(y_std[inside], y_classical[inside])
    # the tail beyond the change point is less compressed
    if rectified.lambda_rew < 1:
        assert y_std.max() > y_classical.max()
    else:
        assert y_std.min() < y_classical.min()


def test_rectified_transform_keeps_outliers_outlying():
    # given
    _, x_bad, _ = _skewed_data_with_outliers()
    # when
    classical = RobustPowerTransformer().fit(x_bad).transform(x_bad)
    rectified = RobustPowerTransformer(rectified=True).fit(x_bad).transform(x_bad)
    # then: the largest (+8 sd) outlier ends up further out
    assert classical.max() < 4
    assert rectified.max() > 6
