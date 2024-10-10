import numpy as np
from robpy.preprocessing.scaling import RobustScaler
from robpy.univariate.mcd import UnivariateMCD


def test_robust_scaler_can_handle_multiple_dimensions():
    # given
    np.random.seed(2)
    X = np.random.rand(100, 2)
    scaler = RobustScaler()
    # when
    X_scaled = scaler.fit_transform(X)
    X_reversed = scaler.inverse_transform(X_scaled)
    # then
    assert X_scaled.shape == X.shape

    np.testing.assert_array_almost_equal(X, X_reversed)


def test_robust_scaler():
    # given
    np.random.seed(2)
    X = np.random.rand(100, 1)
    estimator = UnivariateMCD()
    scaler = RobustScaler(scale_estimator=estimator)
    # when
    estimator.fit(X.flatten())
    X_scaled = scaler.fit_transform(X)
    # then
    np.testing.assert_array_almost_equal(X_scaled, (X - estimator.location) / estimator.scale)


def test_robust_scaler_only_centering():
    # given
    np.random.seed(2)
    X = np.random.rand(100, 2)
    scaler = RobustScaler(with_scaling=False)
    # when
    X_scaled = scaler.fit_transform(X)
    # then
    assert X_scaled.shape == X.shape
    np.testing.assert_array_equal(scaler.scales_, np.ones(X.shape[1]))


def test_robust_scaler_only_scaling():
    # given
    np.random.seed(2)
    X = np.random.rand(100, 2)
    scaler = RobustScaler(with_centering=False)
    # when
    X_scaled = scaler.fit_transform(X)
    # then
    assert X_scaled.shape == X.shape
    np.testing.assert_array_equal(scaler.locations_, np.zeros(X.shape[1]))
