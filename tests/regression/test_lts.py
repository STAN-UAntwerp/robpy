import pytest

import numpy as np

from sklearn.linear_model import LinearRegression

from robpy.regression.lts import FastLTSRegressor, get_correction_factor


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = (2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)).reshape(-1, 1)
    return X, y


def test_fit_predict(sample_data):
    # given
    X, y = sample_data
    model = FastLTSRegressor(
        alpha=0.5, n_initial_subsets=10, n_initial_c_steps=2, n_best_models=5, tolerance=1e-10
    )
    # when
    model.fit(X, y)
    y_pred = model.predict(X)
    # then
    assert y_pred.shape == y.shape
    assert isinstance(model.model, LinearRegression)


def test_fit_with_initial_weights(sample_data):
    # given
    X, y = sample_data
    initial_weights = np.ones(3)  # 2 features + intercept
    model = FastLTSRegressor(
        alpha=0.5, n_initial_subsets=10, n_initial_c_steps=2, n_best_models=5, tolerance=1e-10
    )
    # when
    model.fit(X, y, initial_weights=initial_weights)
    y_pred = model.predict(X)
    # then
    assert y_pred.shape == y.shape
    assert isinstance(model.model, LinearRegression)


def test_predict_without_fit(sample_data):
    # given
    X, y = sample_data
    model = FastLTSRegressor(
        alpha=0.5, n_initial_subsets=10, n_initial_c_steps=2, n_best_models=5, tolerance=1e-10
    )
    # then
    with pytest.raises(ValueError):
        model.predict(X)


def test_get_loss_value(sample_data):
    # given
    X, y = sample_data
    model = FastLTSRegressor()
    lr_model = LinearRegression().fit(X, y)
    h_subset = np.arange(10)
    # when
    loss_value = model._get_loss_value(X, y, h_subset, lr_model)
    # then
    assert isinstance(loss_value, float)


def test_apply_C_steps_untill_convergence(sample_data):
    # given
    X, y = sample_data
    h = 10
    model = FastLTSRegressor(alpha=h / len(X))
    lr_model = LinearRegression().fit(X, y)
    previous_loss = model._get_loss_value(X, y, np.arange(h), lr_model)
    # when
    (
        current_model,
        current_h_subset,
        current_loss,
        iteration,
    ) = model._apply_C_steps_untill_convergence(lr_model, previous_loss, X, y, h=h, tolerance=1e-10)
    # then
    assert iteration > 0
    assert isinstance(current_model, LinearRegression)
    assert len(current_h_subset) == h
    assert current_loss < previous_loss


def test_get_h_subset(sample_data):
    # given
    X, y = sample_data
    h = 10
    model = FastLTSRegressor()
    lr_model = LinearRegression().fit(X, y)

    # when
    h_subset = model._get_h_subset(lr_model, X, y, h=h)
    # then
    assert len(h_subset) == h


def test_apply_C_step(sample_data):
    # given
    X, y = sample_data
    h = 10
    model = FastLTSRegressor()
    lr_model = LinearRegression().fit(X, y)
    # when
    h_subset, new_lr_model = model._apply_C_step(lr_model, X, y, h=h)
    original_loss = model._get_loss_value(X, y, h_subset, lr_model)
    new_loss = model._get_loss_value(X, y, h_subset, new_lr_model)
    # then
    assert len(h_subset) == h
    assert isinstance(new_lr_model, LinearRegression)
    assert new_loss <= original_loss


@pytest.mark.parametrize(
    "n, p, alpha, expected_output",
    [
        (10, 0, 0.5, 1.21523),
        (100, 1, 0.5, 1.10356),
        (100, 5, 0.9, 1.047415),
        (1000, 10, 0.9, 1.008717),
    ],
)
def test_get_correction_factor(n, p, alpha, expected_output):
    # when
    result = get_correction_factor(p=p, n=n, alpha=alpha)
    # then
    assert result == pytest.approx(expected_output, abs=0.0001)
