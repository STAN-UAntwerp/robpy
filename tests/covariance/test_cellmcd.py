import numpy as np
import pytest
import math
import pandas as pd
from sklearn.exceptions import NotFittedError
from pathlib import Path

from robpy.covariance.cellmcd import CellMCD
from robpy.covariance.initial_ddcw import InitialDDCW


def test_fit_sets_object_attributes():
    # given
    X = np.random.normal(0, 1, [1000, 5])
    # when
    estimator = CellMCD()
    estimator.fit(X)
    eigenvalues = np.linalg.eigvals(estimator.covariance_)
    # then
    assert hasattr(estimator, "covariance_")
    assert hasattr(estimator, "location_")
    assert hasattr(estimator, "predictions")
    assert hasattr(estimator, "residuals")
    assert np.all(eigenvalues > estimator.min_eigenvalue)


def test_large_enough_eigenvalues():
    """Test positive definiteness and large enough eigenvalues."""
    # given
    X = np.random.normal(0, 1, [1000, 5])
    # when
    estimator = CellMCD(min_eigenvalue=1e-5)
    estimator.fit(X)
    eigenvalues = np.linalg.eigvals(estimator.covariance_)
    # then
    assert np.all(eigenvalues > estimator.min_eigenvalue)


@pytest.mark.parametrize(
    "alpha, quantile, max_c_steps, min_eigenvalue",
    [
        (0.75, 0.99, 100, 1e-4),
        (0.8, 0.95, 50, 1e-5),
        (0.75, 0.95, 100, 1e-6),
        (0.85, 0.98, 80, 1e-4),
    ],
)
def test_estimator_can_handle_different_settings(alpha, quantile, max_c_steps, min_eigenvalue):
    # given
    X = np.random.normal(0, 1, [1000, 5])
    # when
    estimator = CellMCD(
        alpha=alpha, quantile=quantile, max_c_steps=max_c_steps, min_eigenvalue=min_eigenvalue
    )
    estimator.fit(X)
    # then
    assert hasattr(estimator, "covariance_")
    assert np.all(np.sum(estimator.W, axis=0) > math.floor(X.shape[0] * alpha))


def test_fit_with_missing_values_and_cellwise_outliers():
    """Exercises the mixed flagged/unflagged prediction branch (numpy >= 2.4 regression)."""
    # given
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(np.zeros(5), 0.5 * np.eye(5) + 0.5, size=200)
    X[rng.random(X.shape) < 0.03] = np.nan
    X[:10, 0] = 10.0
    # when
    estimator = CellMCD().fit(X)
    # then
    assert np.all(estimator.W[:10, 0] == 0)
    assert np.all(estimator.W[np.isnan(X)] == 0)
    assert np.all(np.isfinite(estimator.predictions))
    assert np.all(np.isfinite(estimator.X_imputed))


def _contaminated_data(n, p, seed):
    rng = np.random.default_rng(seed)
    sigma = 0.6 ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    X_clean = rng.multivariate_normal(np.zeros(p), sigma, size=n)
    outliers = rng.random((n, p)) < 0.05
    X = X_clean.copy()
    X[outliers] += rng.choice([-1, 1], outliers.sum()) * 6
    X[rng.random((n, p)) < 0.03] = np.nan
    return X_clean, X, outliers


@pytest.fixture(scope="module")
def fitted_cellmcd():
    _, X_train, _ = _contaminated_data(500, 5, seed=0)
    return CellMCD().fit(X_train), X_train


def test_cellwise_outliers_attribute_matches_W(fitted_cellmcd):
    estimator, X_train = fitted_cellmcd
    assert estimator.cellwise_outliers_.dtype == bool
    np.testing.assert_array_equal(
        estimator.cellwise_outliers_, (estimator.W == 0) & ~np.isnan(X_train)
    )


def test_predict_on_training_data_reproduces_training_flags(fitted_cellmcd):
    estimator, X_train = fitted_cellmcd
    flags = estimator.predict(X_train)
    assert flags.shape == X_train.shape
    assert np.mean(flags == estimator.cellwise_outliers_) > 0.99


def test_predict_flags_outliers_in_new_data(fitted_cellmcd):
    estimator, _ = fitted_cellmcd
    _, X_new, outliers = _contaminated_data(500, 5, seed=1)
    flags = estimator.predict(X_new)
    observed = ~np.isnan(X_new)
    assert flags.dtype == bool
    assert not np.any(flags[~observed])
    assert np.mean(flags[outliers & observed]) > 0.95
    assert np.mean(flags[~outliers & observed]) < 0.03


def test_predict_treats_rows_independently(fitted_cellmcd):
    estimator, _ = fitted_cellmcd
    _, X_new, _ = _contaminated_data(100, 5, seed=2)
    flags = estimator.predict(X_new)
    np.testing.assert_array_equal(estimator.predict(X_new[10:17]), flags[10:17])
    np.testing.assert_array_equal(estimator.predict(X_new[3]), flags[[3]])


def test_impute_replaces_flagged_and_missing_cells_only(fitted_cellmcd):
    estimator, _ = fitted_cellmcd
    X_clean, X_new, outliers = _contaminated_data(500, 5, seed=3)
    flags = estimator.predict(X_new)
    X_imputed = estimator.impute(X_new)
    kept = ~flags & ~np.isnan(X_new)
    assert np.all(np.isfinite(X_imputed))
    np.testing.assert_array_equal(X_imputed[kept], X_new[kept])
    # imputed outliers are much closer to the clean values than the contaminated ones
    imputed = outliers & flags
    error_imputed = np.abs(X_imputed[imputed] - X_clean[imputed]).mean()
    error_contaminated = np.abs(X_new[imputed] - X_clean[imputed]).mean()
    assert error_imputed < 0.25 * error_contaminated


def test_impute_on_training_data_matches_X_imputed_where_flags_agree(fitted_cellmcd):
    estimator, X_train = fitted_cellmcd
    X_imputed = estimator.impute(X_train)
    same_row_flags = np.all(estimator.predict(X_train) == estimator.cellwise_outliers_, axis=1)
    np.testing.assert_allclose(
        X_imputed[same_row_flags], estimator.X_imputed[same_row_flags], rtol=1e-10, atol=1e-10
    )


def test_impute_keeps_dataframe_index_and_columns(fitted_cellmcd):
    estimator, _ = fitted_cellmcd
    _, X_new, _ = _contaminated_data(20, 5, seed=4)
    X_df = pd.DataFrame(X_new, columns=list("abcde"), index=range(100, 120))
    X_imputed = estimator.impute(X_df)
    assert isinstance(X_imputed, pd.DataFrame)
    assert list(X_imputed.columns) == list("abcde")
    assert list(X_imputed.index) == list(range(100, 120))
    np.testing.assert_array_equal(estimator.predict(X_df), estimator.predict(X_new))


def test_predict_and_impute_validate_input(fitted_cellmcd):
    estimator, _ = fitted_cellmcd
    with pytest.raises(NotFittedError):
        CellMCD().predict(np.zeros((3, 5)))
    with pytest.raises(ValueError):
        estimator.predict(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        estimator.impute(np.zeros((3, 6)))


def test_quantile_sets_the_flagging_penalty():
    # given
    _, X, _ = _contaminated_data(500, 5, seed=5)
    # when
    strict = CellMCD(quantile=0.999).fit(X)
    loose = CellMCD(quantile=0.9).fit(X)
    # then: a higher quantile means a higher penalty for flagging, hence fewer flagged cells
    assert np.all(strict._q > loose._q)
    assert strict.cellwise_outliers_.sum() < loose.cellwise_outliers_.sum()


def test_fully_flagged_row_has_marginal_conditional_stds():
    # given
    _, X, _ = _contaminated_data(500, 5, seed=6)
    X[0, :] = 50.0
    # when
    estimator = CellMCD().fit(X)
    # then
    assert np.all(estimator.W[0] == 0)
    np.testing.assert_allclose(estimator.predictions[0], estimator.location_)
    np.testing.assert_allclose(
        estimator.conditional_stds[0], np.sqrt(np.diag(estimator.raw_covariance_))
    )


# ------------------------------------------------------------------ reference results of cellWise

REFERENCE = Path(__file__).parents[1] / "reference"


def _reference(name: str) -> np.ndarray:
    return pd.read_csv(REFERENCE / "expected" / f"{name}.csv").to_numpy()


CELLMCD_CASES = {
    # case: (input, arguments)
    "cellmcd_default": (REFERENCE / "data" / "cellmcd.csv", {}),
    "cellmcd_alpha05": (REFERENCE / "data" / "cellmcd.csv", dict(alpha=0.5, quantile=0.999)),
    "cellmcd_center0": (
        REFERENCE / "expected" / "cellmcd_center0_input.csv",
        dict(fixed_center=True),
    ),
}


@pytest.mark.parametrize("case", CELLMCD_CASES)
def test_cellmcd_matches_cellwise(case):
    path, kwargs = CELLMCD_CASES[case]
    X = pd.read_csv(path).to_numpy()
    estimator = CellMCD(**kwargs).fit(X)
    np.testing.assert_array_equal(estimator.W, _reference(f"{case}_W"))
    assert estimator.n_c_steps_ == _reference(f"{case}_nosteps")[0, 0]
    np.testing.assert_allclose(
        estimator.location_, _reference(f"{case}_location")[:, 0], rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        estimator.covariance_, _reference(f"{case}_covariance"), rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        estimator.raw_covariance_, _reference(f"{case}_raw_covariance"), rtol=1e-7, atol=1e-10
    )
    np.testing.assert_allclose(
        estimator.predictions, _reference(f"{case}_predictions"), rtol=1e-7, atol=1e-9
    )
    # rows in which every cell is flagged: cellWise sets their conditional variances to the
    # diagonal of the inverse covariance matrix, robpy to the marginal variances
    full = np.all(estimator.W == 0, axis=1)
    np.testing.assert_allclose(
        estimator.conditional_stds[~full],
        _reference(f"{case}_conditional_stds")[~full],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        estimator.residuals[~full], _reference(f"{case}_residuals")[~full], rtol=1e-7, atol=1e-8
    )
    np.testing.assert_allclose(
        estimator.conditional_stds[full],
        np.broadcast_to(np.sqrt(np.diag(estimator.raw_covariance_)), (full.sum(), X.shape[1])),
    )


@pytest.mark.parametrize("fixed_center", [False, True])
def test_initial_ddcw_matches_cellwise(fixed_center):
    suffix = "_center0" if fixed_center else ""
    X = pd.read_csv(REFERENCE / "expected" / f"ddcw{suffix}_input.csv").to_numpy()
    ddcw = InitialDDCW(alpha=0.75, fixed_center=fixed_center).fit(X)
    np.testing.assert_allclose(
        ddcw.covariance_, _reference(f"ddcw{suffix}_cov"), rtol=1e-7, atol=1e-10
    )
    if not fixed_center:
        np.testing.assert_allclose(ddcw.location_, _reference("ddcw_center")[:, 0], atol=1e-10)
    else:
        np.testing.assert_array_equal(ddcw.location_, 0.0)


# ------------------------------------------------------------------ per-pattern computations


def test_group_rows_by_pattern():
    from robpy.covariance.utils.cellmcd_utils import group_rows_by_pattern

    rng = np.random.default_rng(0)
    W = (rng.random((200, 11)) > 0.3).astype(float)  # 11 columns: two bytes when packed
    groups = group_rows_by_pattern(W)
    patterns = np.unique(W, axis=0)
    assert len(groups) == len(patterns)
    for (pattern, rows), expected in zip(groups, patterns):
        np.testing.assert_array_equal(pattern, expected.astype(bool))
        np.testing.assert_array_equal(rows, np.flatnonzero(np.all(W == expected, axis=1)))


def _conditional(sigma, mu, x, given, j):
    """Conditional mean and variance of variable j given the variables in `given`."""
    if len(given) == 0:
        return mu[j], sigma[j, j]
    coefficients = np.linalg.solve(sigma[np.ix_(given, given)], sigma[given, j])
    return (
        mu[j] + (x[given] - mu[given]) @ coefficients,
        sigma[j, j] - sigma[j, given] @ coefficients,
    )


def test_deltas_and_predictions_match_direct_computation():
    rng = np.random.default_rng(1)
    n, p = 120, 6
    A = rng.normal(size=(p, p))
    sigma = A @ A.T / p + 0.3 * np.eye(p)
    sigma_inv, mu = np.linalg.inv(sigma), rng.normal(size=p)
    X = rng.normal(size=(n, p)) + mu
    X[rng.random((n, p)) < 0.05] = np.nan
    W = (rng.random((n, p)) > 0.3).astype(float)
    W[np.isnan(X)] = 0
    W[0] = 0  # all cells flagged
    W[1] = 0
    X[1, 2], W[1, 2] = 0.5, 1  # a single kept cell
    for j in range(p):
        delta = CellMCD._calculate_delta(X, W, sigma, sigma_inv, mu, j)
        for i in range(n):
            if np.isnan(X[i, j]):
                assert delta[i] == np.inf
                continue
            given = [k for k in range(p) if k != j and W[i, k] == 1]
            mean, var = _conditional(sigma, mu, X[i], given, j)
            expected = (X[i, j] - mean) ** 2 / var + np.log(var) + np.log(2 * np.pi)
            np.testing.assert_allclose(delta[i], expected, rtol=1e-10)
    X0 = np.where(np.isnan(X), 0.0, X)
    predictions, variances = CellMCD._make_predictions(X0, sigma, sigma_inv, mu, W)
    for i in range(n):
        for j in range(p):
            given = [k for k in range(p) if k != j and W[i, k] == 1]
            mean, var = _conditional(sigma, mu, X0[i], given, j)
            np.testing.assert_allclose(predictions[i, j], mean, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(variances[i, j], var, rtol=1e-10)
