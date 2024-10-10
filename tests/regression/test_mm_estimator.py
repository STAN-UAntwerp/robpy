import numpy as np
from sklearn.datasets import make_regression

from robpy.regression.mm import MMRegression


def test_s_estimator_fit_predict():
    # given
    n_features = 5
    X, y = make_regression(n_samples=100, n_features=n_features, noise=0.1, random_state=42)
    model = MMRegression().fit(X, y)

    # when
    y_pred = model.predict(X)

    # then
    assert isinstance(y_pred, np.ndarray)
    assert model.n_iter >= 1
    assert model.scale > 0
