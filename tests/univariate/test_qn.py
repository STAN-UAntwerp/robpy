import numpy as np
import pytest

from robpy.univariate import Qn


def test_Qn_sets_scale():
    # given
    X = np.random.rand(1000)
    # when
    estimator = Qn().fit(X)
    # then
    assert isinstance(estimator.scale, float)


@pytest.mark.parametrize(
    "X, expected_result",
    [
        (
            np.array([12, 10, 1, -5, -9, 0, 30, 7, 2, 4, 21, 98, 35, 2, 4, 7, 1, -30, -13, -4]),
            13.0638,
        ),
        (
            np.array(
                [
                    -78,
                    28,
                    -31,
                    -97,
                    -65,
                    -36,
                    -30,
                    -61,
                    -21,
                    36,
                    92,
                    54,
                    -54,
                    41,
                    -51,
                    2,
                    -46,
                    -48,
                    -93,
                    69,
                    51,
                    40,
                    64,
                    -76,
                    -98,
                    76,
                    -82,
                    8,
                    58,
                    80,
                ]
            ),
            59.18402,
        ),
    ],
)
def test_Qn_calculates_correctly(X, expected_result):
    # when
    estimate = Qn().fit(X).scale
    # then
    np.testing.assert_almost_equal(estimate, expected_result, decimal=1)
