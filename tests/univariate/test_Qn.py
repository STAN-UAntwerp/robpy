import numpy as np
import pytest
from robpy.univariate.Qn import QnEstimator


def test_calculate_Qn():
    # given
    X = np.random.rand(1000)
    # when
    estimator = QnEstimator(X, estimator="Qn")
    estimator.calculate_Qn()
    # then
    assert hasattr(estimator, "Qn")
    assert isinstance(estimator.Qn, float)


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
def test_eval(X, expected_result):
    np.testing.assert_almost_equal(
        QnEstimator(X, estimator="Qn").calculate_Qn(), expected_result, decimal=1
    )
