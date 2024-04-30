import numpy as np

from typing import Any
from robpy.preprocessing.utils import wrapping_transformation


def test_wrapping_transformation_case1():
    # given
    X = np.array(
        [
            [10, 20],
            [11, 21],
            [9, 19],
            [10, 21],
            [11, 20],
            [10, 19],
            [9.5, 30],
            [10.5, 23],
            [12, 20],
            [4, 19.5],
            [7, 20.5],
        ]
    )
    params: dict[str, Any] = dict(
        b=1.5,
        c=4.0,
        q1=1.540793,
        q2=0.8622731,
        rescale=False,
    )
    expected_result = np.array(
        [
            [10, 20],
            [11, 21],
            [9, 19],
            [10, 21],
            [11, 20],
            [10, 19],
            [9.5, 20],
            [10.5, 21.07459061],
            [11.44589267, 20],
            [10, 19.5],
            [8.92540939, 20.5],
        ]
    )

    # when
    result = wrapping_transformation(X, **params)
    # then
    np.testing.assert_array_almost_equal(result, expected_result)


def test_wrapping_transformation_case2():
    # given
    X = np.array(
        [
            [10, 20],
            [11, 21],
            [9, 19],
            [10, 21],
            [11, 20],
            [10, 19],
            [9.5, 30],
            [10.5, 23],
            [12, 20],
            [4, 19.5],
            [7, 20.5],
        ]
    )
    params: dict[str, Any] = dict(b=5.0, c=50.0, q1=0.8, q2=0.05, rescale=True)
    expected_result = np.array(
        [
            [10.05699239, 19.68220346],
            [10.86011906, 20.60916079],
            [9.25386572, 18.75524614],
            [10.05699239, 20.60916079],
            [10.86011906, 19.68220346],
            [10.05699239, 18.75524614],
            [9.65542905, 20.3970934],
            [10.45855572, 22.46307544],
            [11.66324573, 19.68220346],
            [9.43007612, 19.2187248],
            [7.64761237, 20.14568212],
        ]
    )
    # when
    result = wrapping_transformation(X, **params)
    # then
    np.testing.assert_array_almost_equal(result, expected_result)
