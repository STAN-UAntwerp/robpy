import pytest
import numpy as np

from robpy.utils.outlyingness import stahel_donoho


@pytest.mark.parametrize("n_points", [2, 3, 5])
def test_stahel_donoho(n_points):
    # given
    n = 100
    X = np.random.randn(n, 10)
    # when
    outlyingness = stahel_donoho(X, n_points=n_points)
    # then
    assert outlyingness.shape == (n,)
