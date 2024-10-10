import numpy as np
import pytest
from robpy.covariance import RobustCovariance


def test_no_calculation_method_defined():
    # given
    X = np.random.rand(100, 2)
    # then
    with pytest.raises(NotImplementedError):
        RobustCovariance().fit(X)
