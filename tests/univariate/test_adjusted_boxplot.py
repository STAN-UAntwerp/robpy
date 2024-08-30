import pytest
import numpy as np
import matplotlib.pyplot as plt
from robpy.univariate import adjusted_boxplot


def test_adjusted_boxplot_single_column():
    # given
    np.random.seed(100)
    X = np.random.lognormal(size=(100))

    # when
    boxplots = adjusted_boxplot(X, plot=False)

    # then
    assert len(boxplots) == 1
    assert boxplots[0].lower_whisker == pytest.approx(0.083146469)
    assert boxplots[0].upper_whisker == pytest.approx(7.649249676)


def test_adjusted_boxplot_multiple_columns():
    # given
    np.random.seed(100)
    ncols = 3
    X = np.random.lognormal(size=(100, ncols))

    # when
    boxplots = adjusted_boxplot(X, plot=False)

    # then
    assert len(boxplots) == ncols
    assert boxplots[0].lower_whisker == pytest.approx(0.051133497)
    assert boxplots[0].upper_whisker == pytest.approx(7.726035291)
    assert boxplots[1].lower_whisker == pytest.approx(0.21408484)
    assert boxplots[1].upper_whisker == pytest.approx(12.6724713)


def test_adjusted_boxplot_plots():
    # given
    np.random.seed(100)
    ncols = 3
    X = np.random.lognormal(size=(100, ncols))
    _, ax = plt.subplots(1, 1)

    # when
    _ = adjusted_boxplot(X, ax=ax)

    # then
    assert len(list(ax.get_lines())) == ncols * 7  # 7 lines per boxplot
    assert ax.get_lines()[3].get_data()[1][0] == pytest.approx(0.051133497)
    assert ax.get_lines()[4].get_data()[1][0] == pytest.approx(7.726035291)
