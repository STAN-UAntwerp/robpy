import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from dataclasses import dataclass
from statsmodels.stats.stattools import medcouple


@dataclass
class Boxplot:
    """Container for boxplot statistics"""

    median: float
    q1: float
    q3: float
    upper_whisker: float
    lower_whisker: float


def adjusted_boxplot(
    X: np.ndarray | pd.Series | pd.DataFrame,
    plot: bool = True,
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 6),
    **bxp_kwargs,
) -> list[Boxplot]:
    """Calculate and visualize an adjusted boxplot as described in Huber and Vandervieren (2004)

    Args:
        X (np.ndarray or pd.Series or pd.DataFrame): An array of float values
        plot (bool, optional): Whether to plot the boxplot. Defaults to True.
        ax (Axes, optional): The matplotlib axes to plot the boxplot.
          If None, a new figure and axes will be created. Defaults to None.
        figsize (tuple[int, int], optional): Size of the plot. Defaults to (6,6).
        bxp_kwargs (optional): Additional keyword arguments to pass to
          `matplotlib.axes.Axes.bxp`.

    Returns:
        - list[Boxplot]: A list of containers with boxplot statistics for each variable in X

    References:
        - Hubert, M., & Vandervieren, E. (2008). An adjusted boxplot for skewed distributions.
          Computational statistics & data analysis, 52(12), 5186-5201.
    """
    labels = None
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        labels = X.columns if isinstance(X, pd.DataFrame) else [X.name]
        X = X.values
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if labels is None:
        labels = list(range(X.shape[1]))

    medians = np.median(X, axis=0)
    q1s = np.percentile(X, 25, axis=0)
    q3s = np.percentile(X, 75, axis=0)
    mcs = medcouple(X, axis=0)

    lower_whiskers = np.max(
        np.hstack(
            [
                np.min(X, axis=0).reshape(-1, 1),
                (q1s - 1.5 * np.exp(np.where(mcs >= 0, -4, -3) * mcs) * (q3s - q1s)).reshape(-1, 1),
            ]
        ),
        axis=1,
    )
    upper_whiskers = np.min(
        np.hstack(
            [
                np.max(X, axis=0).reshape(-1, 1),
                (q3s + 1.5 * np.exp(np.where(mcs >= 0, 3, 4) * mcs) * (q3s - q1s)).reshape(-1, 1),
            ]
        ),
        axis=1,
    )

    boxplots = [
        Boxplot(m, q1, q3, uw, lw)
        for m, q1, q3, uw, lw in zip(medians, q1s, q3s, upper_whiskers, lower_whiskers)
    ]

    if plot:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        bxp_stats = [
            {
                "med": m,
                "q1": q1,
                "q3": q3,
                "whislo": lw,
                "whishi": uw,
                "fliers": X[(X[:, i] < lw) | (X[:, i] > uw), i],
                "label": l,
            }
            for i, (m, q1, q3, uw, lw, l) in enumerate(
                zip(medians, q1s, q3s, upper_whiskers, lower_whiskers, labels)
            )
        ]
        ax.bxp(bxp_stats, **bxp_kwargs)

    return boxplots
