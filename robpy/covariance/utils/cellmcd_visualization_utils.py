import numpy as np

from matplotlib.axes import Axes
from scipy.stats import chi2
from robpy.utils.distance import mahalanobis_distance
from robpy.univariate.onestep_m import OneStepWrappingEstimator


def annote_outliers(
    ax: Axes,
    row_names,
    x: np.array,
    y: np.array,
    h_thresholds: tuple[float, float],
    v_thresholds: tuple[float, float] = None,
):
    """Label outlying points (x,y) in a plot with their case name if they exceed a vertical
    or hoizontal threshold.

    Arguments:
    - ax (Axes): the plot
    - row_names (list of strings): list containing the names of the cases/rows
    - x (np.array): x-coordinates of the points
    - y (np.array): y-coordinates of the points
    - h_thresholds (list[float, float]): horizontal thresholds (lower and upper)
    - v_thresholds (list[float, float], optional): vertical thresholds (left and right)"""

    for i, (xi, yi) in enumerate(zip(x, y)):
        h_outlier = (yi < h_thresholds[0]) or (yi > h_thresholds[1])
        v_outlier = (v_thresholds is not None) and (
            (xi < v_thresholds[0]) or (xi > v_thresholds[1])
        )
        if h_outlier or v_outlier:
            ax.text(xi, yi, row_names[i], fontsize=9, ha="center", va="bottom")


def annote_outliers_ellipse(
    ax: Axes,
    row_names,
    location: np.array,
    covariance: np.ndarray,
    variable: int,
    second_variable: int,
    x: np.array,
    y: np.array,
    quantile: float = 0.99,
):
    """Label outlying points (x,y) with their case name if they are outside the tolerance ellipse

    - ax (Axes): the plot
    - row_names (list of strings): list containing the names of the cases/rows
    - location (np.array): location estimate of the data
    - covariance (np.ndarray): covariance estimate of the data
    - variable (integer): index of the first variable
    - second variable (integer): index of the second variable
    - x (np.array): x-coordinates of the points
    - y (np.array): y-coordinates of the points
    - quantile (float, optional): Cutoff value to flag cells.
    """
    mask = mahalanobis_distance(
        np.column_stack((x, y)),
        location[[variable, second_variable]],
        covariance=covariance[np.ix_([variable, second_variable], [variable, second_variable])],
    ) > np.sqrt(chi2.ppf(quantile, 2))
    for xi, yi, name in zip(x[mask], y[mask], [row_names[i] for i in np.where(mask)[0]]):
        ax.text(xi, yi, name, fontsize=9, ha="center", va="bottom")


def draw_ellipse(cov: np.ndarray, center: np.array, ax: Axes, quantile: float):
    """Get the ellipse for bivariate data given the covariance matrix (for the shape) and the
    location (for the center)."""

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    shape = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T  # orthogonalize
    angles = np.linspace(0, 2 * np.pi, 200 + 1)
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    radius = np.sqrt(chi2.ppf(quantile, 2))
    ellipse = radius * xy @ shape + center
    ax.plot(ellipse[:, 0], ellipse[:, 1], linewidth=3, color="darkgray")


def get_thresholds(cutoff: float, x: np.array) -> tuple[float, float]:
    scaler = OneStepWrappingEstimator().fit(x, ignore_nan=True)
    return (scaler.location - cutoff * scaler.scale, scaler.location + cutoff * scaler.scale)


def draw_threshold_lines(
    ax: Axes,
    h_thresholds: list[float, float],
    v_thresholds: list[float, float] | None,
):
    for h in h_thresholds:
        ax.axhline(h, color="grey", linestyle="--")
    if v_thresholds is not None:
        for v in v_thresholds:
            ax.axvline(v, color="grey", linestyle="--")
