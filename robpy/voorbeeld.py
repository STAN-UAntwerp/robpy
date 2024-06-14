import numpy as np
import matplotlib.pyplot as plt
from robpy.covariance.cellmcd import CellMCDEstimator

car_models = []
with open("/home/sarahleyder/robpy/robpy/topgear_cellMCD_rownames.txt", "r") as file:
    for line in file:
        car_model = line.strip().strip('"')  # Remove leading/trailing whitespace and newline
        car_models.append(car_model)


def na_handler(x):
    return np.nan if x == b"NA" else float(x)


X = np.genfromtxt(
    "/home/sarahleyder/robpy/robpy/topgear_cellMCD.txt", dtype=float, converters={1: na_handler}
)
cellmcd = CellMCDEstimator()
cellmcd.calculate_covariance(X)
variable = 0
variable_name = "price"
cellmcd.cell_MCD_plot(
    variable=variable, variable_name=variable_name, row_names=car_models, plottype="indexplot"
)
cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="residuals_vs_variable",
)
cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="residuals_vs_predictions",
)
cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="variable_vs_predictions",
)
cellmcd.cell_MCD_plot(
    variable=4,
    variable_name="acceleration",
    second_variable=0,
    second_variable_name="price",
    row_names=car_models,
    plottype="bivariate",
)
plt.show()
