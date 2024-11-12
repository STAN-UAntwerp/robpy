import pathlib
import matplotlib.pyplot as plt
import numpy as np

from robpy.datasets import load_topgear
from robpy.preprocessing import DataCleaner, RobustPowerTransformer, RobustScaler
from robpy.covariance import FastMCD
from robpy.pca import ROBPCA
from robpy.outliers import DDC
from robpy.covariance.cellmcd import CellMCD
from robpy.regression import MMRegression
from robpy.univariate import adjusted_boxplot

outputfolder = pathlib.Path(__file__).parent / "Output"
print(f"Figures will be stored in {outputfolder}")
outputfolder.mkdir(exist_ok=True)
data = load_topgear(as_frame=True)

cleaner = DataCleaner().fit(data.data)
clean_data = cleaner.transform(data.data)

clean_data = clean_data.drop(columns=["Verdict"])

fig, ax = plt.subplots(1, 1, figsize=(2, 2))
_ = adjusted_boxplot(clean_data["Price"], ax=ax)
fig.show()
plt.savefig(outputfolder / "figure 1a - price boxplot.png")

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
clean_data["Price"].hist(bins=20, ax=ax)
fig.show()
plt.savefig(outputfolder / "figure 1bb - price histogram.png")

price_transformer = RobustPowerTransformer(method="auto").fit(clean_data["Price"])

clean_data["Price_transformed"] = price_transformer.transform(clean_data["Price"])

fig, ax = plt.subplots(1, 1, figsize=(2, 2))
adjusted_boxplot(clean_data["Price_transformed"], ax=ax)
fig.show()
plt.savefig(outputfolder / "figure 2a - price boxplot.png")

fig, ax = plt.subplots(1, 1, figsize=(4, 2))
clean_data["Price_transformed"].hist(bins=20, ax=ax)
fig.show()
plt.savefig(outputfolder / "figure 2b - price histogram.png")


fig, axs = plt.subplots(2, 2, figsize=(15, 8))
for col, ax in zip(["Displacement", "BHP", "Torque", "TopSpeed"], axs.flatten()):
    clean_data[col].hist(ax=ax, bins=20, alpha=0.3)
    transformer = RobustPowerTransformer(method="auto").fit(clean_data[col].dropna())
    clean_data.loc[~np.isnan(clean_data[col]), col] = transformer.transform(
        clean_data[col].dropna()
    )
    ax2 = ax.twiny()
    clean_data[col].hist(ax=ax2, bins=20, label="transformed", color="orange", alpha=0.3)
    ax.grid(False)
    ax2.grid(False)
    ax2.legend(loc="upper right")
    ax.set_title(f"{col}: method = {transformer.method}, lambda = {transformer.lambda_rew:.3f}")
fig.tight_layout()
fig.show()
plt.savefig(outputfolder / "figure 3 - feature transformations.png")

clean_data2 = clean_data.dropna()

mcd = FastMCD().fit(clean_data2.drop(columns=["Price"]))
fig = mcd.distance_distance_plot()
fig.show()
plt.savefig(outputfolder / "figure 4 - mcd distance-distance plot.png")


scaled_data = RobustScaler(with_centering=False).fit_transform(clean_data2.drop(columns=["Price"]))
pca = ROBPCA().fit(scaled_data)

score_distances, orthogonal_distances, score_cutoff, od_cutoff = pca.plot_outlier_map(
    scaled_data, return_distances=True
)
plt.show()
plt.savefig(outputfolder / "figure 5 - pca outlier map.png")

X = clean_data2.drop(columns=["Price", "Price_transformed"])
y = clean_data2["Price_transformed"]

estimator = MMRegression().fit(X, y)
estimator.model.coef_

resid, std_resid, distances, vt, ht = estimator.outlier_map(X, y.to_numpy(), return_data=True)
plt.show()
plt.savefig(outputfolder / "figure 6 - mm regression outlier map.png")


ddc = DDC().fit(clean_data.drop(columns=["Price"]))

row_indices = np.array(
    [11, 41, 55, 73, 81, 94, 99, 135, 150, 164, 176, 198, 209, 215, 234, 241, 277]
)
ax = ddc.cellmap(clean_data.drop(columns=["Price"]), figsize=(8, 10), row_zoom=row_indices)
cars = data.data.apply(lambda row: f"{row['Make']} {row['Model']}", axis=1).tolist()
ax.set_yticklabels([cars[i] for i in row_indices], rotation=0)
plt.show()
plt.savefig(outputfolder / "figure 7 - ddc cellmap zoom.png")


### CELL MCD ###

data = load_topgear(as_frame=True)
car_models = data.data["Make"] + data.data["Model"]

cleaner = DataCleaner().fit(data.data)
clean_data = cleaner.transform(data.data)
clean_data = clean_data.drop(columns=["Verdict"])
for col in ["Displacement", "BHP", "Torque", "TopSpeed"]:
    clean_data[col] = np.log(clean_data[col])
clean_data["Price"] = np.log(clean_data["Price"] / 1000)

car_models.drop(cleaner.dropped_rows["rows_missings"], inplace=True)
car_models = car_models.tolist()
clean_data.head()

cellmcd = CellMCD()
cellmcd.fit(clean_data.values)

variable = 0
variable_name = "Price"
cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="indexplot",
    annotation_quantile=0.9999999,
)
plt.show()
plt.savefig(outputfolder / "figure 8a - cellmcd indexplot.png")


cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="residuals_vs_variable",
    annotation_quantile=0.9999999,
)
plt.show()
plt.savefig(outputfolder / "figure 8b - cellmcd residuals vs variable.png")

cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="residuals_vs_predictions",
    annotation_quantile=0.9999999,
)
plt.show()
plt.savefig(outputfolder / "figure 8c - cellmcd residuals vs predictions.png")

cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="variable_vs_predictions",
    annotation_quantile=0.99999,
)
plt.show()
plt.savefig(outputfolder / "figure 8d - cellmcd variable vs predictions.png")


second_variable = 4
second_variable_name = "Acceleration"
cellmcd.cell_MCD_plot(
    second_variable,
    second_variable_name,
    car_models,
    variable,
    variable_name,
    "bivariate",
    annotation_quantile=0.999999,
)
plt.show()
plt.savefig(outputfolder / "figure 9 - cellmcd bivariate.png")
