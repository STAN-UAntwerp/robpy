#!/usr/bin/env python
# coding: utf-8

# # TopGear Data

# This notebook demonstrates most of the functionality offered by the robpy package through an application to the TopGear dataset. Utility functions and univariate estimators are not demonstrated separately, as they are mostly implemented as helpers to the multivariate estimators.

# ## Imports

# In[1]:


import json
import matplotlib.pyplot as plt
import numpy as np

from robpy.datasets import load_topgear
from robpy.preprocessing import DataCleaner, RobustPowerTransformer, RobustScaler
from robpy.covariance import FastMCD, OGK
from robpy.pca import ROBPCA
from robpy.outliers import DDC
from robpy.regression import MMRegression
from robpy.univariate import adjusted_boxplot

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load data

# Robpy has a dataset module which allows the user quick access to a few common datasets used in robust statistics literature. In this demo, we will work with the TopGear dataset.

# In[2]:


data = load_topgear(as_frame=True)

print(data.DESCR)


# The `DESCR` attribute of the data object contains all metadata and a description of the dataset.

# In[3]:


data.data.head()


# ## Preprocess data

# ### Cleaning

# We use `DataCleaner` to remove non-numeric columns, as well as columns and rows with too many missing values, discrete columns, columns with a scale of zero and columns corresponding to the case numbers.

# In[4]:


cleaner = DataCleaner().fit(data.data)
clean_data = cleaner.transform(data.data)


# We can inspect the dropped columns by type:

# In[5]:


print(json.dumps(cleaner.dropped_columns, indent=4))


# As well as the dropped rows:

# In[6]:


cleaner.dropped_rows


# Let's also exclude the variable `Verdict` from the features for further analysis.

# In[7]:


clean_data = clean_data.drop(columns=['Verdict'])


# ### Transforming

# The `Price` variable has a long tail as can be seen from its adjusted boxplot and its histogram. It's a good idea to apply a power transformation to it to make it more symmetric.

# In[8]:


adjusted_boxplot(clean_data['Price'],figsize=(2,2))


# In[9]:


clean_data['Price'].hist(bins=20, figsize=(4,2))


# In[10]:


price_transformer = RobustPowerTransformer(method='auto').fit(clean_data['Price'])

clean_data['Price_transformed'] = price_transformer.transform(clean_data['Price'])


# We can inspect which method was selected as well as which lambda value was applied for the transformation:

# In[11]:


price_transformer.method, price_transformer.lambda_rew


# In[12]:


adjusted_boxplot(clean_data['Price_transformed'],figsize=(2,2))


# In[13]:


clean_data['Price_transformed'].hist(bins=20, figsize=(4,2))


# It's best to also transform `displacement`, `HP`, `Torque` and `Topspeed` as these variables are also skewed.

# In[14]:


fig, axs = plt.subplots(2, 2, figsize=(15, 8))
for col, ax in zip(['Displacement', 'BHP', 'Torque', 'TopSpeed'], axs.flatten()):
    clean_data[col].hist(ax=ax, bins=20, alpha=0.3)
    transformer = RobustPowerTransformer(method='auto').fit(clean_data[col].dropna())
    clean_data.loc[~np.isnan(clean_data[col]), col] = transformer.transform(clean_data[col].dropna())
    ax2=ax.twiny()
    clean_data[col].hist(ax=ax2, bins=20, label='transformed', color='orange', alpha=0.3)
    ax.grid(False)
    ax2.grid(False)
    ax2.legend(loc='upper right')
    ax.set_title(f'{col}: method = {transformer.method}, lambda = {transformer.lambda_rew:.3f}')
fig.tight_layout()


# ### Dropping NA

# Some methods require the data to be entirely NA free.

# In[15]:


clean_data2 = clean_data.dropna()


# ## Covariance

# First we study the covariance of the dataset by calculating a robust covariance matrix on the numeric features.

# ### MCD: Minimum Covariance Determinant

# In[16]:


mcd = FastMCD().fit(clean_data2.drop(columns=['Price']))


# In[17]:


fig = mcd.distance_distance_plot()


# ### OGK: Orthogonalized Gnanadesikan-Kettenring covariance

# We can do the same analysis with other robust covariance estimators, e.g. the OGK covariance:

# In[18]:


ogk = OGK().fit(clean_data2.drop(columns=['Price']))


# In[19]:


fig = ogk.distance_distance_plot()


# Let's have a look at the point that seems to lie very far from the majority of the data:

# In[20]:


data.data.loc[
    clean_data2.index[(ogk._robust_distances > 80) & (ogk._mahalanobis_distances > 12)], 
    ['Make', 'Model'] + list(set(clean_data2.columns).intersection(set(data.data.columns)))
]


# ### CellMCD

# Another robust covariance estimator that can handle missing values is the CellMCD. For more details, we refer to the separate notebook on this topic.

# ## PCA

# Next, we apply a well-known robust PCA method to the data, specifically ROBPCA. As the variables of interest have different measurement units and different scales, we first scale the data.

# In[21]:


scaled_data = RobustScaler(with_centering=False).fit_transform(clean_data2.drop(columns=['Price']))
pca = ROBPCA().fit(scaled_data)


# In[22]:


score_distances, orthogonal_distances, score_cutoff, od_cutoff = pca.plot_outlier_map(scaled_data, return_distances=True)


# We can inspect the bad leverage points:

# In[23]:


data.data.loc[
    clean_data2.loc[(score_distances > score_cutoff) & (orthogonal_distances > od_cutoff)].index, 
    ['Make', 'Model'] + list(set(clean_data2.columns).intersection(set(data.data.columns)))
]


# ## Outlier detection

# We can also detect cellwise outliers with the DDC estimator. Here we can use the data with NAs, as DDC can handle missing values.

# In[24]:


ddc = DDC().fit(clean_data.drop(columns=['Price']))


# In[25]:


ddc.cellmap(clean_data.drop(columns=['Price']), figsize=(15,30))


# In[26]:


row_indices = np.array([ 11,  41,  55,  73,  81,  94,  99, 135, 150, 164, 176, 198, 209,
       215, 234, 241, 277])
ax = ddc.cellmap(clean_data.drop(columns=['Price']), figsize=(8,10), row_zoom=row_indices)
cars = data.data.apply(lambda row: f"{row['Make']} {row['Model']}", axis=1).tolist()
ax.set_yticklabels([cars[i] for i in row_indices], rotation=0);


# We can additionally zoom in on 2 BMWs:

# In[27]:


ax = ddc.cellmap(clean_data.drop(columns=['Price']),row_zoom=[31,41], figsize=(7,1))
ax.set_yticklabels([cars[i] for i in [31,41]], rotation=0);


# total flagged cells:

# In[28]:


ddc.cellwise_outliers_.sum()


# DDC also predicts rowwise outliers:

# In[29]:


clean_data.loc[ddc.predict(clean_data.drop(columns=['Price']), rowwise=True)]


# Finally, DDC can impute missing data:

# In[30]:


ddc.impute(clean_data.drop(columns=['Price'])).loc[ddc.predict(clean_data.drop(columns=['Price']), rowwise=True), :].style.format('{:.2f}')


# ## Regression

# We can use robust regression to predict the (transformed) price using the remaining variables. For regression, we again have to drop all missings.

# In[31]:


X = clean_data2.drop(columns=['Price', 'Price_transformed'])
y = clean_data2['Price_transformed']


# As an example, we use the MM-estimator of regression:

# In[32]:


estimator = MMRegression().fit(X, y)
estimator.model.coef_


# We can now get a diagnostic plot and ask for the underlying data

# In[33]:


resid, std_resid, distances, vt, ht = estimator.outlier_map(X, y.to_numpy(), return_data=True)


# Now we can get an overview of the bad leverage points:

# In[34]:


bad_leverage_idx = (np.abs(std_resid) > vt) & (distances > ht)
data.data.loc[
    clean_data2[bad_leverage_idx].index, ['Make', 'Model', 'Price']
].assign(predicted_price=price_transformer.inverse_transform(estimator.predict(X.loc[bad_leverage_idx])).round())

