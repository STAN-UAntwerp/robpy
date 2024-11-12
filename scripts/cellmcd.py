#!/usr/bin/env python
# coding: utf-8

# # CellMCD

# In this notebook we analyze the covariance of the TopGear dataset using the cellwise minimum covariance determinant which can handle NAs.

# ## Imports

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from robpy.datasets import load_topgear
from robpy.preprocessing import DataCleaner
from robpy.covariance.cellmcd import CellMCD

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Load and preprocess the data

# To preprocess the data, we run `DataCleaner`, as this should always be done before the cellwise analysis. We additionally remove the variable `Verdict` and take log-transforms of the skewed variables.

# In[2]:


data = load_topgear(as_frame=True)
car_models = data.data['Make'] + data.data['Model']

cleaner = DataCleaner().fit(data.data)
clean_data = cleaner.transform(data.data)
clean_data = clean_data.drop(columns=['Verdict'])
for col in ['Displacement', 'BHP', 'Torque', 'TopSpeed']:
    clean_data[col] = np.log(clean_data[col])
clean_data['Price'] = np.log(clean_data['Price']/1000)

car_models.drop(cleaner.dropped_rows["rows_missings"],inplace=True)
car_models = car_models.tolist()
clean_data.head()


# ## CellMCD

# In[3]:


cellmcd = CellMCD()
cellmcd.fit(clean_data.values)


# We focus on the variable `Price` and make several diagnostic plots.

# In[4]:


variable = 0
variable_name = "Price"
cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="indexplot",
    annotation_quantile=0.9999999
)
plt.show()


# In[5]:


cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="residuals_vs_variable",
    annotation_quantile=0.9999999
)
plt.show()


# In[6]:


cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="residuals_vs_predictions",
    annotation_quantile=0.9999999
)
plt.show()


# In[7]:


cellmcd.cell_MCD_plot(
    variable=variable,
    variable_name=variable_name,
    row_names=car_models,
    plottype="variable_vs_predictions",
    annotation_quantile=0.99999
)
plt.show()


# Next we look at the interaction between the variable `Price` and the variable `Acceleration`.

# In[8]:


second_variable = 4
second_variable_name = "Acceleration"
cellmcd.cell_MCD_plot(
    second_variable,second_variable_name,car_models,variable,variable_name,"bivariate",
    annotation_quantile=0.999999
)
plt.show()

