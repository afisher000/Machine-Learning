# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 22:49:07 2024

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 10)


# %% Load and split data
df = pd.read_csv('housing.csv')

from sklearn.model_selection import ShuffleSplit
split = ShuffleSplit(n_splits = 1, test_size = .2)
mask_train, mask_test = list(split.split(df))[0]

target_col = 'median_house_value'
X = df.iloc[mask_train].drop(columns = [target_col])
y = df.iloc[mask_train][target_col]
Xtest = df.iloc[mask_test].drop(columns = [target_col])
ytest = df.iloc[mask_test][target_col]

# %% Analyze
X.hist()

from pandas.plotting import scatter_matrix
X['bedrooms'] = X.total_bedrooms / X.households
X['rooms'] = X.total_rooms / X.households
X['family_size'] = X.population / X.households
scatter_matrix(X[['median_income', 'family_size', 'rooms', 'bedrooms']], alpha=.01)



# %%
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cat_features = X.select_dtypes(include = ['object']).columns.tolist()
num_features = X.select_dtypes(exclude = ['object']).columns.tolist()


cat_pipeline = Pipeline([
        
    ])

