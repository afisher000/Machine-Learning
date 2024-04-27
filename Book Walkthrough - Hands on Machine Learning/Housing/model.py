# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:52:17 2024

@author: afisher
"""

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# %% Reading and displaying
pd.set_option('display.max_columns', 10)
df = pd.read_csv('housing.csv')

df.info() #What are the columns/datatypes
df.head(n=5)
df.tail(n=5) 
df.describe() #Numerical, Look for skewed distributions, cutoff values, outliers, unit meaning

# %% Split into test/train data
# Avoid "snooping bias"
# Can stratify on target variable (group on feature)
# Understand these functions and how to use (returns generator)
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
# split = ShuffleSplit(n_splits = 1, test_size = .2, train_size = .8)
# split = StratifiedShuffleSplit(n_splits = 1, test_size = .2, train_size = .8)
# split = KFold(n_splits = 5, shuffle=True)


# %% Understanding relationships
# df.hist(bins = 50, figsize=(20,15))

# from pandas.plotting import scatter_matrix 
# features = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
# scatter_matrix(df[features], alpha=.01)

# df.ocean_proximity.value_counts() #Count for categorical

# corr_matrix = df.corr() #does not include nonlinear relationships
# corr_matrix.population.sort_values(ascending = False)


# %% Feature engineering (adding features, imputation, scaling)
# df.isnull().sum() and df.notnull().sum()
# df.dropna(axis=0, thresh=2) or df.dropna(axis=0, how='any'/'all', subset = [''])
# fit and transform methods.

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
# Custom transformers can be useful for comparing models with/without additional features?


# %% Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Create Pipelines for subsets of features, then combine with column transformer. 
# num_pipeline = Pipeline([
#         ('imputer', SimpleImputer(strategy='median'),
#         ('add_features', AddFeatures()),
#         ('std_scaler', StandardScaler()),
#     ])

# cat_pipeline = Pipeline([
#         ('encoder', OneHotEncoder()),
#     ])

# num_features = df.select_dtypes(exclude=['object']).columns.tolist()
# cat_features = df.select_dtypes(include=['object']).columns.tolist()
# drop_features= "features I want dropped"
# full_pipeline = ColumnTransformer([
#         ('num', num_pipeline, num_features),
#         ('cat', cat_pipeline, cat_features),
#         ('drop', 'drop', drop_features),
#     ], remainer='drop'/'passthrough')

# X = full_pipeline.fit_transform(df_train)
# y = df_train['median_house_value'].copy()
# Xtest = full_pineline.fit_transform(df_test)
# ytest = df_test['median_house_value'].copy()




# %% Models
# Models have fit method and can be made as last step in pipeline
from sklearn.linear_model import LinearRegression
model = LinearRegression()

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# model.fit(X, y)
# ypred = model.predict(Xtest)


from sklearn.metrics import mean_squared_error
# error_rms = mean_squared_error(ytest, y_pred)

from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10)

# What are common classification and regression metrics?


# %% Fine tuning models
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# These must be parameter of base estimator (all lists)
param_grid = [
        {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
        {'bootstrap':[False], 'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
    ]
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X, y)







