# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:09:17 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from time import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load data
df = pd.read_csv('train.csv')

# Choose features for X and y
features = [
    'LotArea',
    #'LandContour',
    #'Neighborhood',
    'OverallQual',
    'OverallCond',
    'YearBuilt']

target = 'SalePrice'

def print_model_results(model, X, y, model_type):
    ts = time()

    squared_scores = -1*cross_val_score(
        model, X, y, scoring='neg_mean_squared_error', cv=10
        )
    scores = np.sqrt(squared_scores)

    print(model_type)
    print(f'\tError = {scores.mean():.0f} ({scores.std():.0f})')
    print(f'\tTime = {time()-ts:.2f}s')
    return


y = df[target].values
X = df[features].values

pipe = Pipeline([('Scale',StandardScaler())])


# Simple linear regression
linreg = LinearRegression()
print_model_results(linreg, X, y, 'Simple Linear Regression')

# Ridge Regression
ridge = Ridge(alpha=500)
print_model_results(ridge, X, y, 'Ridge Regression')


