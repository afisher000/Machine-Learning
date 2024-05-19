# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:48:33 2024

@author: afisher
"""
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


# %% Read all datasets and concatenate
sim_data_folder = '../Particle Simulations/Simulation Data'

dfs = [pd.read_csv( os.path.join(sim_data_folder, file)) 
       for file in os.listdir(sim_data_folder)]

df = pd.concat(dfs)

# %% Split data/labels into training and test
targets = ['time_std', 'G_mean', 'G_std']
X = df.drop(columns = targets)
y = df[targets]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# %% Scale inputs
pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(2))
    ])

X_train_scaled = pipeline.fit_transform(X_train)


# %% Fit G_mean
target = 'G_mean'

model = MLPRegressor(random_state=1, max_iter=500)
# model = Ridge(alpha=1)

model.fit(X_train_scaled, y_train[target])
y_pred = model.predict(X_train_scaled)

plt.scatter(y_pred, y_train[target])

# %% Cross validation
scores = cross_val_score(model, X_train_scaled, y_train[target], 
                         scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)

print(f'RMSE = {-scores.mean():.2f} +/- ({scores.std():.2f})')
