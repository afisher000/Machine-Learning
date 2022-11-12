# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:11:56 2022

@author: afisher
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = pd.read_csv('data.csv')


def evaluate_model(features):
    X = data[features]
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f'Features = {features}\nError={mse:.3f}\n')

# Feature Engineering
data.sex = data.sex.map({'male':1, 'female':0})
data['agesex'] = data.sex*data.age
y = data.price


# Evaluate for different feature sets
evaluate_model(['age','sex'])
evaluate_model(['age','sex', 'agesex'])
evaluate_model(['age','agesex'])


