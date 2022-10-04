# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 18:05:52 2022

@author: afisher
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import PartialDependenceDisplay
from time import time

### Linear model (1 variable)
xdata = np.linspace(0, 10, 100) 
ydata = 3*xdata + 5 + np.random.normal(0, 1, size=xdata.shape)

X = xdata.reshape(-1,1) # Must be 2 dimensional
y = ydata

model = sklearn.linear_model.LinearRegression()
model.fit(X,y)

xnew = xdata.reshape(-1,1)
ynew = model.predict(xnew)

fig, ax = plt.subplots()
plt.scatter(X,y, label='Data')
plt.scatter(xnew, ynew, label='Fit')

print('Coefficients: ' + str(model.coef_))
print('Intercept: ' + str(model.intercept_))