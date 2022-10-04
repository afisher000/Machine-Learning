# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:09:17 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from time import time

df = pd.read_csv('train.csv')

features = [
    #'LotArea',
    #'LandContour',
    #'Neighborhood',
    'OverallQual',
    'OverallCond',
    'YearBuilt']

target = 'SalePrice'

y = df[target].values
X = df[features].values


##### How to reduce dimensions


## Partial Dependence Plots
st = time()
model = RandomForestClassifier(max_depth=2).fit(X,y)
print('%-20s: %.2fs' % ('Time to train model',time()-st) )

features = [0, 1]
PartialDependenceDisplay.from_estimator(model, X, features, target=0)



