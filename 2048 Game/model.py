# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:08:53 2023

@author: afisher
"""

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

# Design scoring function with parameters


df = pd.read_csv('gamelog_3x3.csv', header=None)

# inputs = {'up':1, 'left':2, 'right':3, 'down':4}
# targets = df[9].map(inputs)
y_train = df[9].values
X_train = df.drop(columns=[9]).values


# %% Naively train models
estimator = RandomForestClassifier(n_estimators = 50)
#estimator = AdaBoostClassifier(n_estimators=100)



scores = cross_val_score(estimator, X_train, y_train)
print(f'Mean = {scores.mean()*100:.1f}% +/- {scores.std()*100:.1f}%')

model = estimator.fit(X_train, y_train)
joblib.dump(model, 'model.joblib')
