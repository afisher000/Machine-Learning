# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:13:08 2024

@author: afisher
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:57:11 2024

@author: afisher
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


# %% Read all datasets and concatenate
dataset_folder = 'Datasets/Accidentals'

dfs = [pd.read_csv( os.path.join(dataset_folder, file)) 
       for file in os.listdir(dataset_folder)]

df = pd.concat(dfs)

# %% Split into testing and training
training_columns = [
    'area','width','height','aspectratio','extent','solidity','angle', 'faults'
    ]

 
X = df[training_columns]
y = df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# %% Make pipeline
classifier = Pipeline([ 
        ('model', RandomForestClassifier())
    ])

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


print( f'Confusion matrix:\n{confusion_matrix(y_pred, y_test)}' )

# %% Feature importances
feature_importances = classifier['model'].feature_importances_

plt.bar(np.arange(len(feature_importances)), feature_importances)
plt.xticks(ticks = np.arange(len(X_train.columns)), labels=X_train.columns, rotation=45)

# %% Save model
pickle.dump(classifier, open('Saved Models/accidentals.pkl', 'wb'))


