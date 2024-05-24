# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:23:36 2024

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
plt.close('all')



# %% Read data, split into test/train
df = pd.read_csv('Datasets/expanded_digits.csv')
y = df['label']
X = df.drop(columns=['label'])

sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)

for train_mask, test_mask in sss.split(X, y):
    X_train, y_train = X.loc[train_mask].values, y.loc[train_mask].values
    X_test, y_test = X.loc[test_mask].values, y.loc[test_mask].values
    
    
# %% Plot random value
def plot_digit(X, idx):
    pixels = X[idx].reshape(16, 16)
    plt.imshow(pixels, cmap = mpl.cm.binary)
    return

# plot_digit(X_train, 0)

# %% Train and save model
estimator = RandomForestClassifier(n_estimators=400, min_samples_split=5)

tune_hyperparameters = False
if tune_hyperparameters:
    
    param_grid = [{'n_estimators':[50, 150, 400], 'max_depth':[5, 10]}]
    gridsearch = GridSearchCV(estimator = estimator, param_grid=param_grid, cv=5)
    gridsearch.fit(X_train, y_train)
    model = gridsearch.best_estimator_
else:    
    model = estimator
    model.fit(X_train, y_train)
    
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'CV Accuracy: {cv_scores.mean()*100:.2f} ({cv_scores.std()*100:.2f})')

pickle.dump(model, open('Models/randomforest.pkl', 'wb'))


# %% Print Accuracy (do not tune model to optimize test_accuracy)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_pred_train, y_train)
test_accuracy = accuracy_score(y_pred_test, y_test)

print(f'Training Accuracy = {train_accuracy*100:.2f}')
print(f'Testing Accuracy = {test_accuracy*100:.2f}')

# %% Plot feature importances
def plot_pixel_importances(feature_importances):    
    plt.matshow(feature_importances.reshape(16, 16), cmap=mpl.cm.binary)
    return

# plot_pixel_importances(model.feature_importances_)

# %% Plot incorrect assignments

def plot_misclassifications(y_pred_test, y_test):
    incorrect_images = []
    error_mask = y_pred_test!=y_test
    
    for idx in np.nonzero(error_mask)[0]:
        incorrect_images.append( X_test[idx].reshape(16, 16) )
    
    plt.matshow( np.hstack(incorrect_images))
    plt.xticks(8+16*np.arange(len(incorrect_images)), y_pred_test[error_mask])
    plt.yticks([])
    return

# plot_misclassifications(y_pred_test, y_test)

# %% Show confusion matrix
def plot_confusion_matrix(y_pred_test, y_test):
    conf_matrix = confusion_matrix(y_pred_test, y_test)
    
    norm_matrix = conf_matrix/conf_matrix.sum(axis=1)
    np.fill_diagonal(norm_matrix, 0)
    
    plt.matshow(norm_matrix, cmap=mpl.cm.binary)

    return
