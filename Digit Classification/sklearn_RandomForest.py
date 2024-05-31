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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score

from sklearn.svm import SVC
plt.close('all')



# %% Read data, split into test/train
df = pd.read_csv('Datasets/expanded_digits.csv')
y = df['label'].values
X = df.drop(columns=['label']).values

    
# %% Plot random value
def plot_digit(X, idx):
    pixels = X[idx].reshape(16, 16)
    plt.imshow(pixels, cmap = mpl.cm.binary)
    return

# plot_digit(X_train, 0)

# %% Train and save model
# estimator = RandomForestClassifier(n_estimators=200, min_samples_split=5)
# estimator = SGDClassifier(alpha = 1e-4)
estimator = SVC(C = 10, kernel='rbf')

tune_hyperparameters = False
if tune_hyperparameters:
    
    param_grid = [{
        # 'n_estimators':[50, 150, 200], 
        # 'max_depth':[5, 10],
        # 'ccp_alpha':[0, 1e-1]
        }]
    
    # param_grid = [{'alpha':[5e-6, 1e-5, 5e-5]}]
    # param_grid = [{
        # 'kernel':['linear', 'poly','rbf'],
        # 'C':[1,5,10],
         # }]
    
    gridsearch = GridSearchCV(estimator = estimator, param_grid=param_grid, cv=5)
    gridsearch.fit(X, y)
    model = gridsearch.best_estimator_
    print(gridsearch.best_params_)
else:    
    model = estimator   
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')    
print(f'CV Accuracy: {cv_scores.mean()*100:.2f} ({cv_scores.std()*100:.2f})')

model.fit(X, y)
pickle.dump(model, open('Models/randomforest.pkl', 'wb'))



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
        incorrect_images.append( X[idx].reshape(16, 16) )
    
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
