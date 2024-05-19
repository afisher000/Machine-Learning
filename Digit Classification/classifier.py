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
df = pd.read_csv('Data/expanded_digits.csv')
y = df['label']
X = df.drop(columns=['label'])

sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)

for train_mask, test_mask in sss.split(X, y):
    X_train, y_train = X.loc[train_mask].values, y.loc[train_mask].values
    X_test, y_test = X.loc[test_mask].values, y.loc[test_mask].values
    
# %% Plot random value
pixels = X_train[0].reshape(16, 16)
# plt.imshow(pixels, cmap = mpl.cm.binary)

# %% Train and save model

hyperparam_tuning = False

estimator = RandomForestClassifier(n_estimators=400, min_samples_split=5)
if hyperparam_tuning:
    
    param_grid = [{
            'n_estimators':[50, 100, 200], 'max_depth':[10]
        }]
    gridsearch = GridSearchCV(estimator = estimator, param_grid=param_grid, cv=5)
    gridsearch.fit(X_train, y_train)
    model = gridsearch.best_estimator_
else:    
    model = estimator
    model.fit(X_train, y_train)
    
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'CV Accuracy: {cv_scores.mean()*100:.2f} ({cv_scores.std()*100:.2f})')

pickle.dump(model, open('Models/randomforest.pkl', 'wb'))


# %% Print Accuracy
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_pred_train, y_train)
test_accuracy = accuracy_score(y_pred_test, y_test)

print(f'Training Accuracy = {train_accuracy*100:.2f}')
print(f'Testing Accuracy = {test_accuracy*100:.2f}')

# %% Plot feature importances
# feature_importances = model.feature_importances_.reshape(16, 16)
# plt.matshow(feature_importances, cmap=mpl.cm.binary)


# %% Plot incorrect assignments
# incorrect_images = []
# error_mask = y_pred_test!=y_test

# for idx in np.nonzero(error_mask)[0]:
#     incorrect_images.append( X_test[idx].reshape(16, 16) )

# plt.matshow( np.hstack(incorrect_images))
# plt.xticks(8+16*np.arange(len(incorrect_images)), y_pred_test[error_mask])
# plt.yticks([])



# %% Show confusion matrix
# conf_matrix = confusion_matrix(y_pred_test, y_test)

# norm_matrix = conf_matrix/conf_matrix.sum(axis=1)
# np.fill_diagonal(norm_matrix, 0)

# plt.matshow(norm_matrix, cmap=mpl.cm.binary)


