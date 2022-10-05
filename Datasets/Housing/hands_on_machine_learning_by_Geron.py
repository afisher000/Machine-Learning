# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:27:34 2022

@author: afish
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

# TO IMPLEMENT:
    # List of common algorithms with pros/cons
    # Custom codes to implement them

##### Quick summary #####
### Data Analysis
### DataFrame functions
housing = pd.read_csv('housing.csv')
housing.head()
housing.info()
housing.describe()
housing.ocean_proximity.value_counts()
housing.isna().sum()

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


### Plots
housing.hist(bins=50, figsize=(20,15))
housing.plot.scatter(x='latitude', y='longitude', alpha=0.4,
                      s = housing.population/100, label='population',
                      c='median_house_value', cmap=plt.get_cmap('coolwarm'),
                      colorbar=True, figsize=(20,15))
scatter_matrix(housing[['median_house_value', 'median_income','total_rooms']])



##### Split test/training data #####
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=0)

### Alternate call with X,y
X_train, y_train, X_test, y_test = train_test_split(
    housing_X, housing_y, test_size=0.2, random_state=0
    )

### Suppose we wanted to ensure that we sampled from a categorical feature correctly
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in split.split(housing, housing.desired_category):
    strat_train_index = housing.loc[train_index]
    strat_test_index = housing.loc[test_index]
    


##### Clean and process the data #####

### DataFrame functions
housing.dropna(subset=['total_bedrooms']) #drop all entries with NA in subset
housing.drop('total_bedrooms', axis=1) #drop entire attribute
housing.total_bedrooms.fillna(housing.total_bedrooms.median(), inplace=True)

### Separate numerical and categorical
num_housing = housing.drop('ocean_proximity', axis=1)
cat_housing = housing.ocean_proximity

from sklearn.impute import SimpleImputer
X = SimpleImputer(strategy = 'median').fit_transform(num_housing) # returns np.array

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
X = OrdinalEncoder().fit_transform(cat_housing) # returns sparse matrix
X = OneHotEncoder().fit_transform(cat_housing)
### Can also do one-hot with pandas
pd.get_dummies(housing[['ocean_proximity']], prefix=['OP'], columns=['ocean_proximity'])

### Can create custom transformers, just include fit and transfrom methods
from sklearn.base import BaseEstimator, TransformerMixin
class custom_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, hyperparam=True):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # Do stuff here
        pass
        return
    
### Scale with minmax or standardization
from sklearn.preprocessing import MinMaxScaler, StandardScaler
X = MinMaxScaler().fit_transform(num_housing)
X = StandardScaler().fit_transform(num_housing)


##### Creating Pipelines #####
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
num_attributes = housing.columns.to_list()
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('custom', custom_transformer()),
    ('std_scaler', StandardScaler())
    ])
pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', OneHotEncoder(), cat_attributes)
    ])
housing_prepared = pipeline.fit_transform(housing)


##### Cross Validation ######
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_prepared, y, 
                         scoring='neg_mean_squared_error', cv=10)


##### Tuning Hyperparameters #####
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
param_grid = [
    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]}, #3x4 = 12 simulations
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]} #2*3 = 6 simulations
    ] #12 + 6 = 18 simulations in all

# If grid search is too long, use randomized search. 
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True) #18 * 5kfolds = 90 simulations
grid_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error',
                           return_train_score=True) #18 * 5kfolds = 90 simulations


##### Save model #####
from sklearn.externals import joblib
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')


##### Binary Classification #####
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
sgd_clf= SGDClassifier
sgd_clf.fit(X_train, y_train)
y_i = sgd_clf.predict([X_i])

forest_clf = RandomForestClassifier(random_state=0)

from sklearn.model_selection import cross_val_score, cross_val_predict
cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy') #returns scores of kfold
y_scores = cross_val_predict(model, X_train, y_train, cv=3, method='predict_proba' or 'decision function') #returns "clean" predictions using kfold

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, roc_curve
confusion_matrix(y_train, y_pred)
precision_score(y_train, y_pred)
recall_score(y_train, y_pred)
f1_score(y_train, y_pred)
# Plot precision and recalls vs thresholds or ROC curve
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
fpr, tpr, thresholds = roc_curve(y_train, y_scores)

# Confusion matrix = [[TN, FP],[FN, TP]]
# Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = TP/(TP+FP/2+FN/2)

# We can't change thresholds in model, but we can get decision scores
sgd_clf.decision_function(X)
cross_val_predict(model, X_train, y_train, cv=3, method='decision_function')


##### Multiclass Classification #####
# Some algorithms handle multiclass directly (Random Forest, naive Bayes)
# But any binary classifier can do multiclass using OvO (N*(N-1)/2) or OvA (N)
# OvA is fewer classifiers, but larger data samples. This is most common.
# With Support Vector Machines (scale poorly with sample size), OvO with smaller samples is better

# Sklearn chooses best by default, but if you want to force it...
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))

# Look at confusion matrix, scale to percent error, subtract diagonal if needed
conf_mx = confusion_matrix(y_train, y_pred)
norm_conf_mx = conf_mx / conf_mx.sum(axis=1, keepdims=True)
np.fill_diagonal(norm_conf_mx, 0)

### Can also use multiple labels (targets)
from sklearn.neighbors import KNeighborsClassifier
y_train = np.c_[target1, target2]
knn_clf.fit(X_train, y_train)
pred_targets = knn_clf.predict([Xi])


