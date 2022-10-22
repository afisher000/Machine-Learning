# %% Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from Inspection import Inspection
from Models import ModelAnalysis
from Pipelines import FeatureEngineering, Imputation, Scaling


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y_train = np.log(1+df_train.SalePrice.values)
id_test = df_test.Id


# # %% Inspection
# ins = Inspection(df_train, target='SalePrice')


# %% Munging
pipeline = Pipeline([ 
    ('Features', FeatureEngineering()),
    ('Complex Imputation', Imputation()),
    # ('Simple Imputer', SimpleImputer(strategy='median')),
    ('Scaling', Scaling())
])
X_train = pipeline.fit_transform(df_train).values
X_test = pipeline.transform(df_test).values


# %% Model Selection

# model = SVR(cache_size=1000) #-0.126
# param_grid = {'gamma':np.logspace(-2,2,7), 'C':np.logspace(-1,3,6)}

# model = Ridge() #-0.162
# param_grid = {'alpha':np.logspace(-2,2,8)}

# model = RandomForestRegressor()
# param_grid = { 
#     'n_estimators':[50, 100, 150],
#     'min_samples_split':[2,8,12]
# }

base_model = SVR(cache_size=1000, gamma=.02, C=100)
model = AdaBoostRegressor(base_model)
param_grid = { 
    'n_estimators':[50, 150],
    'learning_rate':np.logspace(-1,1,5)
}


analysis = ModelAnalysis(model, X_train, y_train, param_grid=param_grid)

best_model = analysis.grid.best_estimator_
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_train)

df = pd.read_csv('train.csv')
df['Pred'] = y_pred
df['Target'] = np.log(1+df.SalePrice)
df['Error'] = df.Pred-df.Target
df.LotShape = df.LotShape.map({'Reg':3, 'IR1':2, 'IR2':1, 'IR3':0})
df.corr()['Error'].abs().sort_values(ascending=False)[:20]


# %% Create Submission

y_test = best_model.predict(X_test)
submission = pd.DataFrame({'Id':id_test, 'SalePrice':np.exp(y_test)-1})
submission.to_csv('submission.csv', index=False)


# %%
