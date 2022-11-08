# %% Imports
import sys
sys.path.append('../../Datasets')
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

from RegressorAnalysis import RegressorModels
from Pipelines import Imputation, Scaling

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
y_train = np.log(1+train_df.SalePrice.values)
id_test = test_df.Id

# %% Pipelines 
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        return
    def fit(self, df, y=None):
        df['DecadeBuilt'] = df.YearBuilt//10*10
        self.median_hood_price = df.groupby('Neighborhood')['SalePrice'].median()
        self.median_decadebuilt_price = df.groupby('DecadeBuilt')['SalePrice'].median()
        return self

    def transform(self, df, y=None):
        
        # Old Examples        
        # df.RoofStyle = np.where(df.RoofStyle=='Gable', 1, 0)
        # df.ExterQual = df.ExterQual.map({'Ex':2, 'Gd':1, 'TA':0, 'Fa':-1, 'Po':-2})
        # df.Neighborhood = df.Neighborhood.map(self.median_price_by_hood)


        # New Features/Alteration
        df['DecadeBuilt'] = df.YearBuilt//10*10
        df['Bath'] = 1*df.FullBath + 0.5*df.HalfBath
        df['WoodDeckSF'] = np.log(1+df.WoodDeckSF)
        df['Porch'] = df.OpenPorchSF + df.EnclosedPorch + df['3SsnPorch'] + df.ScreenPorch
        

        # Encodings
        df.DecadeBuilt = df.DecadeBuilt.map(self.median_decadebuilt_price)
        df.Neighborhood = df.Neighborhood.map(self.median_hood_price)
        df.HeatingQC = df.HeatingQC.map({'Ex':2, 'Gd':1, 'TA':0, 'Fa':-1, 'Po':-2})
        df.ExterQual = df.ExterQual.map({'Ex':2, 'Gd':1, 'TA':0, 'Fa':-1, 'Po':-2})
        df.CentralAir = df.CentralAir.map({'Y':1, 'N':0})
        df.Functional = df.Functional.map({'Typ':0, 'Min1':1, 'Min2':2, 'Mod':3, 'Maj1':4, 'Maj2':5, 'Sev':6, 'Sal':7})

        # Truncations
        max_LivArea = 3500
        df.loc[df.GrLivArea>max_LivArea,'GrLivArea'] = max_LivArea
        max_LotArea = 30000
        df.loc[df.LotArea>max_LotArea, 'LotArea'] = max_LotArea
        max_BsmtArea = 2500
        df.loc[df.TotalBsmtSF>max_BsmtArea, 'TotalBsmtSF'] = max_BsmtArea

        # Drop unused Features
        keep_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'Bath', 'DecadeBuilt', 'TotRmsAbvGrd',
        'Fireplaces', 'Neighborhood', 'WoodDeckSF', 'HeatingQC', 'BedroomAbvGr', 'OverallCond',
        'LotArea', 'BsmtFullBath', 'TotalBsmtSF', 'Porch', 'BsmtUnfSF', 'YrSold', 
        'ExterQual', 'CentralAir', 'Functional', 'KitchenAbvGr']

        drop_features = list(set(df.columns)-set(keep_features))
        df.drop(columns=drop_features, inplace=True)

        return df



# %% Munging

# Imputation strategies are 'mean', 'median', 'most_frequent', 'bycategory', 'byvalue'
imputation_data = {
    # 'mean':[],
    'median':['GarageCars', 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath'], 
    # 'most_frequent':[],
    # 'bycategory':[],
    'byvalue':[('Functional', 0)]
}

scaling_data = { 
    # 'quantile':[],
    'standard':['TotRmsAbvGrd','GrLivArea','LotArea','TotalBsmtSF'],
    'minmax':['OverallQual','GarageCars','Bath','DecadeBuilt','Fireplaces','Neighborhood','WoodDeckSF',
        'HeatingQC','BedroomAbvGr','OverallCond','YrSold','BsmtFullBath','Porch','BsmtUnfSF','ExterQual',
        'CentralAir','Functional','KitchenAbvGr'
    ]
}


pipeline = Pipeline([ 
    ('Features', FeatureEngineering()),
    ('Imputation', Imputation(imputation_data)),
    ('Scaling', Scaling(scaling_data))
])
X_train = pipeline.fit_transform(train_df).values
X_test = pipeline.transform(test_df).values


# %% Model Selection

rm = RegressorModels(X_train, y_train, scoring='neg_root_mean_squared_error')
rm.gridsearch('ridge')
rm.gridsearch('svc', hyper_params = ['C', 'kernel'])
rm.gridsearch('randomforest', hyper_params = ['n_estimators'])

# %% Create Submission

y_test = best_model.predict(X_test)
submission = pd.DataFrame({'Id':id_test, 'SalePrice':np.exp(y_test)-1})
submission.to_csv('submission.csv', index=False)


# %%
