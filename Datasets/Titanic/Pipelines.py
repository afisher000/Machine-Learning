
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class Imputation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        return
    def fit(self, df, y=None):
        self.Pclass_median_fares = ( 
            df[df.Fare!=0]
            .groupby(by='Pclass')
            .median(numeric_only=True)['Fare']
            )
        
        self.Pclass_mean_ages = ( 
            df[~df.Age.isna()]
            .groupby(by='Pclass')
            .mean(numeric_only=True)['Age']
        )
        return self

    def transform(self, df, y=None):
        df.Fare = np.where( 
            (df.Fare==0) | df.Fare.isna(),
            self.Pclass_median_fares.loc[df.Pclass],
            df.Fare
        )
        
        df.Age = np.where( 
            df.Age.isna(),
            self.Pclass_mean_ages.loc[df.Pclass],
            df.Age
        )

        return df

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        return
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        df['Family_Size'] = df.SibSp + df.Parch
        df.Sex = df.Sex.map({'male':0, 'female':1})
        return df

class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self, std_scale_features=None, qnt_scale_features=None):
        self.std_scale_features = std_scale_features
        self.qnt_scale_features = qnt_scale_features
        return
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        ssf = self.std_scale_features
        qsf = self.qnt_scale_features
        df[ssf] = StandardScaler().fit_transform( 
            df[ssf].values.reshape(-1,len(ssf))
        )
        df[qsf] = QuantileTransformer(n_quantiles=100).fit_transform( 
            df[qsf].values.reshape(-1, len(qsf))
        )
        return df

class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, keep_features=None):
        self.keep_features = keep_features
        return
    def fit(self, df, y=None):
        return self
    def transform(self, df, y=None):
        df = df[self.keep_features]
        return df