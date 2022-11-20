from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

class FeatureTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, gui, file):
        self.gui = gui
        self.file = file

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        with open(self.file,'r') as f:
            code = f.read()
        local_dict = {'df':df}
        exec(code, globals(), local_dict)
        return local_dict['df']



class Imputation(BaseEstimator, TransformerMixin):
    '''Pass imputation instructions through a dictionary parameter. Simple mean and median 
    imputation needs to be implemented. You can also impute according to another category.'''
    def __init__(self, data):
        self.data = data
        self.mean_imputer = SimpleImputer(strategy='mean')
        self.median_imputer = SimpleImputer(strategy='median')
        self.frequency_imputer = SimpleImputer(strategy='most_frequent')
        return
    def fit(self, df, y=None):
        for strategy, features in self.data.items():
            if strategy =='median':
                self.median_imputer.fit(df[features])
            elif strategy == 'mean':
                self.mean_imputer.fit(df[features])
            elif strategy =='most_frequent':
                self.frequency_imputer.fit(df[features])
            elif strategy == 'bycategory':
                for na_feature, category in features:
                    values_by_category = (df[df[na_feature].notna()]
                        .groupby(by=category)
                        .median(numeric_only=True)[na_feature]
                    )
                    setattr(self, f'{na_feature}_by_{category}', values_by_category)
        return self

    def transform(self, df, y=None):
        row_indexer = df.index!=-1000
        for strategy, features in self.data.items():
            if strategy =='median':
                df[features] = self.median_imputer.transform(df[features])
            elif strategy == 'mean':
                df[features] = self.mean_imputer.transform(df[features])
            elif strategy =='most_frequent':
                df[features] = self.frequency_imputer.transform(df[features])
            elif strategy == 'bycategory':
                for na_feature, category in features:
                    values_by_category = getattr(self, f'{na_feature}_by_{category}')
                    df.loc[row_indexer,[na_feature]] = np.where(
                        df[na_feature].isna(),
                        values_by_category.loc[df[category]], # Median values by category
                        df[na_feature]
                    )
            elif strategy == 'byvalue':
                for na_feature, value in features:
                    df.loc[row_indexer,[na_feature]] = df[na_feature].fillna(value)

        return df


class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self, data):
        self.data = data
        self.minmax_scaler = MinMaxScaler()
        self.standard_scaler = StandardScaler()
        self.quantile_scaler = QuantileTransformer()
        return

    def fit(self, df, y=None):
        for scale_type, features in self.data.items():
            if scale_type=='minmax':
                self.minmax_scaler.fit(df[features])
            elif scale_type=='standard':
                self.standard_scaler.fit(df[features])
            elif scale_type=='quantile':
                self.quantile_scaler.fit(df[features])  
        return self

    def transform(self, df, y=None):
        row_indexer = df.index!=-1000
        for scale_type, features in self.data.items():
            if scale_type=='minmax':
                df.loc[row_indexer,features] = self.minmax_scaler.transform(df[features])
            elif scale_type=='standard':
                df.loc[row_indexer,features] = self.standard_scaler.transform(df[features])
            elif scale_type=='quantile':
                df.loc[row_indexer,features] = self.quantile_scaler.transform(df[features])  
        return df


class KeepSelectedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df = df[self.features]
        nan_features = df.columns[df.isna().sum()>0].to_list()
        if len(nan_features)>0:
            raise ValueError(f'Null values in {", ".join(nan_features)}')
        return df[self.features]