
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np



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



class Imputation(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        return

    def fit(self, df, y=None):
        self.median_garagecars = df.GarageCars.median()
        self.median_BsmtUnfSF = df.BsmtUnfSF.median()
        self.median_TotalBsmtSF = df.TotalBsmtSF.median()
        self.median_BsmtFullBath = df.BsmtFullBath.median()
        return self

    def transform(self, df, y=None):
        df.GarageCars = df.GarageCars.fillna(self.median_garagecars)
        df.BsmtUnfSF = df.BsmtUnfSF.fillna(self.median_BsmtUnfSF)
        df.TotalBsmtSF = df.TotalBsmtSF.fillna(self.median_TotalBsmtSF)
        df.BsmtFullBath = df.BsmtFullBath.fillna(self.median_BsmtFullBath)
        df.Functional = df.Functional.fillna(0)
        return df

class Scaling(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.std_scale_feat = ['TotRmsAbvGrd', 'GrLivArea', 'LotArea', 'TotalBsmtSF'
        ]

        self.minmax_scale_feat = ['OverallQual', 'GarageCars', 'Bath', 'DecadeBuilt', 'Fireplaces', 'YrSold',
            'Neighborhood', 'WoodDeckSF', 'HeatingQC', 'BedroomAbvGr', 'OverallCond', 'BsmtFullBath', 
            'Porch', 'BsmtUnfSF', 'ExterQual', 'CentralAir', 'Functional', 'KitchenAbvGr'
        ]
        return
    def fit(self, df, y=None):
        self.std_scaler = StandardScaler().fit( 
            df[self.std_scale_feat].values.reshape(-1,len(self.std_scale_feat))
        )
        self.minmax_scaler = MinMaxScaler().fit( 
            df[self.minmax_scale_feat].values.reshape(-1,len(self.minmax_scale_feat))
        )
        return self
    def transform(self, df, y=None):
        ssf = self.std_scale_feat
        mmsf = self.minmax_scale_feat

        df[ssf] = self.std_scaler.transform( 
            df[ssf].values.reshape(-1,len(ssf))
        )
        df[mmsf] = self.minmax_scaler.transform( 
            df[mmsf].values.reshape(-1,len(mmsf))
        )
        return df
