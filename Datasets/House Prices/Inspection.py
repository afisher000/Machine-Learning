import pandas as pd
import numpy as np

def compute_outliers(array):
    Z = (array - array.mean())/array.std()
    outliers = array[abs(Z)>4].sort_values()
    return outliers


class Inspection():
    def __init__(self, df, target=None):
        self.df = df
        self.target = target
        self.num_features = [col for col in df.columns if df[col].unique().shape[0]>50]
        self.cat_features = [col for col in df.columns if df[col].unique().shape[0]<=50]
        self.correlations = self.get_correlations()
        self.largest_category_pct = self.get_largest_category_pct()
        self.na_fractions = self.get_nafractions()
        return

    def examine(self, column):
        self.get_nafractions(column=column)
        if column in self.num_features:
            self.get_outliers(column=column)
            self.df[column].hist(bins=30)
        else:
            self.df[column].hist()

    def get_correlations(self, n=10):
        if not self.target:
            return
        correlations = self.df.corr(numeric_only=True)[self.target].sort_values(ascending=False)
        print(correlations[:n])
        return correlations


    def get_skews(self, n=10):
        skews = self.df[self.num_features].skew().abs().sort_values(ascending=False)
        print(skews[:n])
        return skews

    def get_nafractions(self, n=10, column=[]):
        if column:
            na_fraction = self.df[column].isna().sum()/len(self.df)
            print(f'{na_fraction}% null values')
            return na_fraction
        else:
            na_fractions = (self.df.isna().sum()/len(self.df)).sort_values(ascending=False)
            print(na_fractions[:n])
            return na_fractions

    def get_outliers(self, n=10, column=[]):
        if column:
            outliers = compute_outliers(self.df[column])
            print(f'{len(outliers)} total outliers:')
            print(outliers.values)
            return outliers
        else:
            outliers = pd.Series( 
                {col:compute_outliers(self.df[col]).shape[0] for col in self.num_features}
            ).sort_values(ascending=False)
            print(outliers[:n])
            return outliers


    def get_largest_category_pct(self, n=10):
        largest_category_pct = pd.Series( 
            {col:self.df[col].value_counts().max()/self.df[col].notna().sum()
            for col in self.cat_features}
        ).sort_values(ascending=False)
        print(largest_category_pct[:n])
        return largest_category_pct

    

    




