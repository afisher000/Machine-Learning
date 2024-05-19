# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:47:50 2024

@author: afisher
"""

import pandas as pd
import numpy as np

def agg_numeric(df, df_name, group_var, agg_fcns=['count', 'mean', 'max', 'min', 'sum']):
    # Remove all IDs other than group_var
    drop_cols = [col for col in df if col!=group_var and 'SK_ID' in col]
    df = df.drop(columns = drop_cols)
    
    # Aggregate
    agg = df.groupby(group_var).agg(agg_fcns)
    
    # Rename columns, remove end _ and fill spaces
    agg.columns = [f'{df_name}_{col[0]}_{col[1]}'.rstrip('_').replace(' ', '_') for col in agg.columns]

    # Replace group_var column
    agg.reset_index(inplace=True)
    
    return agg

def agg_categorical(df, df_name, group_var, cat_limits=None, agg_fcns=['mean']):
    # Limit categories as specified
    if cat_limits is not None:
        df = group_small_categories(df, cat_limits)
    
    # Convert to dummy counts
    encoded = pd.get_dummies(df.select_dtypes(['object', 'category']))
    encoded = encoded.set_index(df[group_var]).reset_index()
    
    # Aggregate
    agg = agg_numeric(encoded, df_name, group_var, agg_fcns=agg_fcns)
        
    return agg

def group_small_categories(df, cat_limits):
    # Dictionary cat_limits specifies columns and number of groups
    for col, n in cat_limits.items():
        cats = df[col].value_counts().head(n).index.tolist()
        df[col] = df[col].apply(lambda x: x if x in cats else 'other')
    return df
    
def missing_values_table(df):
    missing_values = df.isnull().sum()
    missing_percent= 100 * missing_values / len(df)
    
    missing_table = pd.concat([missing_values, missing_percent], axis=1)
    missing_table = (
        missing_table.rename(columns = {0:'Total Missing', 1:'Percent Missing'})
        .sort_values(by='Percent Missing', ascending=False)
        .round(1)
        )
    return missing_table


def remove_collinear(df, threshold=.8):
    corrs = df.corr()
    
    # Only count above diagonal
    col_set = set()
    xs, ys = np.where(corrs>threshold)
    for x, y in zip(xs, ys):
        if y>x:
            col_set.add(y)
            
    cols_to_remove = df.columns[list(col_set)]
    return df.drop(columns=cols_to_remove)