# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:58:31 2024

@author: afisher
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


raw_df = pd.read_csv('Thyroid_Diff.csv')
# %%
# Separate target
df = raw_df.copy()
y = df.Recurred.copy()
X = df.drop(columns = ['Recurred'])

# Split into test and train
sss = StratifiedShuffleSplit(n_splits = 1, test_size = .7)
for j, (train_index, test_index) in enumerate(sss.split(X, y)):
    Xtest = X.iloc[test_index]
    ytest = y.iloc[test_index]
    Xtrain = X.iloc[train_index]
    ytrain = y.iloc[train_index]


# %% Exploratory
print(df.nunique()) #how many in each category
print(df.isnull().sum())

# %%
# convert gender to onehot
# What are categorical? 

### Dictionary mappings for column names
def risk_mapping(x):
    x.Risk = x.Risk.map(dict(zip(['Low', 'Intermediate', 'High'], [0, 1, 2])))
    return x
risk_transformer = FunctionTransformer(risk_mapping, validate=False)

def exam_mapping(x):
    x['Physical Examination'] = x['Physical Examination'].map(dict(zip(
        ['Single nodular goiter-left', 'Single nodular goiter-right'], ['Single nodular']*2
    )))
    return x
exam_transformer = FunctionTransformer(exam_mapping, validate=False)

### Columns with binary features
onehot_features = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Focality', 'M', 'Response', 'Physical Examination']

### Specific column orderings
thyroid_categories = [[
        'Clinical Hypothyroidism', 'Subclinical Hypothyroidism', 'Euthyroid',
        'Subclinical Hyperthyroidism', 'Clinical Hyperthyroidism'
    ]]
T_categories = [['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b']]
N_categories = [['N0', 'N1a', 'N1b']]
stage_categories = [['I', 'II', 'III', 'IVA', 'IVB']]


###
preprocessing = ColumnTransformer(
    [
        ('risk_transform',  risk_transformer,                               ['Risk']),
        ('exam_transform',  exam_transformer,                               ['Physical Examination']),
        
        ('Onehot',          OneHotEncoder(drop='if_binary'),                onehot_features),
        
        ('Thyroid Function',OrdinalEncoder(categories=thyroid_categories),  ['Thyroid Function']),
        ('T',               OrdinalEncoder(categories=T_categories),        ['T']),
        ('N',               OrdinalEncoder(categories=N_categories),        ['N']),
        ('Stage',           OrdinalEncoder(categories=stage_categories),    ['Stage']),
        
    ],
        remainder = 'drop'
    )

X_transformed = preprocessing.fit_transform(Xtrain)
print(X_transformed)

