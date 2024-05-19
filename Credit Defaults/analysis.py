# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:50:29 2024

@author: afisher
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

import Utils.plotting as up
import Utils.feature_engineering as ufe
import Utils.memory as umem
import Utils.model as umod


pd.set_option('display.max_columns', 20) 
warnings.filterwarnings('ignore') #ignore pandas warnings
plt.style.use('fivethirtyeight')

# from utils_plotting import plot_kde_by_class
# from utils_analysis import agg_numeric, agg_categorical, group_small_categories, missing_values_table, remove_collinear


# %% Load, convert dtypes to reduce memory
data_folder = 'Datasets'

test = pd.read_csv( os.path.join(data_folder, 'application_test.csv') )
train = pd.read_csv( os.path.join(data_folder, 'application_train.csv') )
bureau = pd.read_csv( os.path.join(data_folder, 'bureau.csv') )
bureau_balance = pd.read_csv( os.path.join(data_folder, 'bureau_balance.csv') )

test, train, bureau, bureau_balance = umem.convert_types(test, train, bureau, bureau_balance)

# %% Compute previous_loan_counts from bureau
previous_loan_counts = (
    bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU']
    .count()
    .rename(columns={'SK_ID_BUREAU':'previous_loan_counts'})
    )

train = train.merge(previous_loan_counts, on='SK_ID_CURR', how='left')
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)

test = test.merge(previous_loan_counts, on='SK_ID_CURR', how='left')
test['previous_loan_counts'] = test['previous_loan_counts'].fillna(0)

umem.delete_from_memory('previous_loan_counts')

# %% compute aggregations from bureau
cat_limits = {'CREDIT_TYPE':3}
bureau_cat_agg = ufe.agg_categorical(bureau, 'bureau', 'SK_ID_CURR', cat_limits=cat_limits)
bureau_num_agg = ufe.agg_numeric(bureau, 'bureau', 'SK_ID_CURR')

train = train.merge(bureau_num_agg, on='SK_ID_CURR', how='left')
train = train.merge(bureau_cat_agg, on='SK_ID_CURR', how='left')

test = test.merge(bureau_num_agg, on='SK_ID_CURR', how='left')
test = test.merge(bureau_cat_agg, on='SK_ID_CURR', how='left')

umem.delete_from_memory('bureau_cat_agg', 'bureau_num_agg')

# %% compute aggregations from bureau_balance
bureau_balance_cat_agg = ufe.agg_categorical(bureau_balance, 'bureau_balance', 'SK_ID_BUREAU')
bureau_balance_num_agg = ufe.agg_numeric(bureau_balance, 'bureau_balance', 'SK_ID_BUREAU')

bureau_by_loan = bureau_balance_num_agg.merge(bureau_balance_cat_agg, on='SK_ID_BUREAU', how='left')
bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how='left')
bureau_balance_by_client = ufe.agg_numeric(bureau_by_loan, group_var='SK_ID_CURR', df_name='client')

train = train.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')
test = test.merge(bureau_balance_by_client, on='SK_ID_CURR', how='left')

umem.delete_from_memory('bureau_balance_cat_agg', 'bureau_balance_num_agg', 'bureau_by_loan', 'bureau_balance_by_client')


# %% Example of KDE plots
plt.close('all')
up.plot_kde_by_class('EXT_SOURCE_3', train)
up.plot_kde_by_class('previous_loan_counts', train)

# 
new_corrs = train[bureau_num_agg.columns].corrwith(train['TARGET']).sort_values(ascending=False, key = lambda x: abs(x))
print(new_corrs[:10])
up.plot_kde_by_class(new_corrs.index[0], train)

# %% Align test and train before data
train_labels = train['TARGET']
train, test = train.align(test, join='inner', axis=1)
train['TARGET'] = train_labels

train.to_csv( os.path.join(data_folder, 'train_added_features.csv'), index=False)
test.to_csv( os.path.join(data_folder, 'test_added_features.csv'), index=False)

# %% Remove collinear variables 
train_nocollinear = ufe.remove_collinear(train, threshold = 0.8)
test_nocollinear = ufe.remove_collinear(test, threshold = 0.8)
train_nocollinear, test_nocollinear = train_nocollinear.align(test_nocollinear, join='inner', axis=1)

train_nocollinear.to_csv( os.path.join(data_folder, 'train_removed_collinear.csv'), index=False)
test_nocollinear.to_csv( os.path.join(data_folder, 'test_removed_collinear.csv'), index=False)
       
    
# %% Train model
umod.model(train, test, 'SK_ID_CORR', 'TARGET', n_folds = 5)




