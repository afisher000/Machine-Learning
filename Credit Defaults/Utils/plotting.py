# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:15:24 2024

@author: afisher
"""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_kde_by_class(col_name, df):
    ''' 
    Plots the kde distribution of each target outcome for input variable
        
    Parameters:
        col_name: name of column in df to plot.
        df: dataframe containing col_name and TARGET
    '''
   
    # Print correlation and medians
    mask = df['TARGET']==1

    corr = df['TARGET'].corr(df[col_name])    
    median_repaid = df.loc[mask, col_name].median()
    median_not_repaid = df.loc[~mask, col_name].median()
    
    print(f'The correlation between {col_name} and TARGET is {corr:0.4f}')
    print(f'Median value for REPAID loan: {median_repaid:0.4f}')
    print(f'Median value for NOT REPAID loan: {median_not_repaid:0.4f}')
    
    # Figure
    plt.figure(figsize=(12,6))
    
    sns.kdeplot(df.loc[mask, col_name], label='Paid')
    sns.kdeplot(df.loc[~mask, col_name], label='Not Paid')
    
    plt.xlabel(col_name)
    plt.ylabel('Density')
    plt.title('KDE distribution')
    plt.legend()
    
    return
    

def plot_feature_importances(df, nfeatures):
    # ASSUMES COLUMNS ARE "feature" and "importance"
    feature_col = 'feature'
    importance_col = 'importance'
    
    # Sort by importance
    df = df.sort_values(importance_col, ascending=False)
    
    # Normalize so they add up to 1
    df[importance_col] = df[importance_col]/df[importance_col].sum()
    
    # Plot with horizontal bars
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    ax.barh(
        list(df.index[:nfeatures])[::-1],
        df[importance_col].head(nfeatures),
        align='center', edgecolor='k',
        )
    ax.set_yticks = list(df.index[:nfeatures])[::-1]
    ax.set_yticklabels = df[feature_col].head(nfeatures)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()
    
    return df
    
    
    
    
