# %% Imports
from re import X
from tkinter import Y
from unicodedata import numeric
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Examples of analysis from kaggle workbooks

def plot_nanpct(df, thresh=0.05, ax=None):
    nanpct = (df.isna().sum()/len(df)*100).sort_values(ascending=False)
    nanpct = nanpct[nanpct>thresh]
    
    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(x=nanpct.index, y=nanpct.values, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Percent Nan')
    ax.set_title(f'Data with >{thresh*100:.0f}% NaN')
    return nanpct

def plot_correlations(df, target, thresh=0.3, ax=None):
    corrs = df.corr(numeric_only=True)[target].abs().sort_values(ascending=False)
    corrs = corrs[corrs>thresh]

    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(x=corrs.index, y=corrs.values, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel(f'Abs Correlation with {target}')
    ax.set_title(f'Features with rho>{thresh:.2f}')
    return corrs

def plot_skews(df, n=20, ax=None):
    skews = df.skew(numeric_only=True).abs().sort_values(ascending=False)[:n]

    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(x=skews.index, y=np.log1p(skews.values), ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel(f'log1p of skew')
    ax.set_title(f'Skew of feature distributions')
    return skews

def compare_with_logdist(df, column, target):
    logcolumn = 'log' + column
    df[logcolumn] = np.log1p(df[column])
    fig, ax = plt.subplots(nrows=2, ncols=2)
    plot_regression(df, column, target, ax[0,0])
    plot_regression(df, logcolumn, target, ax[0,1])
    plot_kde_distribution(df, column, ax[1,0])
    plot_kde_distribution(df, logcolumn, ax[1,1])

    return 


def plot_regression(df, xcol, ycol, ax=None):
    pearson = df[xcol].corr(df[ycol])

    if ax is None:
        fig, ax = plt.subplots()
    sns.regplot(data=df, x=xcol, y=ycol, scatter_kws={'alpha':.2}, ax=ax)
    ax.legend([f'Pearson = {pearson:.2f}'])

def plot_distributions_as_boxplot(df, columns):
    fig, ax =plt.subplots()
    ax=sns.boxplot(data = df[columns],
                orient='h', palette='Set1')
    ax.set_xscale('log')
    ax.set_xlabel('Numeric Values')
    ax.set_ylabel('Numeric Features')

def plot_valuecounts_as_pie(series):
    valcounts = series.value_counts()/len(series)*100
    valcounts['NA'] = series.isna().sum()/len(series)*100
    valcounts.plot.pie(autopct='%.0f%%')

def plot_kde_distribution(df, column, ax=None):
    (mu, sigma) = stats.norm.fit(df[column])
    Z = np.linspace(-5,5,1000)
    x = Z*sigma + mu

    if ax is None:
        fig, ax = plt.subplots()
    sns.kdeplot(data=df, x=column, label='Dist.', ax=ax)
    sns.lineplot(x=x, y=np.exp(-Z**2/2)/np.sqrt(2*np.pi*sigma**2), label='Gaussian Fit', ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.legend()

def plot_largecategorypct(df, n=10, ax=None):
    largest_pct = pd.Series( 
        {col:df[col].value_counts().max()/df[col].notna().sum()
        for col in df.select_dtypes('object')}
    ).sort_values(ascending=False)[:n]

    if ax is None:
        fig, ax = plt.subplots()
    sns.barplot(x=largest_pct.index, y=largest_pct.values, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylabel('Largest Category Pct')
    ax.set_title(f'Top {n} results')
    return largest_pct
# %% Saleprice by OverallQual
# plot_regression(df, 'GrLivArea', 'SalePrice')
# ax = sns.boxplot(data=df, x='OverallQual', y='SalePrice')
# ax = sns.lineplot(data=df, x='OverallQual', y='SalePrice', hue='LotShape')
