# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 08:38:17 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import skew, norm


# Computing imputed values from test AND train?
# Create more features using log and square operators?
# Use blending


df = pd.read_csv('test.csv')



(mu, sigma) = stats.norm.fit(df.SalePrice)
sns.displot()
Z = np.linspace(-5,5,1000)
x = Z*sigma + mu

sns.kdeplot(data=df, x='SalePrice', label='Act. Dist.')
sns.lineplot(x=x, y=np.exp(-Z**2/2)/np.sqrt(2*np.pi*sigma**2), label='Norm Fit.')
plt.legend()


fig, ax =plt.subplots()
ax=sns.boxplot(data = df.select_dtypes(include=('int64','float64')),
            orient='h', palette='Set1')
ax.set_xscale('log')
ax.xlabel('Numeric Values')
ax.ylabel('Numeric Features')