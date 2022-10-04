# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 17:35:50 2022

@author: afisher
"""

from sklearn.datasets import make_moons, make_circles, make_classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# TO IMPLEMEMT:
    # Not all of these can be 2D? Separate from higher order datasets?
    
dataset_type = 'classification'

### Moon Dataset
if dataset_type=='moons':
    X, y = make_moons(n_samples=(50, 60), shuffle=True, noise=0.1)
    
    df = pd.DataFrame(
        data=np.hstack([X,y.reshape(-1,1)]),
        columns=['p1','p2','state']
        )
    
    cmap = plt.get_cmap('coolwarm')
    df.plot.scatter(x='p1', y='p2', c='state', cmap=cmap)

### Circles Dataset
if dataset_type=='circles':
    X, y = make_circles(n_samples=(500, 600), shuffle=True, noise=0.1, factor=0.1)
    
    df = pd.DataFrame(
        data=np.hstack([X,y.reshape(-1,1)]),
        columns=['p1','p2','state']
        )
    
    cmap = plt.get_cmap('coolwarm')
    df.plot.scatter(x='p1', y='p2', c='state', cmap=cmap)
    
### Classification
if dataset_type=='classification':
    X, y = make_classification(n_samples = 1000, n_classes=4, n_features=6, n_informative=4, n_redundant=2)
    df = pd.DataFrame(
        data=np.hstack([X,y.reshape(-1,1)]),
        columns=['p1','p2','p3','p4','p5','p6','state']
        )
    
    cmap = plt.get_cmap('coolwarm')
    df.plot.scatter(x='p1', y='p2', c='state', cmap=cmap)

else:
    print('dataset_type is not valid.')