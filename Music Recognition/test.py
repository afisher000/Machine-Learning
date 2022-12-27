# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 19:11:53 2022

@author: afisher
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import utils_model as um


database = 'training_data_blobs.csv'
data = pd.read_csv(database)

y = data.state
X = data[um.training_columns]


rfc = RandomForestClassifier()
rfc.fit(X,y)
ypred = rfc.predict(X)

accuracy = sum(ypred==y)/len(y)
print(f'Accuracy = {accuracy*100:.1f}%')

