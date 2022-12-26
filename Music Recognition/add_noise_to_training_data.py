# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 20:07:10 2022

@author: afisher
"""

import pandas as pd
import numpy as np

data = pd.read_csv('training_data_contours1.csv')
state = data.state.copy()
params = data.drop(columns='state')
datas = []
for j in range(1):
    temp_data = params *  (1+0.01*np.random.randn(*params.shape))
    temp_data['state'] = state
    datas.append(temp_data.copy())

pd.concat(datas).to_csv('training_data_contours.csv', index=False)