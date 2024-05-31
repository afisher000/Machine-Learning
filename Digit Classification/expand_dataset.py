# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:54:33 2024

@author: afisher
"""


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import Utils.opencv as ucv
plt.close('all')

df = pd.read_csv('Datasets/digits.csv')


y = df['label'].values
X = df.drop(columns=['label']).values

# %% Expand dataset by applying shifts and angles
angles = np.linspace(-15, 15, 5)
xshifts = np.linspace(-1, 1, 3)

# New labels
rep_factor = len(angles)*len(xshifts)+1
y_new = np.repeat(y, rep_factor)
X_new = np.zeros((len(X)*rep_factor, 256))

for j in [35]: # range(len(X)):  
    
    # Copy original
    X_new[j*rep_factor] = X[j] 
    
    # Reshape into image
    image = X[j].reshape((16,16))

    for jangle, angle in enumerate(angles):
        rotated_image = ucv.rotate_image(image, angle)
        
        for jshift, xshift in enumerate(xshifts):
            idx = 1 + jshift + jangle*len(xshifts) + j*rep_factor    
            
            shifted_image = ucv.xshift_image(rotated_image, xshift)
            X_new[idx] = shifted_image.reshape(1, -1)
  
# %%
# plt.matshow(shifted_image, cmap=mpl.cm.binary)
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# plt.savefig(f'Website Pictures/2_transforms/angle={angle}_shift={xshift}.jpg')
            
            
            
# %%
# data = pd.DataFrame(index=y_new, data=X_new)
# data = data.reset_index().rename(columns={'index':'label'})
# data.to_csv('Datasets/expanded_digits.csv', index=False)
