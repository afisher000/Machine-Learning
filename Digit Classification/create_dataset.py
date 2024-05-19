# -*- coding: utf-8 -*-
"""
Created on Sat May 18 13:38:08 2024

@author: afisher
"""

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import Utils.opencv as ucv

plt.close('all')

        
# %% Read image

filename = 'Images/digits'
gray = cv.imread( filename+'.jpg', cv.IMREAD_GRAYSCALE)
_, binary_image = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

# %% Find index of lines
col_black_percent = binary_image.sum(axis=0)/binary_image.shape[0]/255
row_black_percent = binary_image.sum(axis=1)/binary_image.shape[1]/255

col_peaks, _ = find_peaks(col_black_percent, prominence = 0.4, distance = 10)
row_peaks, _ = find_peaks(row_black_percent, prominence = 0.4, distance = 10)

num_cols = len(col_peaks)-1
num_rows = len(row_peaks)-1

print(f'There are {num_rows} rows and {num_cols} columns')

# %% Loop over digits, clean images
pixels = np.zeros((num_rows*num_cols, 256))
bool_mask = np.array([True]*(num_rows*num_cols))
labels = np.repeat(list('1234567890'), num_cols)

for jrow in range(num_rows):
    for jcol in range(num_cols):     
        j = jcol + jrow*num_cols
        # Use try/except to skip bad images
        try:
            image = binary_image[row_peaks[jrow]:row_peaks[jrow+1],
                                 col_peaks[jcol]:col_peaks[jcol+1]]
            
            # Remove black edges
            image = ucv.flood_fill_edges(image)
            
            # Remove small spots
            image = ucv.remove_noise(image)
            
            # Scale image
            image = ucv.scale_image(image)
            
            # Save pixels
            pixels[j] = image.reshape(1, -1)/255
        except:
            # Set boolean to false
            bool_mask[j] = False

print(f'There are {int(sum(bool_mask))} good images')

# %% Save to dataframe

# Ordered digits in 10 rows so labels known
if num_rows==10:
    data = pd.DataFrame(index=labels[bool_mask], data=pixels[bool_mask])
    data = data.reset_index().rename(columns={'index':'label'})
else:
    data = pd.DataFrame(data=pixels[bool_mask])
    
data.to_csv(filename+'.csv', index=False)