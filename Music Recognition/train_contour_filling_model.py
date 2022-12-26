# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 18:39:26 2022

@author: afisher
"""
# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_ellipse_goodness, convert_to_jpg
import pickle
from utils_contours import select_contours, get_contour_data
from utils import show_image, get_staffline_separation, pdf2jpg
from utils_music_cleaning import split_music_into_lines, remove_words_from_line
from sklearn.svm import SVC

import os
# To implement 



# Read file, separate into lines
song_file = 'Songs\\fairest_lord_jesus.pdf'
if song_file.endswith('.pdf'):
    pdf2jpg(song_file)
song_file = song_file[:-4]+'.jpg'

_, orig = cv.threshold(cv.imread(song_file, 0), 127, 255, 0)
imgs, line_sep = split_music_into_lines(orig)
cleaned_imgs = [remove_words_from_line(img, line_sep) for img in imgs]

# Load or create contour dataframe
if os.path.exists('training_data_contours.csv'):
    contour_data = pd.read_csv('training_data_contours.csv')
else:
    contour_data = pd.DataFrame(columns=['state', 'cx', 'cy', 'area', 'x','y', 'width','height','aspectratio','extent','solidity', 'normangle'])

save_status=True
for img in cleaned_imgs:
    print('next image')
    if not save_status:
        break

    # Cut line into slices
    ptr = 0
    xmax = 1500
    xoverlap = 100
    while ptr<img.shape[1]:

        # Get slice
        if ptr+xmax>img.shape[1]:
            gray = img[:, ptr:]
        else:
            gray = img[:, ptr:ptr+xmax]
        ptr += xmax-xoverlap

        # Select contours
        gray, save_status = select_contours(gray)

        if save_status:
            # Create selection masks
            selected_mask = np.where(np.logical_and(gray!=0, gray!=255), 255, 0).astype('uint8')
            unselected_mask = np.where(gray==255, 255, 0).astype('uint8')

            # Add data and save
            get_contour_data(selected_mask, line_sep, contour_df=contour_data, is_selected=True)
            get_contour_data(unselected_mask, line_sep, contour_df=contour_data, is_selected=False)
            contour_data.to_csv('training_data_contours.csv', index=False)
        else:
            break


ytrain = contour_data.state.astype(int)
Xtrain = contour_data.drop(columns=['state','cx','cy', 'x', 'y']).values
svc = SVC(gamma=100, C=1)
svc.fit(Xtrain, ytrain)
ypred = svc.predict(Xtrain)

pickle.dump(svc, open('model_to_fill_notes.pkl', 'wb'))






# %%
