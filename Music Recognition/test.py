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
import utils_io as uio
import utils_music_munging as umm
import cv2 as cv
from scipy import ndimage

# Open song file as image
song_file = 'test.jpg'
dirty_orig = uio.import_song(song_file)
clean_orig = dirty_orig.copy()
blobs = pd.read_csv('blobs.csv')

# uio.show_image(orig, reduce=2)
_, line_sep = umm.isolate_music_lines(dirty_orig)

line_height = int(12*line_sep)
n_lines = round(dirty_orig.shape[0]/line_height)

for j in range(n_lines):
    top = j*line_height
    bottom = (j+1)*line_height
    dirty_line = dirty_orig[top:bottom, :]

    # Populate mask with largest external contour (staff)
    mask = np.zeros_like(dirty_line)
    contours, _ = cv.findContours(255-dirty_line, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)
    cv.drawContours(mask, [c], -1, 255, -1)

    # Add white patches to cover notes
    is_correct_line = blobs.cy.between(top, bottom)
    is_not_none = blobs.state!='none'
    line_blobs = blobs[is_correct_line & is_not_none]
    for index, row in line_blobs.iterrows():
        if cv.pointPolygonTest(c, (int(row.cx), int(row.cy%line_height)), False)==-1:
            x, y = int(row.x), int(row.y%line_height)
            w, h = int(row.width*line_sep), int(row.height*line_sep)
            cv.rectangle(mask, (x,y), (x+w,y+h), 255, -1)

    clean_line = ~cv.bitwise_and(~dirty_line, mask)
    clean_orig[top:bottom] = clean_line

uio.show_image(clean_orig, reduce=1)






# color = cv.cvtColor(line, cv.COLOR_GRAY2BGR)