# %%
# # -*- coding: utf-8 -*-
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
import utils_image_processing as uip
import cv2 as cv
from scipy import ndimage
import os
import pickle


song_file = 'Songs\\sharps.jpg'
raw_music = uio.import_song(song_file)
# # Remove words and equalize line spacing
cleaned_img = umm.strip_words(raw_music.copy(), margin=1.5)
orig, line_sep =  umm.equalize_music_lines(cleaned_img.copy(), margin=5)

line_height = 12*line_sep
# cv.imwrite('Test\\nostaff.jpg', orig)

filled_img = um.fill_contours_using_model(orig.copy(), line_sep)
nostaff_img = umm.remove_staff_lines(filled_img)
cv.imwrite('Test\\nostaff.jpg', nostaff_img)

# Identify nonnotes
blobs = um.identify_nonnotes_using_model(nostaff_img.copy(), line_sep)
# labeled_img = um.label_nonnotes(orig, blobs, line_sep)
# uio.show_image(labeled_img, reduce=3)
class NonNote_Validation():
    def __init__(self, orig, img, blobs, line_sep):
        self.orig = orig
        self.img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        self.blobs = blobs
        self.line_sep = line_sep
        self.database = 'training_nonnote_blobs.csv'
        self.model_file = 'model_to_identify_nonnotes.pkl'
        
        # Label image
        self.labeled_img = um.label_nonnotes(orig, self.blobs, line_sep)
        
        # Main loop
        cv.namedWindow='zoomed_out'
        cv.imshow('zoomed_out', self.zoomed_out_image())
        cv.setMouseCallback('zoomed_out', self.roi_callback)
        
        while True:
            cv.imshow('zoomed_out', self.zoomed_out_image())
            k = cv.waitKey(20)
            if k==ord('s'):
                self.save_to_database()
                break
            elif k==ord('q'):
                # Quit without saving
                break
        cv.destroyWindow('zoomed_out')
    
    def zoomed_out_image(self):
        return cv.pyrDown(cv.pyrDown(self.labeled_img))


    def roi_callback(self, event, x, y, flags, param):
        # Fill or unfill contour
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.namedWindow='zoomed_in'
            
            # Find blob with closest centroid
            idx = np.argmin((self.blobs.cx-4*x)**2+(self.blobs.cy-4*y)**2)
            x, y, cx, cy, height, width = self.blobs.loc[idx, ['x','y','cx','cy','height','width']]
            pad = 1
            top = int(y)-pad
            bottom = int(y + height*self.line_sep)+pad
            left = int(x)-pad
            right = int(x + width*self.line_sep)+pad
            
            # Show closeup, respond with keypress
            cv.imshow('zoomed_in', self.orig[top:bottom, left:right])
            key = cv.waitKey(0)
            cv.destroyWindow('zoomed_in')
            
            # Update blob and color label
            if chr(key) in 'sfn23468dmqwert\r':
                self.blobs.loc[idx, 'state'] = chr(key)
                cv.circle(self.labeled_img, (int(cx), int(cy)), 15, um.nonnote_colors[chr(key)], -1)
        return
    
    def save_to_database(self):
        # Update training dataq
        if os.path.exists(self.database):
            blob_data = pd.read_csv(self.database)
        else:
            blob_data = pd.DataFrame(columns=um.data_columns)
         
        blob_data = pd.concat([blob_data, self.blobs])
        blob_data.to_csv(self.database)
        
        # Train model
        ytrain = blob_data.state
        Xtrain = blob_data[um.training_columns].values
        model = RandomForestClassifier()
        model.fit(Xtrain, ytrain)
        pickle.dump(model, open(self.model_file, 'wb'))


test = NonNote_Validation(orig, nostaff_img, blobs, line_sep)
# %%

# %%
