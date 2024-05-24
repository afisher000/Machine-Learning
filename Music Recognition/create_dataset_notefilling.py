# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:42:30 2024

@author: afisher
"""


from Utils import model as um
from Utils import io as uio
from Utils import music_munging as umm
from Utils import image_processing as uip
from Utils import music_theory as umt

import cv2 as cv
import numpy as np
import os
from datetime import datetime

# %% Read and clean music
song_name = 'fairest_lord_jesus'
song_file = os.path.join('Sheet Music', song_name + '.pdf')

raw_music = uio.import_song(song_file)

params = uio.save_song_params(raw_music)
orig = umm.clean_music(raw_music.copy())


# %% Filter and sort contours by area
contours, _ = cv.findContours(orig, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

max_contour_area = .8 * params['line_sep']**2
min_contour_area = 0.05 * params['line_sep']**2

fillable_contours = [c for c in contours if cv.contourArea(c)<max_contour_area 
                     and cv.contourArea(c)>min_contour_area] 
sorted_contours = sorted(fillable_contours, key = lambda x: cv.contourArea(x))


# %% Manually identify contours to fill
color_image = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
labels = ['n']*len(sorted_contours)
j = 0
keep_looping = True
while keep_looping:
    # Get new contour
    j = max(j, 0)
    j = min(j, len(sorted_contours)-1)
    print(f'{j+1} of {len(sorted_contours)}')
    c = sorted_contours[j]
    x, y, w, h = cv.boundingRect(c)

    # Fill with green and show
    cv.drawContours(color_image, [c], -1, (0, 255, 0), -1)
    
    cv.imshow('image', color_image[max(0,y-2*h):min(y+3*h, color_image.shape[0]),
                                   max(0,x-2*w):min(x+3*w, color_image.shape[1])])
    
    # Parse key input
    key = cv.waitKeyEx(0)
    if key == 2555904:
        j += 1
    elif key==2424832:
        j -= 1
    elif key==ord('y'):
        labels[j] = 'y'
        print('Fill')
        j += 1        
    elif key == ord('n'):
        labels[j] = 'n'
        print('No fill')
        j += 1
    elif key == ord('q'):
        keep_looping = False
    
    # Remove colors and window
    cv.drawContours(color_image, [c], -1, (255, 255, 255), -1)
    cv.destroyAllWindows()


# %% Save labels and contour dimensions to csv
df = um.get_contour_data( sorted_contours )
df.index = labels
df = df.reset_index().rename(columns={'index':'labels'})

# Save for given minute
datestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f' {song_name}.csv'
file_path = os.path.join('Datasets/Note Filling', datestr)
df.to_csv(file_path, index=False)