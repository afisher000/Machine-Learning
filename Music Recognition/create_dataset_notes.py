# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:05:57 2024

@author: afisher
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 12:13:55 2024

@author: afisher
"""

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

# %% Import and clean music
song_name = 'single_line'
extension = 'jpg'
song_file = f'Sheet Music\\Original\\{song_name}.{extension}'

raw_music = uio.import_song(song_file) #Create song params
orig = uio.import_song(f'Sheet Music\\Notes\\{song_name}.jpg', save_params=False)


# %% Filter and sort contours by area
contours, _ = cv.findContours(~orig, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key = lambda x: cv.boundingRect(x)[3])


# %% Manually identify contours to fill
color_image = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
labels = ['0']*len(sorted_contours)
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
    
    H = min(h, 20)
    W = min(w, 20)
    cv.imshow('image', color_image[max(0,y-2*H):min(y+h+2*H, color_image.shape[0]),
                                   max(0,x-2*W):min(x+w+2*W, color_image.shape[1])])
    
    # Parse key input
    key = cv.waitKeyEx(0)
    if key == 2555904:
        j += 1
    elif key==2424832:
        j -= 1
    elif key in map(ord, list('01234')):
        labels[j] = chr(key)
        print('Fill')
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
file_path = os.path.join('Datasets/Notes', datestr)
df.to_csv(file_path, index=False)