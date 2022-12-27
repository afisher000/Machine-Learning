# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:26:45 2022

@author: afisher
"""

import utils_io as uio
import utils_music_cleaning as umc
import utils_model as um

import numpy as np
import pandas as pd
import cv2 as cv
import pickle
import sys


validate_filling_model = True
validate_identifying_model = True

# Open song file as image
song_file = 'Songs\\scales.pdf'
orig = uio.import_song(song_file)


# Separate image into lines and clear words, and recombine
imgs, line_sep = umc.split_music_into_lines(orig)
cleaned_imgs = [umc.remove_words_from_line(img, line_sep) for img in imgs]
cleaned_orig = np.vstack(cleaned_imgs)

# Fill notes with model and optionally validate
if validate_filling_model:
    filled_value = 100
    filled_img = um.fill_contours_using_model(cleaned_orig.copy(), line_sep, fill_value=filled_value)
    um.Filling_Model_Validation(filled_img, line_sep)
    sys.exit()
else:
    filled_img = um.fill_contours_using_model(cleaned_orig.copy(), line_sep, fill_value=0)



# Close image to remove lines
close_pixels = int(0.25*line_sep)
kernel = np.ones((close_pixels,close_pixels), dtype=np.uint8)
closed_img = cv.morphologyEx(filled_img.copy(), cv.MORPH_CLOSE, kernel)

# Validate
um.Identifying_Model_Validation(cleaned_orig, closed_img, line_sep)







