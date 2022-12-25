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
import utils_music_cleaning as umc
from sklearn.svm import SVC

# To implement 
# Identify more objects

# Convert to jpg if necesary
song_file = 'test_whole_notes.pdf'
if song_file.endswith('.pdf'):
    pdf2jpg(song_file)
song_file = song_file[:-4]+'.jpg'

# Separate image into lines and clear words
_, orig = cv.threshold(cv.imread(song_file, 0), 127, 255, 0)
imgs, line_sep = umc.split_music_into_lines(orig)
img_heights = [img.shape[0] for img in imgs]
cleaned_imgs = [umc.remove_words_from_line(img, line_sep) for img in imgs]
cleaned_orig = np.vstack(cleaned_imgs)

# Fill notes with model
filled_img = umc.fill_notes_from_model(cleaned_orig.copy(), line_sep)
cv.imwrite('test_filled.jpg',filled_img)

# Close image to remove lines
close_pixels = int(0.25*line_sep)
kernel = np.ones((close_pixels,close_pixels), dtype=np.uint8)
closed_img = cv.morphologyEx(filled_img.copy(), cv.MORPH_CLOSE, kernel)
cv.imwrite('test_closed.jpg', closed_img)


# Identify notes with model
blobs = umc.identify_blobs_from_model(closed_img.copy(), line_sep)

# Add meaures
blobs = umc.append_measure_lines(cleaned_orig, blobs, line_sep)
umc.print_marked_music(cleaned_orig, blobs, line_sep)

## Apply munnging to blobs to get input for song parsing
song_input = umc.munging_before_parsing_song(blobs, line_sep, img_heights)

## Parse song accidentals and key signature
notes = umc.parse_song_notes(song_input, line_sep)




# %%
