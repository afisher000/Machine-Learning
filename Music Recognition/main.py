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
from write_to_WAV import write_to_WAV
from identify_chords import identify_chords
# # To implement 
# Identify more objects
# Add slight noise to training data so more robust
# Try linear model?


# Convert to jpg if necesary
song_file = 'Songs\\crown_him_with_many_crowns.pdf'
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
cv.imwrite('Processed Images\\test_filled.jpg',filled_img)

# Close image to remove lines
close_pixels = int(0.25*line_sep)
kernel = np.ones((close_pixels,close_pixels), dtype=np.uint8)
closed_img = cv.morphologyEx(filled_img.copy(), cv.MORPH_CLOSE, kernel)
cv.imwrite('Processed Images\\test_closed.jpg', closed_img)


# Identify notes with model
blobs = umc.identify_blobs_from_model(closed_img.copy(), line_sep)

# Add meaures
blobs = umc.append_measure_lines(cleaned_orig, blobs, line_sep)

## Munge blobs to get ready for note computations
song_input = umc.munge_blob_data(blobs, line_sep, img_heights, closed_img)
umc.print_marked_music(cleaned_orig, song_input, line_sep)

## Parse song accidentals and key signature
notes, key_type = umc.parse_song_notes(song_input, line_sep)
identify_chords(notes.copy(), key_type)
notes.to_csv('notes.csv', index=False)
write_to_WAV(notes.copy())




# %%
