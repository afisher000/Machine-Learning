# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import utils_model as um
import utils_io as uio
import utils_music_cleaning as umc
import utils_music_theory as umt
from sklearn.svm import SVC

# # To implement 
# Identify more objects
# Try linear model?

# Open song file as image
song_file = 'Songs\\crown_him_with_many_crowns.pdf'
orig = uio.import_song(song_file)


# Separate image into lines and clear words, and recombine
imgs, line_sep = umc.split_music_into_lines(orig)
cleaned_imgs = [umc.remove_words_from_line(img, line_sep) for img in imgs]
cleaned_orig = np.vstack(cleaned_imgs)
line_height = cleaned_orig.shape[0]/len(imgs)

# Fill notes with model
filled_img = um.fill_contours_using_model(cleaned_orig.copy(), line_sep)
cv.imwrite('Processed Images\\test_filled.jpg',filled_img)

# Close image to remove lines
close_pixels = int(0.25*line_sep)
kernel = np.ones((close_pixels,close_pixels), dtype=np.uint8)
closed_img = cv.morphologyEx(filled_img.copy(), cv.MORPH_CLOSE, kernel)
cv.imwrite('Processed Images\\test_closed.jpg', closed_img)


# Identify notes with model
blobs = um.identify_blobs_using_model(closed_img.copy(), line_sep)

labeled_img = um.label_blobs_on_music(cleaned_orig, blobs, line_sep)
uio.show_image(labeled_img)

# # Add measures
# blobs = umc.append_measure_lines(cleaned_orig, blobs, line_sep)

# ## Munge blobs to get ready for note computations
# song_input = umc.munge_blob_data(closed_img, blobs, line_sep, line_height)

# ## Parse song accidentals and key signature
# notes, key_type = umc.parse_song_notes(song_input, line_sep)
# umt.identify_chords(notes.copy(), key_type)
# notes.to_csv('notes.csv', index=False)
# uio.write_to_WAV(notes.copy())




# %%
