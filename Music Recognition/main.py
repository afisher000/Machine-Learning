# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import utils_model as um
import utils_io as uio
import utils_music_munging as umm
import utils_music_theory as umt
from sklearn.svm import SVC

# # To implement 
# add function that strips black away from staff when notes not present

validate_models = False

# Open song file as image
song_file = 'Songs\\crown_him_with_many_crowns.jpg'
raw_music = uio.import_song(song_file)

# Separate cut out non-staffrecombine
orig, line_sep = umm.isolate_music_lines(raw_music)
line_height = 12*line_sep

# Fill notes with model
if validate_models:
    # Validate note filling
    filled_value = 100
    filled_img = um.fill_contours_using_model(orig.copy(), line_sep, fill_value=100)
    um.Filling_Model_Validation(filled_img, line_sep, filled_value=filled_value)
    
    # Turn filled to black
    filled_img[filled_img==filled_value]=0
else:
    filled_img = um.fill_contours_using_model(orig.copy(), line_sep)

# Close image to remove lines
close_pixels = int(0.25*line_sep)
kernel = np.ones((close_pixels,close_pixels), dtype=np.uint8)
closed_img = cv.morphologyEx(filled_img.copy(), cv.MORPH_CLOSE, kernel)


# Identify notes with model
blobs = um.identify_blobs_using_model(orig, closed_img.copy(), line_sep)
if validate_models:
    um.Identifying_Model_Validation(orig, closed_img, blobs, line_sep)

    
## Munge blobs to get ready for note computations
song_input = umm.munge_blob_data(closed_img, blobs, line_sep, line_height)

## Parse song accidentals and key signature
notes = umm.parse_song_notes(song_input, line_sep)

# Clean up lines and annotate chords
cleaned_music = umm.clean_music(orig, blobs, line_sep)
annotated_orig = umt.annotate_chords(cleaned_music.copy(), notes.copy(), line_height)
uio.show_image(annotated_orig, reduce=2)



# Write to audio
# uio.write_to_WAV(notes.copy())




# %%