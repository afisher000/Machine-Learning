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
from utils_music_cleaning import split_music_into_lines, remove_words_from_line, fill_notes_from_model
import os
from sklearn.svm import SVC

# Read file, separate into lines
#


song_file = 'Songs\\as_water_to_the_thirsty.jpg'
if song_file.endswith('.pdf'):
    pdf2jpg(song_file)
song_file = song_file[:-4]+'.jpg'



_, orig = cv.threshold(cv.imread(song_file, 0), 127, 255, 0)
imgs, line_sep = split_music_into_lines(orig)
cleaned_imgs = [remove_words_from_line(img, line_sep) for img in imgs]
cleaned_orig = np.vstack(cleaned_imgs)

# Fill notes with model
filled_img = fill_notes_from_model(cleaned_orig.copy(), line_sep)

# Close image to remove lines
close_pixels = int(0.25*line_sep)
kernel = np.ones((close_pixels,close_pixels), dtype=np.uint8)
closed_img = cv.morphologyEx(filled_img.copy(), cv.MORPH_CLOSE, kernel)


# Manually label contours
contours, _ = cv.findContours(255-closed_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv.contourArea) # sort by area
pad = 1
states = []
valid_contours = []
for c in contours:
    if len(c)<=4:
        continue
    
    x,y,w,h = cv.boundingRect(c)

    cv.imshow('test', cleaned_orig[int(y-pad):int(y+h+pad), int(x-pad):int(x+w+pad)])
    k = cv.waitKey(0)
    cv.destroyAllWindows()
    if k==27:
        break
    elif k in list(map(ord, ['1','2','3','4','f','s','n'])):
        states.append(chr(k))
        valid_contours.append(c)
    elif k==13:
        states.append('none')
        valid_contours.append(c)

# Combine with old data
new_data = get_contour_data(None, line_sep, specified_contours=valid_contours)
new_data.state = states
if os.path.exists('training_data_blobs.csv'):
    old_data = pd.read_csv('training_data_blobs.csv')
    data = pd.concat([old_data, new_data])
else:
    data = new_data
data.to_csv('training_data_blobs.csv', index=False)

# Train model and save
svc = SVC(C=100)
ytrain = data.state
Xtrain = data.drop(columns=['state','cx','cy', 'x', 'y']).values

svc.fit(Xtrain, ytrain)
pickle.dump(svc, open('model_to_identify_notes.pkl', 'wb'))
# %%
