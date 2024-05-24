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

from Utils import model as um
from Utils import io as uio
from Utils import music_munging as umm
from Utils import image_processing as uip
from Utils import music_theory as umt

import cv2 as cv
from scipy import ndimage
import os
import pickle

# How to train better, more variety (make database?)
# General clean up/organization
    # Can we have entire data dataframes be integer type
# Once I build up large enough datasets, I can optimize the machine learning models
# Handle multiple page pdfs
# Noisy pdfs?


# %% Import and clean music
song_name = 'joy_to_the_world'
song_file = f'Sheet Music\{song_name}.pdf'

raw_music = uio.import_song(song_file)
cv.imwrite('Processed Images\\1 raw music.jpg',raw_music)

orig = umm.clean_music(raw_music.copy())
cv.imwrite('Processed Images\\2 strip words.jpg', orig)


# %% Fill notes (and flats??)
model = pickle.load(open('Saved Models/notefilling.pkl', 'rb'))


# %%
# Fill notes and flats

filled_img, _ = um.run_model(orig.copy(), model_type='filling')
cv.imwrite('Processed Images\\3 fill notes.jpg', filled_img)


#%%
# Remove staff lines
nostaff_img = umm.remove_staff_lines(filled_img.copy())
cv.imwrite('Processed Images\\4 remove staff.jpg', nostaff_img)

# Identify notations, separate into structures
if validate:
    um.NotationValidation(nostaff_img.copy())
_, notations = um.run_model(nostaff_img.copy(), 'notations')
measures, rests, accs, dots = umm.separate_notations(notations, orig.copy())

# Remove notations
no_notations_img = umm.remove_nonnotes(nostaff_img, notations)    
cv.imwrite('Processed Images\\5 remove notations.jpg', no_notations_img)

# Close to remove lines all lines
line_sep = uio.get_song_params(['line_sep'])
closed_img = uip.morphology_operation(
    no_notations_img.copy(), (.25*line_sep, .25*line_sep), cv.MORPH_CLOSE
)
cv.imwrite('Processed Images\\6 no lines.jpg', closed_img)

# Identify notes and separate with kmeans clustering
if validate:
    um.NoteValidation(closed_img)
_, grouped_notes = um.run_model(closed_img.copy(), 'notes')
notes = umm.separate_grouped_notes(closed_img.copy(), grouped_notes)

# %%
# Perform computations on notes
notes = umm.compute_stems_and_tails(no_notations_img.copy(), notes)
notes = umm.compute_is_filled(orig.copy(), filled_img, notes)
notes = umm.apply_accidentals(notes, accs)
notes, keysig = umm.apply_keysignature(notes, orig.copy(), accs, measures)
notes = umm.apply_dots(notes, dots)
notes['duration'] = 2.0**(1+notes.is_filled - notes.is_stemmed - notes.tails) * (1+0.5*notes.is_dotted)

# %%
# Compute beats and write to wav file
notes = umm.compute_beats(notes, rests)
uio.write_to_WAV(notes)

# notes = notes.sort_values(by=['beat','pitch']).reset_index(drop=True)

# %%

def check_note_booleans(img, notes, col):
    color_img = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
    mcolors = [(0,0,255),(0,255,0),(255,0,0)]
    for row in notes.itertuples():
        x,y,w,h = row.x, row.y, row.w, row.h
        col_val = int(getattr(row, col))
        if col.startswith('is_'):
            cv.rectangle(color_img, (x,y), (x+w,y+h), mcolors[col_val], 5)
        elif col=='duration':
            if row.duration==0.25 or row.duration==2:
                cv.rectangle(color_img, (x,y), (x+w,y+h), (0,0,255), 5)
            if row.duration==0.5 or row.duration==4:
                cv.rectangle(color_img, (x,y), (x+w,y+h), (0,255,0), 5)
            if row.duration==1:
                cv.rectangle(color_img, (x,y), (x+w,y+h), (255,0,0), 5)
            
        elif col=='tails':
            cv.rectangle(color_img, (x,y), (x+w,y+h), mcolors[col_val],5)
        elif col=='chord':
            cv.rectangle(color_img, (x,y), (x+w,y+h), mcolors[col_val%3], 5)

        # cv.rectangle(color_img, (x,y), (x+w,y+h), mcolors[int(row.is_filled)], 5)
        # cv.circle(color_img, (x+w//2,y+h//2), 10, mcolors[row.tails], thickness=-1)
        
    uio.show_image(color_img, reduce=2)
    return
check_note_booleans(orig, notes, 'tails')
chords_img = umt.annotate_chords(orig.copy(), notes.copy(), keysig['acc'])
cv.imwrite('Processed Images//7 add chords.jpg', chords_img)
# %%
