# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:14:46 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from Utils import io as uio


def get_pitch_as_string(pitch, keysig_acc='f'):
    if isinstance(pitch, pd.Series):
        pitch = pitch.values
        
    if keysig_acc=='sharp':
        note_strings = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    else:
        note_strings = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
        
    note_string = note_strings[int(pitch)%12]
    return note_string
    

def get_pitch(y):
    if isinstance(y, pd.Series):
        y = y.values
        
    semitone_mapping = pd.Series(
        index=range(7),
        data=[0,2,4,5,7,9,11]
    )
    staff_pitch = get_staff_pitch(y)
    pitch = semitone_mapping[staff_pitch%7].values + 12*(staff_pitch//7)
    return pitch
    
def get_staff_pitch(y):
    if isinstance(y, pd.Series):
        y = y.values
    line_sep, line_height, margin = uio.get_song_params(['line_sep','line_height', 'margin'])
    is_treble = (y//line_height)%2==0
    mody = y%line_height
    
    # Apply bass linear fit
    slope = -2/line_sep
    intercept = -2 + (2*margin)
    staff_pitch =np.round(slope*mody + intercept).astype(int)
    
    # Shift up 12 if treble
    staff_pitch[is_treble] = staff_pitch[is_treble] + 12
    return staff_pitch


def get_chord_string(pitches, keysig_acc='f', include_root=False):
    chord_patterns = { 
        '':{0,4,7},
        'm':{0,3,7},
        'dim':{0,3,6},
        'aug':{0,4,8},
        '7':{0,4,7,10},
        'maj7':{0,4,7,11},
        'm7':{0,3,7,11},
        # 'sus2':{0,2,4,7},
        # 'sus4':{0,4,5,7}
    }
    
    # Ensure sorting to identify root
    pitches = np.sort(pitches)
    
    c = np.unique(np.sort(pitches%12))
    if len(c)<2:
        return '-'
    
    for name, pattern in chord_patterns.items():
        for j in range(len(c)):
            inversion = (np.roll(c,-j)-c[j])%12
            if set(inversion).issubset(pattern):
                key = get_pitch_as_string(c[j], keysig_acc)
                root = get_pitch_as_string(pitches[0], keysig_acc)
                
                chord_string = key + name
                if root!=key and include_root:
                    chord_string += '/' + root
                return chord_string
    return '?'
    
def annotate_chords(img, notes, keysig_acc):
    line_sep, line_height = uio.get_song_params(['line_sep', 'line_height'])
    
    # Define text properties
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color=0
    thickness=5

    for beat in notes.beat.unique():
        current_chord_index = notes.beat==beat
        chord_index = np.logical_and(
            notes.beat<=beat,
            notes.beat+notes.duration>beat
        )
        
        pitches = notes.loc[chord_index, 'pitch'].values
        cx = notes.loc[current_chord_index, 'cx'].max()
        cy = notes.loc[current_chord_index, 'cy'].max()
        chord_string = get_chord_string(pitches, keysig_acc, include_root=False)

        text_width, text_height = cv.getTextSize(chord_string, font, fontScale, thickness)[0]
        x, y = int(cx-text_width/2), int((cy//(2*line_height)+1)*2*line_height-10)
        cv.putText(img, chord_string, (x,y), font, fontScale, color, thickness, cv.LINE_AA)
        
    return img
                    
