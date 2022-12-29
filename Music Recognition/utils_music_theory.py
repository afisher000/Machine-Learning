# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:14:46 2022

@author: afisher
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numbers
import cv2 as cv
import utils_io as uio

def get_pitch_as_string(pitch, keysig='sharp'):
    # Check if numeric
    if isinstance(keysig, numbers.Number):
        keysig = 'sharp' if keysig>=0 else 'flat'
        
    if keysig=='sharp':
        note_strings = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    else:
        note_strings = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
        
    note_string = note_strings[int(pitch)%12]
    return note_string
    

def get_staff_pitch(y, line_sep, line_height, is_bass):
    # Independent variable is line pixel and dependent variable is staff_pitch
    slope = -2/line_sep
    intercept = 6 if is_bass else 18
    mody = y%line_height
    staff_pitch = round(slope*mody + intercept)
    return staff_pitch


def get_pitch_from_staff_pitch(staff_pitch):
    semitone_map = {
        0:0, 1:2, 2:4, 3:5, 4:7, 5:9, 6:11
    }
    pitch = semitone_map[staff_pitch%7] + 12*(staff_pitch//7)
    return pitch

def get_chord_string(pitches):
    chord_patterns = { 
        '':{0,4,7},
        'm':{0,3,7},
        'dim':{0,3,6},
        'aug':{0,4,8},
        '7':{0,4,7,10},
        'maj7':{0,4,7,11},
        'm7':{0,3,7,11},
        'sus2':{0,2,4,7},
        'sus4':{0,4,5,7}
    }
    
    # Ensure sorting to identify root
    pitches = np.sort(pitches)
    
    c = np.unique(np.sort(pitches%12))
    if len(c)<3:
        return '-'
    
    for name, pattern in chord_patterns.items():
        for j in range(len(c)):
            inversion = (np.roll(c,-j)-c[j])%12
            if set(inversion).issubset(pattern):
                key = get_pitch_as_string(c[j])
                root = get_pitch_as_string(pitches[0])
                
                chord_string = key + name
                if root!=key:
                    chord_string += '/' + root
                return chord_string
    return '?'
    
def annotate_chords(img, notes, line_height):
    # Get keysig
    keysig = 'sharp' if notes.string.str.contains('#').sum()>0 else 'flat'
    
    # Define text properties
    font = cv.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color=0
    thickness=5
    
    
    # Group notes into chords using 'line' and 'x'
    notes = notes.set_index(['x', 'line'])
    for index in notes.index.unique():
        pitches = notes.loc[index, 'pitch'].values
        chord_string = get_chord_string(pitches)
        
        
        text_width, text_height = cv.getTextSize(chord_string, font, fontScale, thickness)[0]
        x, y = int(index[0]-text_width/2), int((index[1]+1)*2*line_height-10)
        cv.putText(img, chord_string, (x,y), font, fontScale, color, thickness, cv.LINE_AA)
        
    return img
                    
