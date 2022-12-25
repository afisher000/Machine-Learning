import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from contour_utils import get_contour_data

def split_music_into_lines(img):
    # Remove all-white columns
    non_white_cols = img.sum(axis=0)<255*img.shape[0]
    img = img[:, non_white_cols]

    # Parse staff lines
    is_line = (img.sum(axis=1)/img.shape[1]) < 127
    is_line_start = np.logical_and(~is_line[:-1], is_line[1:])
    line_starts = np.where(is_line_start)[0]
    line_sep = np.diff(line_starts[:4]).mean()
    print(f'Line Separation = {line_sep:.1f}')
    num_lines = len(line_starts)//5

    imgs = []
    # Return list of imgs
    for j in range(num_lines):
        top = int(line_starts[5*j] - 4*line_sep)
        bottom = int(line_starts[4+5*j] + 4*line_sep)
        left = int(3.5*line_sep)
        right = int(-1.0*line_sep)
        imgs.append(img[top:bottom,left:right].copy())
    return imgs, line_sep

def remove_words_from_line(img, line_sep):
    # Get bounding box or largest contour
    mask = 255-img
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(c)

    # Get masks of vertical lines
    short_lines = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=np.ones((int(4*line_sep),1), dtype=np.uint8))
    stem_rows = np.where(short_lines.sum(axis=1)/short_lines.shape[1] < 255)[0]
    
    # Find largest of union
    top = min(stem_rows.min(), y)
    bottom = max(stem_rows.max(), y+h)
    img[:top] = 255
    img[bottom:] = 255
    return img

def fill_notes_from_model(img, line_sep):
    svc = pickle.load(open('model_to_fill_notes.pkl','rb'))
    contour_data = get_contour_data(img, line_sep)
    X = contour_data.drop(columns=['state', 'cx','cy']).values
    contour_data.state = svc.predict(X)

    for index, row in contour_data.iterrows():
        if row.state==1:
            cv.floodFill(img, None, (int(row.cx), int(row.cy)), 0)
    return img

def identify_blobs_from_model(img, line_sep):
    svc = pickle.load(open('model_to_identify_notes.pkl','rb'))
    blobs = get_contour_data(img, line_sep)
    X = blobs.drop(columns=['state','cx','cy']).values
    blobs.state = svc.predict(X)
    return blobs


def munging_before_parsing_song(blobs, line_sep, img_heights):
    # Drop NANs
    blobs = blobs[blobs.state!='none']

    # Split doubled notes
    doubles = blobs[blobs.state=='2'].copy()
    blobs = blobs[blobs.state!='2']
    top_notes = doubles.copy()
    top_notes.cy = top_notes.cy + 0.5*line_sep
    bottom_notes = doubles.copy()
    bottom_notes.cy = bottom_notes.cy - 0.5*line_sep
    blobs = pd.concat([blobs, top_notes, bottom_notes])

    # Compute line
    blobs['raw_line'] = blobs.cy.apply(lambda x: np.where(x-np.cumsum(img_heights)<0)[0][0])
    blobs['is_bass'] = blobs.raw_line.mod(2).astype('bool')
    blobs['line'] = blobs.raw_line//2

    # Compute "staff pitch" (0 is center c, increment every 0.5*line_sep)
    blobs['staff_pitch'] = (-2/line_sep * blobs.cy.mod(img_heights[0]) + 18 - 12*blobs.is_bass).round().astype(int)

    # Compute actual pitch (have to use semitone map)
    semitone_map = {0:0, 1:2, 2:4, 3:5, 4:7, 5:9, 6:11}
    blobs['pitch'] = blobs.staff_pitch.mod(7).map(semitone_map) + 12*blobs.staff_pitch.floordiv(7)

    # Compute measure (separate into treble and bass to compute, then recombine)
    blobs = blobs.sort_values(by=['line','cx'])
    bass_blobs = blobs[blobs.is_bass].copy()
    treble_blobs = blobs[~blobs.is_bass].copy()
    bass_blobs['measure'] = np.cumsum(bass_blobs.state=='m') + bass_blobs.line
    treble_blobs['measure'] = np.cumsum(treble_blobs.state=='m') + treble_blobs.line
    blobs = pd.concat([bass_blobs, treble_blobs])
    blobs.loc[blobs.state=='m', 'measure'] -= 0.5

    # Sort 
    blobs = blobs.sort_values(by=['line','measure','staff_pitch','cx']).reset_index(drop=True)
    return blobs

def parse_song_notes(song_input, line_sep):
    # Initialize key signals and dictionary maps
    measure_start_x = 0
    keysignature = {}
    accidentals = {}
    accidental_map = {'s':1, 'f':-1, 'n':0}
    pitch_to_string = { 
        0:'C',
        1:'C#/Db',
        2:'D',
        3:'D#/Eb',
        4:'E',
        5:'F',
        6:'F#/Gb',
        7:'G',
        8:'G#/Ab',
        9:'A',
        10:'A#/Bb',
        11:'B'
    }

    # Apply key signal and accidentals to notes
    notes = pd.DataFrame(columns=['line', 'measure','pitch','x'])
    for index, row in song_input.iterrows():
        
        if row.state in ['s','f','n']:
            has_target = (song_input.loc[index+1, 'cx']-row.cx<2*line_sep)
            if has_target:
                # If accidental has target, add to temporary accidentals dict
                accidentals[row.staff_pitch] = accidental_map[row.state]
            else:
                # If no target, add to permanent keysignature. Define as measure start
                keysignature[row.staff_pitch%7] = accidental_map[row.state]
                measure_start_x = row.cx

        if row.state in ['1','2']:
            # Check if note pitch should be influenced by accidentals or keysignature
            # If accidental applies, do not apply keysignature
            pitch = row.pitch
            if row.staff_pitch in accidentals.keys():
                pitch = row.pitch + accidentals[row.staff_pitch]
            elif row.staff_pitch%7 in keysignature.keys():
                pitch = row.pitch + keysignature[row.staff_pitch%7]
            notes.loc[len(notes)] = [row.line, row.measure, pitch, row.cx-measure_start_x]
        if row.state == 'm':
            # Reset accidentals on new measure. Reset measure_start_x
            accidentals = {}
            measure_start_x = row.cx

    # Create string labels
    notes['string'] = notes.pitch.mod(12).map(pitch_to_string)

    # Round horizontal position to nearest 10s of pixels
    notes.x = notes.x.round(-1)
    
    # Sort output for grouping chords
    notes = notes.sort_values(by=['line','measure','x','pitch'])
    return notes

def append_measure_lines(img, blobs, line_sep):
    # Identify measure lines
    kernel = np.ones((int(3.5*line_sep), 1), dtype=np.uint8)
    vert_lines = ~cv.morphologyEx(img.copy(), cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(vert_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        M = cv.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        is_far_in_x = np.abs(cx-blobs.cx)>1*line_sep
        is_far_in_y = np.abs(cy-blobs.cy)>3*line_sep
        # Measure lines must be either far away in x, or far away in y for all blobs
        if np.all(np.logical_or(is_far_in_x, is_far_in_y)):
            blobs.loc[len(blobs), ['state','cx','cy']] = ['m', cx, cy] 
    return blobs

def print_marked_music(img, blobs, line_sep):
    # Add colored tags to image
    color_map = { 
        '1':(0,0,255),
        '2':(0,0,255),
        's':(255,0,0),
        'f':(255,0,0),
        'n':(255,0,0),
        'm':(0,255,255),
        'none':(0,255,0)
    }

    color_orig = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for key, color in color_map.items():
        for index, (cx, cy) in blobs.loc[blobs.state==key, ['cx','cy']].iterrows():
            if key=='2':
                cv.circle(color_orig, (int(cx), int(cy+0.5*line_sep)), 15, color, -1)
                cv.circle(color_orig, (int(cx), int(cy-0.5*line_sep)), 15, color, -1)
            else:
                cv.circle(color_orig, (int(cx), int(cy)), 15, color, -1)
    cv.imwrite('test_identified_music.jpg',color_orig)