import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from utils_contours import get_contour_data

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
    X = contour_data.drop(columns=['state','cx','cy', 'x', 'y']).values
    contour_data.state = svc.predict(X)

    for index, row in contour_data.iterrows():
        if row.state==1:
            cv.floodFill(img, None, (int(row.cx), int(row.cy)), 0)
    return img

def identify_blobs_from_model(img, line_sep):
    svc = pickle.load(open('model_to_identify_notes.pkl','rb'))
    blobs = get_contour_data(img, line_sep)
    X = blobs.drop(columns=['state','cx','cy', 'x', 'y']).values
    blobs.state = svc.predict(X)
    return blobs



def munge_blob_data(blobs, line_sep, img_heights, closed_img):
    blobs = blobs[blobs.state!='none']
    blobs['raw_line'] = blobs.cy.apply(lambda x: np.where(x-np.cumsum(img_heights)<0)[0][0])
    blobs['is_bass'] = blobs.raw_line.mod(2).astype(bool)
    blobs['total_cx'] = blobs.raw_line*blobs.cx.max() + blobs.cx
    blobs = blobs.sort_values(by=['raw_line','cx'])
    
    
    def get_kmean_cluster_centers(c, closed_img):
        x,y,w,h = int(c.x), int(c.y), int(c.width*line_sep), int(c.height*line_sep)
        subimg = closed_img[y:y+h, x:x+w]
    
        # Apply kmeans clustering
        ypoints, xpoints = np.where(subimg==0)
        points = np.float32(np.vstack([ypoints, xpoints]).T)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv.kmeans(points, int(c.state), None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        return centers + [y, x]
    

    semitone_map = {0:0, 1:2, 2:4, 3:5, 4:7, 5:9, 6:11}
    song_input = pd.DataFrame(columns=['state','cx','cy','line','measure','staff_pitch','pitch'])
    for index, row in blobs.iterrows():
        line = row.raw_line//2
        staff_pitch= round(-2/line_sep * (row.cy%img_heights[0]) + 18 - 12*row.is_bass)
        pitch = semitone_map[staff_pitch%7] + 12*(staff_pitch//7)
        measure = sum((blobs.state=='m')&(blobs.is_bass==row.is_bass)&(blobs.total_cx<row.total_cx))
        if row.state in '234':
            centers = get_kmean_cluster_centers(row, closed_img)
            for (y0,x0) in centers:
                staff_pitch = round(-2/line_sep * (y0%img_heights[0]) + 18 - 12*row.is_bass)
                pitch = semitone_map[staff_pitch%7] + 12*(staff_pitch//7)
                song_input.loc[len(song_input)] = ['1', x0, y0, line, measure, staff_pitch, pitch]
        else:
            song_input.loc[len(song_input)] = [row.state, row.cx, row.cy, line, measure, staff_pitch, pitch]
            
            

            
    return song_input
            

def parse_song_notes(song_input, line_sep):
    # Initialize key signals and dictionary maps
    measure_start_x = 0
    keysignature = {}
    accidentals = {}
    accidental_map = {'s':1, 'f':-1, 'n':0}
    pitch_to_string = { 
        'flat':{
            0:'C', 1:'Db', 2:'D', 3:'Eb', 4:'E',  5:'F', 
            6:'Gb', 7:'G', 8:'Ab', 9:'A', 10:'Bb', 11:'B'
            },
        'sharp':{        
            0:'C', 1:'C#', 2:'D', 3:'D#', 4:'E', 5:'F',
            6:'F#', 7:'G', 8:'G#', 9:'A', 10:'A#', 11:'B'
             }
    }
    
    
    # Sort 
    song_input = song_input.sort_values(by=['line','measure','cx'])

    # Apply key signal and accidentals to notes
    notes = pd.DataFrame(columns=['line', 'measure','pitch','x'])
    for index, row in song_input.iterrows():
        
        if row.state in 'sfn':
            has_target = np.any( 
                    (song_input.line==row.line)&
                    (song_input.measure==row.measure)&
                    (song_input.staff_pitch==row.staff_pitch)&
                    (song_input.cx - row.cx<2*line_sep)&
                    (song_input.cx - row.cx>0)
                )
            if has_target:
                # If accidental has target, add to temporary accidentals dict
                accidentals[row.staff_pitch] = accidental_map[row.state]
            else:
                # If no target, add to permanent keysignature. Define as measure start
                keysignature[row.staff_pitch%7] = accidental_map[row.state]
                measure_start_x = row.cx

        if row.state in '1':
            # Check if note pitch should be influenced by accidentals or keysignature
            # If accidental applies, do not apply keysignature
            pitch = row.pitch
            if row.staff_pitch in accidentals.keys():
                pitch = row.pitch + accidentals[row.staff_pitch]
            elif row.staff_pitch%7 in keysignature.keys():
                pitch = row.pitch + keysignature[row.staff_pitch%7]
                # I choose to leave as full x
            notes.loc[len(notes)] = [row.line, row.measure, pitch, row.cx]
        if row.state == 'm':
            # Reset accidentals on new measure. Reset measure_start_x
            accidentals = {}
            measure_start_x = row.cx

    # Create string labels
    key_type = 'flat' if sum(keysignature.values())<=0 else 'sharp'
    notes['string'] = notes.pitch.mod(12).map(pitch_to_string[key_type])
    notes[['line','measure','pitch']] = notes[['line','measure','pitch']].astype(int)

    # Group cx into chords
    notes = notes.sort_values(by=['line','x'])
    xmin, xmax, dx = 0, 0, 2*line_sep
    xarr = notes.x.values
    for j in range(len(xarr)):
        x = xarr[j]
        if x>xmax or x<xmin:
            xmin = int(x)
            xmax = x+dx
        xarr[j] = xmin
    
    # Sort output for grouping chords
    notes = notes.sort_values(by=['line','measure','x','pitch'])
    return notes, key_type

def append_measure_lines(img, blobs, line_sep):
    # Identify measure lines
    kernel = np.ones((int(3.5*line_sep), 1), dtype=np.uint8)
    vert_lines = ~cv.morphologyEx(img.copy(), cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(vert_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        M = cv.moments(c)
        # Zero area contours possible 
        if M['m00']==0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        is_far_in_x = np.abs(cx-blobs.cx)>1.5*line_sep
        is_far_in_y = np.abs(cy-blobs.cy)>3*line_sep
        # Measure lines must be either far away in x, or far away in y for all blobs
        if np.all(np.logical_or(is_far_in_x, is_far_in_y)):
            blobs.loc[len(blobs), ['state','cx','cy']] = ['m', cx, cy] 
    return blobs

def print_marked_music(img, song_input, line_sep):
    # Add colored tags to image
    color_map = { 
        '1':(0,0,255),
        '2':(0,0,255),
        '3':(255,255,0),
        's':(255,0,0),
        'f':(255,0,0),
        'n':(255,0,0),
        'm':(0,255,255),
        'none':(0,255,0)
    }

    color_orig = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for key, color in color_map.items():
        for index, (cx, cy) in song_input.loc[song_input.state==key, ['cx','cy']].iterrows():
            cv.circle(color_orig, (int(cx), int(cy)), 15, color, -1)
    cv.imwrite('Processed Images\\test_identified_music.jpg',color_orig)