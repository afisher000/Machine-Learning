import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import utils_image_processing as uip
import utils_music_theory as umt
import utils_model as um


staff_margin = 4

def clean_music(img, blobs, line_sep):
    clean_img = img.copy()
    line_height = int((4+2*staff_margin)*line_sep)
    n_lines = round(img.shape[0]/line_height)
    
    for j in range(n_lines):
        top = j*line_height
        bottom = (j+1)*line_height
        dirty_line = img[top:bottom, :]
    
        # Populate mask with largest external contour (staff)
        mask = np.zeros_like(dirty_line)
        contours, _ = cv.findContours(255-dirty_line, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv.contourArea)
        cv.drawContours(mask, [c], -1, 255, -1)
    
        # Add white patches to cover notes
        is_correct_line = blobs.cy.between(top, bottom)
        is_not_none = blobs.state!='none'
        line_blobs = blobs[is_correct_line & is_not_none]
        for index, row in line_blobs.iterrows():
            if cv.pointPolygonTest(c, (int(row.cx), int(row.cy%line_height)), False)==-1:
                if not np.isnan(row.x):
                    x, y = int(row.x), int(row.y%line_height)
                    w, h = int(row.width*line_sep), int(row.height*line_sep)
                    cv.rectangle(mask, (x,y), (x+w,y+h), 255, -1)
    
        clean_line = ~cv.bitwise_and(~dirty_line, mask)
        clean_img[top:bottom] = clean_line
    return clean_img
    
def isolate_music_lines(img):
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
        top = int(line_starts[5*j] - staff_margin*line_sep)
        bottom = int(line_starts[4+5*j] + staff_margin*line_sep)
        left = int(3.5*line_sep)
        right = int(-1.0*line_sep)
        imgs.append(img[top:bottom,left:right].copy())
    img = np.vstack(imgs)
    return img, line_sep

def remove_words_from_line(img, line_sep):
    # Get bounding box or largest contour
    mask = 255-img
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv.contourArea)
    x,y,w,h = cv.boundingRect(c)

    # Get masks of vertical lines
    short_lines = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel=np.ones((int(4*line_sep),1), dtype=np.uint8))
    stem_rows = np.where(short_lines.sum(axis=1)/short_lines.shape[1] < 255)[0]
    
    # Strip uninteresting while keeping line_height same
    # IMPROVE
    top = min(stem_rows.min(), y)
    bottom = max(stem_rows.max(), y+h)
    img[:top] = 255
    img[bottom:] = 255
    return img


def munge_blob_data(closed_img, blobs, line_sep, line_height):
    blobs = blobs[blobs.state!='none']
    blobs['raw_line'] = blobs.cy.floordiv(line_height)
    blobs['line'] = blobs.raw_line.floordiv(2)
    blobs['is_bass'] = blobs.raw_line.mod(2).astype(bool)
    blobs['time'] = blobs.raw_line*blobs.cx.max() + blobs.cx
    blobs = blobs.sort_values(by=['raw_line','cx'])
    
    song_input = pd.DataFrame(columns=['state','cx','cy','line','measure','staff_pitch','pitch'])
    for index, row in blobs.iterrows():
        # CLEANUP?
        measure = sum((blobs.state=='m')&(blobs.is_bass==row.is_bass)&(blobs.time<row.time))
        
        # If multiple notes, apply kmeans clustering before saving
        if row.state in '234':
            rect = [row.x, row.y, row.width*line_sep, row.height*line_sep]
            n_clusters = int(row.state)
            cluster_centers = uip.get_cluster_centers(rect, closed_img, n_clusters)
            for (cy0,cx0) in cluster_centers:
                staff_pitch = umt.get_staff_pitch(cy0, line_sep, line_height, row.is_bass)
                pitch = umt.get_pitch_from_staff_pitch(staff_pitch)
                song_input.loc[len(song_input)] = [
                    '1', cx0, cy0, row.line, measure, staff_pitch, pitch
                ]
        else:
            staff_pitch = umt.get_staff_pitch(row.cy, line_sep, line_height, row.is_bass)
            pitch = umt.get_pitch_from_staff_pitch(staff_pitch)
            song_input.loc[len(song_input)] = [
                row.state, row.cx, row.cy, row.line, measure, staff_pitch, pitch
            ]
      
    return song_input
            

def parse_song_notes(song_input, line_sep):
    # Initialize key signals and dictionary maps
    measure_start_x = 0
    keysig = {}
    acc = {}
    acc_map = {'s':1, 'f':-1, 'n':0}
    song_input = song_input.sort_values(by=['line','measure','cx'])

    # Apply key signal and accidentals to notes
    notes = pd.DataFrame(columns=['line', 'measure','pitch','x'])
    for index, row in song_input.iterrows():
        if row.state in 'sfn':
            # CLEANUP?
            has_target = np.any( 
                    (song_input.line==row.line)&
                    (song_input.measure==row.measure)&
                    (song_input.staff_pitch==row.staff_pitch)&
                    (song_input.cx - row.cx<2*line_sep)&
                    (song_input.cx - row.cx>0)
                )
            if has_target:
                # If accidental has target, add to temporary accidentals dict
                acc[row.staff_pitch] = acc_map[row.state]
            else:
                # If no target, add to permanent keysignature. Define as measure start
                keysig[row.staff_pitch%7] = acc_map[row.state]
                measure_start_x = row.cx

        if row.state in '1':
            # Check if note pitch should be influenced by accidentals or keysignature
            # If accidental applies, do not apply keysignature
            pitch = row.pitch
            if row.staff_pitch in acc.keys():
                pitch = row.pitch + acc[row.staff_pitch]
            elif row.staff_pitch%7 in keysig.keys():
                pitch = row.pitch + keysig[row.staff_pitch%7]
                # I choose to leave as full x
            notes.loc[len(notes)] = [int(row.line), int(row.measure), int(pitch), row.cx]
        if row.state == 'm':
            # Reset accidentals on new measure. Reset measure_start_x
            accidentals = {}
            measure_start_x = row.cx

    # Create string labels
    notes['string'] = notes.pitch.apply(
        lambda x: umt.get_pitch_as_string(x, sum(keysig.values()))
    )
    

    # Group cx into chords
    # CLEANUP?
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
    return notes

