import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from Utils import io as uio
from Utils import image_processing as uip
from Utils import music_theory as umt


def remove_staff_lines(img):
    # Find start and end of staff lines
    is_line = (img.sum(axis=1)/img.shape[1]) < 127
    line_starts = np.logical_and(~is_line[:-1], is_line[1:]).nonzero()[0]+1
    line_ends = np.logical_and(is_line[:-1], ~is_line[1:]).nonzero()[0]+1
    
    # Turn line white where pixel above and below line is white
    for start, end in zip(line_starts, line_ends):
        line_fill = (img[start-1,:])&(img[end,:])
        img[start:end+1,:] = line_fill
    return img

    
def clean_music(img):
    line_sep, n_lines, line_height = uio.get_song_params(
        ['line_sep','n_lines','line_height']
    )
    
    # Compute center of staffs
    is_horz_line = img.sum(axis=1)/img.shape[1]<127
    line_starts = np.logical_and(~is_horz_line[:-1], is_horz_line[1:]).nonzero()[0]+1
    line_ends = np.logical_and(is_horz_line[:-1], ~is_horz_line[1:]).nonzero()[0]+1
    staff_centers = (line_starts[0::5]/2 + line_ends[4::5]/2).astype(int)

    # Remove white columns and vertical line that would connect treble and bass
    is_white_col = img.sum(axis=0)/img.shape[0]==255
    img = img[:, ~is_white_col]
    is_vert_line = (img.sum(axis=0)/img.shape[0]) < 170
    img[:, is_vert_line] = 255

    # Fill bounding rects of staff contours, then floodfill white to isolate words
    words_img = img.copy()
    contours, _ = cv.findContours(~words_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cs = sorted(contours, key=cv.contourArea)[-n_lines:]
    for c in cs:
        x,y,w,h = cv.boundingRect(c)
        words_img[y:y+h, :] = 0
        cv.floodFill(words_img, None, (x+w//2, y+h//2), 255)

    # Combine with original to remove words
    no_words_img = cv.bitwise_or(
        img,
        cv.bitwise_not(words_img)
    )
    cv.imwrite('Test\\no_words.jpg', no_words_img)


    # Add contours onto new image with equal line heights
    size = (n_lines*line_height, words_img.shape[1])
    cleaned_img = np.full(size, 255, dtype=np.uint8)
    cs, hs = cv.findContours(~no_words_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cs:
        x,y,w,h = cv.boundingRect(c)

        # Compute line based on centroid 
        line = np.abs((y+h//2)-staff_centers).argmin(axis=0)
        
        # Update new image
        row_shift = int((line+0.5)*line_height) - staff_centers[line]
        rows = np.s_[y:y+h]
        dest_rows = np.s_[y+row_shift: y+h+row_shift]
        cols = np.s_[x:x+w]
        
        # Check for clipping on boundary
        if (y+row_shift<0) or (y+h+row_shift>cleaned_img.shape[0]):
            continue
        else:
            cleaned_img[dest_rows,cols] = no_words_img[rows,cols]

    return cleaned_img
    



def separate_notations(notations, orig):
    line_sep, line_height = uio.get_song_params(['line_sep', 'line_height'])
    notations['total_pixel'] = notations.cx + notations.cy//(2*line_height)*orig.shape[1]
    
    # Reset tag as index
    notations = notations.set_index('tag')
    
    # Get measures
    measures = notations.loc[ord('m')].reset_index()

    # Get rests, add time column based on tag
    rest_duration_map = {'q':.25, 'w':.5, 'e':1, 'r':2, 't':4}
    rest_index = notations.index.intersection( map(ord, list('qwert')))
    rests = notations.loc[rest_index].reset_index()
    rests['duration'] = rests.tag.apply(chr).map(rest_duration_map)
    
    # Get accidentals
    idx = notations.index.intersection(map(ord, list('sfn')))
    accs = notations.loc[idx].reset_index()
    accs.loc[accs.tag.apply(chr)=='f', 'cy'] += line_sep//3
    
    # Get dots
    dots = notations.loc[ord('d')].reset_index()
    

    return measures, rests, accs, dots

def compute_is_filled(orig, filled_img, notes):
    fill_mask = cv.bitwise_and(orig, cv.bitwise_not(filled_img))
    
    # Compute is_filled column for notes
    notes['is_filled'] = 0
    for index, x, y, w, h in notes[['x','y','w','h']].itertuples():
        is_filled = int(fill_mask[y:y+h, x:x+w].sum()>0)
        notes.loc[index, 'is_filled'] = is_filled
    return notes
    
def remove_nonnotes(img, nonnotes):
    for index, tag, x, y, w, h in nonnotes[['tag','x','y','w','h']].itertuples():
        if chr(tag)!='\r':
            cv.rectangle(img, (x,y), (x+w, y+h), 255,-1)
    return img

        
def separate_grouped_notes(closed_img, grouped_notes):
    # Create notes table that contains correct centroid and bounding rects for notes
    notes = grouped_notes[grouped_notes.tag.apply(chr)=='1'].reset_index(drop=True)
    for index, tag, x, y, w, h in grouped_notes[['tag','x','y','w','h']].itertuples():
        if chr(tag) in list('234'):
            rect = [x, y, w, h]
            n_clusters = int(chr(tag))
            centers, bounding_rects = uip.get_clusters(rect, closed_img, n_clusters)
            for center, bounding_rect in zip(centers, bounding_rects):
                # Table row = [tag, cx, cy, x, y, w, h]
                notes.loc[len(notes)] = [ord('1'), center[1], center[0], *bounding_rect] 
    return notes



def compute_stems_and_tails(img, notes):
    line_sep = uio.get_song_params(['line_sep'])
    
    # Operations to isolate note stems
    no_stems = uip.morphology_operation(img, (.25*line_sep, .25*line_sep), cv.MORPH_DILATE)
    hollowed_notes = ~((~img)&(no_stems))
    stems = uip.morphology_operation(hollowed_notes, (1.5*line_sep,1), cv.MORPH_CLOSE)
    dilated_stems = uip.morphology_operation(
        stems, (.25*line_sep, .25*line_sep), cv.MORPH_ERODE
    )
    stem_contours, _ = cv.findContours(~dilated_stems, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    
    def count_tails(col_arr, ymin, ymax):
        is_white = True
        tails = 0
        for y, val in enumerate(col_arr):
            if val==0 and is_white and (y<ymin or y>ymax):
                tails += 1
                is_white = False
            elif val!=0 and not is_white:
                is_white = True
        return tails
        
    # Compute tails and is_stemmed
    notes[['tails', 'is_stemmed']] = 0
    for c in stem_contours:
        x,y,w,h = cv.boundingRect(c)
        is_overlapping = uip.check_rectangle_overlap(
            (x,y,w,h), (notes.x,notes.y,notes.w,notes.h)
        )
    
        ymin = notes.loc[is_overlapping, 'y'].min() - 0.5*line_sep
        ymax = notes.loc[is_overlapping, ['y','h']].sum(axis=1).max() + 0.5*line_sep
        # Fill note rects with white, compute tails
        for row in notes.loc[is_overlapping].itertuples():
            img[row.y:row.y+row.h, row.x-row.w:row.x+2*row.w] = 255     

        tails = max(
             count_tails(img[y:y+h, x], ymin-y, ymax-y),
             count_tails(img[y:y+h, x+w], ymin-y, ymax-y),
        )
        notes.loc[is_overlapping, ['tails']] = tails
        notes.loc[is_overlapping, ['is_stemmed']] = 1

    return notes


def apply_accidentals(notes, accs):    
    line_sep, line_height = uio.get_song_params(['line_sep', 'line_height'])
    
    # Apply accidentals to closest note
    column_dict = {
        's':'is_sharped',
        'f':'is_flatted',
        'n':'is_naturaled',
    }
    for col in column_dict.values():
        notes[col] = 0
    
    for index, cx, cy in notes[['cx','cy']].itertuples():
        dists = np.array([cx,cy]) - accs[['cx','cy']]

        # Check for valid accidentals
        is_valid_acc = np.logical_and(
            dists.cx.between(0, 3*line_sep),
            dists.cy.abs()<0.25*line_sep,
        )
        
        
        # If one accidental, apply to note and drop from accs
        acc_idx = accs.index[is_valid_acc].values
        if len(acc_idx)==1:
            key = chr(accs.loc[acc_idx[0], 'tag'])
            notes.loc[index, column_dict[key]] = 1
            accs = accs.drop(acc_idx)
        elif len(acc_idx)>1:
            print('More than 1 accidental nearby! Skipping note')


    return notes

def apply_dots(notes, dots):
    line_sep = uio.get_song_params(['line_sep'])
    notes = notes.reset_index(drop=True)
    notes['is_dotted'] = 0

    for index, cx, cy in notes[['cx','cy']].itertuples():
        dot_dists = np.array([cx, cy]) - dots[['cx','cy']]

        # Apply dots
        is_close = np.logical_and(
            dot_dists.cx.between(-3*line_sep, 0),
            dot_dists.cy.abs()<0.75*line_sep
        )
        
        if is_close.sum()>1:
            notes.loc[index, 'is_dotted'] = 1
        elif is_close.sum()==1:
            # Make sure no note same pitch before dot
            dot_pos = dot_dists[is_close].values[0]
            note_dists = np.array([cx, cy]) - notes[['cx','cy']]
            is_note_between = np.logical_and(
                note_dists.cx.between(dot_pos[0], -0.5*line_sep),
                note_dists.cy.between(dot_pos[1]-.75*line_sep, dot_pos[1]+.75*line_sep)
            )
            if is_note_between.sum()==0:
                notes.loc[index, 'is_dotted'] = 1

    return notes
    

def apply_keysignature(notes, orig, accs, measures):
    line_height = uio.get_song_params(['line_height'])
        
    # Remaining accs are keysignature, isolate treble first line
    keysig = {}
    keysig_accs = accs[accs.cy<line_height]
    keysig['acc'] = 'f' if (keysig_accs.tag==ord('f')).sum() else 's'
    keysig['pitches'] = umt.get_pitch(keysig_accs.cy)%12
    
    # Initialize key signals and dictionary maps
    acc_map = {'s':1, 'f':-1, 'n':0}
    acc_staff_pitches = {}
    
    notes['staff_pitch'] = umt.get_staff_pitch(notes.cy)
    notes['pitch'] = umt.get_pitch(notes.cy)
    notes['total_pixel'] = notes.cx + notes.cy//(2*line_height)*orig.shape[1]
    
    notes_and_measures = pd.concat([notes, measures]).sort_values(by='total_pixel')
    for row in notes_and_measures.itertuples():
        # Reset accidentals for new measure
        if chr(int(row.tag))=='m':
            acc_staff_pitches = {}
        else:
            # Check for accidentals
            if row.is_sharped:
                notes.loc[row.Index, 'pitch'] += 1
                acc_staff_pitches[row.staff_pitch] = 's'
            elif row.is_flatted:
                notes.loc[row.Index, 'pitch'] -= 1
                acc_staff_pitches[row.staff_pitch] = 'f'
            elif row.is_naturaled:
                acc_staff_pitches[row.staff_pitch] = 'n'
            
            # Check if in accidentals (must be exact line)
            elif row.staff_pitch in acc_staff_pitches.keys():
                notes.loc[row.Index, 'pitch'] += acc_map[acc_staff_pitches[row.staff_pitch]]
            
            # Check if in keysig
            elif row.pitch%12 in keysig['pitches']:
                notes.loc[row.Index, 'pitch'] += acc_map[keysig['acc']]
    notes['label'] = notes.pitch.apply(lambda x: umt.get_pitch_as_string(x, keysig['acc']))
    return notes, keysig


def compute_beats(notes, rests):
    line_sep = uio.get_song_params(['line_sep'])
    
    # Make sure notes is sorted so indices line up when adding later
    notes = notes.sort_values(by='total_pixel')
    notes_and_rests = pd.concat([notes.copy(), rests]).sort_values(by='total_pixel').reset_index(drop=True)
    
    def check_chord_overlap(j, idxs):
        rect0 = notes_and_rests.loc[j, ['x','y','w','h']]
        for idx in idxs:
            rect = notes_and_rests.loc[idx, ['x','y','w','h']]
            if uip.check_rectangle_overlap(rect0, rect):
                return True
        return False
        
    beats = {0}
    dpixels = 1.75*line_sep
    idxs = []
    pixels = notes_and_rests.total_pixel.values
    pixel_limit = pixels[0]+dpixels
    chord_counter = 0
    for j, pixel in enumerate(pixels):
        if pixel>pixel_limit:
            is_overlapping = np.logical_or(
                check_chord_overlap(j, idxs), #current with last chord
                check_chord_overlap( j-1, range(j, min(j+4,len(pixels)) ) ) #previous with next chord
            )
            if not is_overlapping:
                # Assign chord and beat, update beats set
                min_beat = min(beats)
                notes_and_rests.loc[idxs, 'beat'] = min_beat
                notes_and_rests.loc[idxs, 'chord'] = chord_counter
                durations = notes_and_rests.loc[idxs, 'duration']
                beats.remove(min_beat)
                beats.update(min_beat + np.unique(durations))
                
                idxs = [j]
                pixel_limit = pixel + dpixels
                chord_counter += 1
            else:
                # Update pixel limit, add to chord
                pixel_limit = pixel + dpixels
                idxs.append(j)
        # elif j+1==len(pixels):
        #     # If last note, assign chord and beat
        #     idxs.append(j)

        else:
            # Add to chord
            idxs.append(j)
    notes_and_rests.loc[idxs, 'beat'] = min(beats)
    notes_and_rests.loc[idxs, 'chord'] = chord_counter

    notes_idx = notes_and_rests.tag.apply(chr)=='1'
    notes.loc[:, ['beat','chord']] = notes_and_rests.loc[notes_idx, ['beat','chord']].values
    # notes.chord = notes.chord.astype(int)
    return notes

