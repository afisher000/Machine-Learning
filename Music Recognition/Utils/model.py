# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:01:55 2022

@author: afisher
"""
import pickle
import cv2 as cv
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from Utils import io as uio

training_columns = [
    'area','width','height','aspectratio','extent','solidity','angle', 'faults'
    ]
data_columns = [
        'tag', 'cx', 'cy','area', 'x','y','w','h','width',
        'height','aspectratio','extent','solidity', 'angle', 'faults'
    ]

database_files = {
    'filling':'training_data_filling.csv',
    'notations':'training_data_notations.csv',
    'notes':'training_data_notes.csv'
}

model_files = { 
    'filling':'model_to_fill_contours.pkl',
    'notations':'model_to_identify_notations.pkl',
    'notes':'model_to_identify_notes.pkl'
}
notation_colors = { 
        's':(255,0,0), #sharp
        'n':(255,100,100), #natural
        'f':(255,200,200), #flat
        'm':(0,255,255), #measure
        'd':(100,255,255), #dot
        '2':(255,0,255),
        '3':(255,60,255),
        '4':(255,120,255),
        '6':(255,180,255),
        '8':(255,240,255),
        'q':(200,200,255), #1/16th rest
        'w':(100,100,255), #1/8th rest
        'e':(0,0,255), #1/4rest
        'r':(255,255,0), #half rest
        't':(255,255,100), #whole rest
        '\r':(100,100,100)
    }

note_colors = {
    '1':(255,0,0),
    '2':(0,255,0),
    '3':(0,0,255),
    '\r':(100,100,100)
    }

def annotate_contour(img, row, model_type, validation=False):
    if model_type == 'filling':
        if chr(row.tag)!='\r':
            fill_value = 100 if validation else 0
            cv.floodFill(img, None, (row.cx, row.cy), fill_value)
    elif model_type == 'notations':
        #If grayscale, convert to color
        if len(img.shape)==2: #Is grayscale
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        color = notation_colors[chr(row.tag)]
        cv.circle(img, (row.cx, row.cy), radius=15, color=color, thickness=-1)
    elif model_type == 'notes':
        if len(img.shape)==2: #Is grayscale
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        color = note_colors[chr(row.tag)]
        cv.circle(img, (row.cx, row.cy), radius=15, color=color, thickness=-1)  
    return img


def get_image_contours(img, model_type=None):
    # Ensure outside border is white
    img[[0,img.shape[0]-1],:]=255
    img[:,[0,img.shape[1]-1]]=255
    
    # RETR_CCOMP returns only two hierarchy levels, so if contour has 
    # no parent, it is white enclosing
    cs, hs = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    contours = []
    for j, c in enumerate(cs):
        if model_type is None:
            contours.append(c)
        elif model_type=='filling':
            if hs[0,j,3]==-1: #is white enclosing
                contours.append(c)
        else:
            if hs[0,j,3]!=-1: #is black enclosing
                 contours.append(c)
    return contours
    
def run_model(img, model_type, validation=False, verbose=False):
    # Get contour data from image
    contours = get_image_contours(img.copy(), model_type)
    data = get_contour_data(contours).reset_index(drop=True)

    # Apply model if exists
    model_file = model_files[model_type]
    if os.path.exists(model_file):
        model = pickle.load(open(model_file, 'rb'))
        X = data[training_columns].values
        data.tag = model.predict(X).astype(int)
        # if data.tag.dtype=='O':
            # data.tag = data.tag.apply(ord)
    else:
        print(f'Model file {model_file} does not exist...')
    
    # Convert columns to integers
    int_cols = ['tag','cx','cy','x','y','w','h']
    data[int_cols] = data[int_cols].astype(int)
        
    # Alter image
    for row in data.itertuples():
        img = annotate_contour(img, row, model_type, validation)

    if not verbose:    
        # Only return tag, centroid, and boundingrect
        data = data.loc[data.tag.apply(chr)!='\r', int_cols]
    return img, data
    
    
def get_contour_data(contours):
    line_sep = uio.get_song_params(['line_sep'])

    cxs, cys, areas, xs, ys, ws, hs, widths, heights = [], [], [], [], [], [], [], [], []
    aspect_ratios, extents, solidities, angles, tags = [], [], [], [], []
    faults = []
    for c in contours:
        # Faults can arise when m00=0 or len(c)<4 and ellipse cannot be fit.
        fault = 0
        try:
            M = cv.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00']) 
        except:
            cx, cy = 0, 0
            fault = 1
            
        
        area = cv.contourArea(c)
        x,y,w,h = cv.boundingRect(c)
        
        
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        try:
            _,(MA,ma),angle = cv.fitEllipse(c)
        except:
            MA, ma, angle = 0, 0, 0
            fault = 1

        # Ratios
        aspect_ratio = float(w)/h
        extent = float(area)/(w*h)
        try:
            solidity = float(area)/hull_area
        except:
            solidity = 0
            fault = 1

        # Normalize
        area = area/line_sep**2
        hull_area = hull_area/line_sep**2
        MA = MA/line_sep
        ma = ma/line_sep
        width = w/line_sep
        height = h/line_sep
        
        # # Append to lists
        cxs.append(cx)
        cys.append(cy)
        areas.append(area)
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)
        widths.append(width)
        heights.append(height)
        aspect_ratios.append(aspect_ratio)
        extents.append(extent)
        solidities.append(solidity)
        angles.append(angle)
        tags.append(13)
        faults.append(fault)
        
    # Create dataframe
    contour_df = pd.DataFrame(
        data = np.vstack([
            tags, cxs, cys, areas, xs, ys, ws, hs, widths, heights, 
            aspect_ratios, extents, solidities, angles, faults
        ]).T,
        columns=data_columns
    )
    
    return contour_df
    