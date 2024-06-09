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
from collections import defaultdict


training_columns = [
    'area','width','height','aspectratio','extent','solidity','angle', 'faults'
    ]
data_columns = [
        'tag', 'cx', 'cy','area', 'x','y','w','h','width',
        'height','aspectratio','extent','solidity', 'angle', 'faults'
    ]

def get_image_contours(img, model_type=None):
    # Ensure outside border is white
    img[[0,img.shape[0]-1],:]=255
    img[:,[0,img.shape[1]-1]]=255
    

    if model_type == 'external':
        contours, _ = cv.findContours(~img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    else:
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
    
def run_notes_model(orig):
    img = orig.copy()
    
    # Get contour data from image
    contours = get_image_contours(img.copy(), 'external')
    data = get_contour_data(contours).reset_index(drop=True)
    
    # Apply model if exists
    model_file = 'Saved Models/notes.pkl'
    if os.path.exists(model_file):
        model = pickle.load(open(model_file, 'rb'))
        X = data[training_columns].values
        preds = model.predict(X)
    else:
        print(f'Model "{model_file}" does not exist...')
        
    labels = defaultdict(list)
    fill_value = 255
    for j in range(len(preds)):
        labels[preds[j]].append(contours[j])
    
    return labels
    

def run_accidental_model(orig):
    img = orig.copy()
    
    # Get contour data from image
    contours = get_image_contours(img.copy(), 'external')
    data = get_contour_data(contours).reset_index(drop=True)
    
    # Apply model if exists
    model_file = 'Saved Models/accidentals.pkl'
    if os.path.exists(model_file):
        model = pickle.load(open(model_file, 'rb'))
        X = data[training_columns].values
        preds = model.predict(X)
    else:
        print(f'Model "{model_file}" does not exist...')
        
    labels = defaultdict(list)
    fill_value = 255
    for j in range(len(preds)):
        labels[preds[j]].append(contours[j])
        if preds[j]!='z':
            x, y, w, h = cv.boundingRect(contours[j])
            cv.rectangle(img, (x,y), (x+w, y+h), fill_value, -1)
    
    return img, labels
    
    
    
def run_cleaning_model(orig):
    img = orig.copy()
    
    # Get contour data from image
    contours = get_image_contours(img.copy(), 'cleaning')
    data = get_contour_data(contours).reset_index(drop=True)
    
    # Apply model if exists
    model_file = 'Saved Models/cleaning.pkl'
    if os.path.exists(model_file):
        model = pickle.load(open(model_file, 'rb'))
        X = data[training_columns].values
        preds = model.predict(X)
    else:
        print(f'Model "{model_file}" does not exist...')
        
    # Remove non-music
    labels = defaultdict(list)
    fill_value = 255
    for j in range(len(preds)):
        labels[preds[j]].append(contours[j])
        if preds[j]=='n':
            x, y, w, h = cv.boundingRect(contours[j])
            cv.rectangle(img, (x,y), (x+w, y+h), fill_value, -1)
            
    return img, labels

def run_note_filling_model(orig):
    img = orig.copy()
    
    # Get contour data from image
    contours = get_image_contours(img.copy(), 'filling')
    data = get_contour_data(contours).reset_index(drop=True)
    
    # Apply model if exists
    model_file = 'Saved Models/note_filling.pkl'
    if os.path.exists(model_file):
        model = pickle.load(open(model_file, 'rb'))
        X = data[training_columns].values
        preds = model.predict(X)
    else:
        print(f'Model "{model_file}" does not exist...')
        
    # Fill image
    labels = defaultdict(list)
    fill_value = 0
    for j in range(len(preds)):
        labels[preds[j]].append(contours[j])
        if preds[j]=='y':
            centroid = int(data.loc[j,'cx']), int(data.loc[j, 'cy']) 
            cv.floodFill(img, None, centroid, fill_value)
            
    return img, labels

    
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
    