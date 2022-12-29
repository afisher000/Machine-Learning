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

training_columns = [
    'area','width','height','aspectratio','extent','solidity','angle'
    ]
data_columns = [
        'state', 'cx', 'cy','area', 'x','y','width',
        'height','aspectratio','extent','solidity', 'angle'
    ]

blob_colors = { 
        '1':(0,0,255),
        '2':(0,255,),
        '3':(255,0,0),
        's':(255,255,0),
        'f':(255,0,255),
        'n':(0,255,255),
        'm':(0,255,255),
        'none':(100,100,100)
    }

def identify_blobs_using_model(img, line_sep):
    blobs = pd.DataFrame(columns=data_columns)
    blobs = get_contour_data(img, blobs, line_sep)
    
    if not os.path.exists('model_to_identify_blobs.pkl'):
        blobs.state = '1'
        return blobs
    
    model = pickle.load(open('model_to_identify_blobs.pkl','rb'))
    X = blobs[training_columns].values
    blobs.state = model.predict(X)
    return blobs

def label_blobs_on_music(orig, blobs, line_sep):
    # Add colored tags to image
    color_orig = cv.cvtColor(orig, cv.COLOR_GRAY2BGR)
    for key, color in blob_colors.items():
        for index, (cx, cy) in blobs.loc[blobs.state==key, ['cx','cy']].iterrows():
            cv.circle(color_orig, (int(cx), int(cy)), 15, color, -1)
    return color_orig

def fill_contours_using_model(img, line_sep, fill_value=0):
    if not os.path.exists('model_to_fill_contours.pkl'):
        return img
    model = pickle.load(open('model_to_fill_contours.pkl','rb'))
    
    contour_data = pd.DataFrame(columns = data_columns)
    get_contour_data(img, contour_data, line_sep)
    X = contour_data[training_columns].values
    contour_data.state = model.predict(X)

    # TO CLEAN?
    for index, row in contour_data.iterrows():
        if row.state==1:
            cv.floodFill(img, None, (int(row.cx), int(row.cy)), fill_value)
    return img

def get_contour_data(mask, contour_df, line_sep, is_selected=True):
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # Faults can arise when m00=0 or len(c)<4 and ellipse cannot be fit.
        try: 
            M = cv.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00']) 
            
            area = cv.contourArea(c)
            x,y,w,h = cv.boundingRect(c)
            
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            _,(MA,ma),angle = cv.fitEllipse(c)
            
            # Normalize
            area = area/line_sep**2
            hull_area = hull_area/line_sep**2
            MA = MA/line_sep
            ma = ma/line_sep
            w = w/line_sep
            h = h/line_sep
            
            # Ratios
            aspect_ratio = float(w)/h
            extent = float(area)/(w*h)
            solidity = float(area)/hull_area
        except:
            continue
        
        # Append to dataframe
        contour_df.loc[len(contour_df)] = [
            is_selected, cx, cy, area, x,y, w, 
            h, aspect_ratio, extent, solidity, angle
        ]

    return contour_df
    
class Filling_Model_Validation():
    def __init__(self, img, line_sep, filled_value=100):
        self.orig = img.copy()
        self.img = img
        self.line_sep = line_sep
        self.filled_value = filled_value
        self.database = 'training_data_contours.csv'
        self.model_file = 'model_to_fill_notes.pkl'

        # Main loop
        cv.namedWindow='zoomed_out'
        cv.imshow('zoomed_out', self.zoomed_out_image())
        cv.setMouseCallback('zoomed_out', self.roi_callback)
        
        while True:
            cv.imshow('zoomed_out', self.zoomed_out_image())
            k = cv.waitKey(20)
            if k==ord('s'):
                self.save_to_database()
                break
            elif k==ord('q'):
                # Quit without saving
                break
        cv.destroyWindow('zoomed_out')
    
    def zoomed_out_image(self):
        return cv.pyrDown(cv.pyrDown(self.img))

    def selection_callback(self, event, x, y, flags, param):
        # Get x,y on original
        x += self.left
        y += self.top
        
        # Fill selected contour
        if event == cv.EVENT_LBUTTONUP:
            if self.img[y,x]==255:
                fill_value = self.filled_value
            elif self.img[y,x]==self.filled_value:
                fill_value = 255
            else:
                fill_value = 0
            cv.floodFill(self.img, None, (x,y), fill_value)
            cv.imshow('zoomed_in', self.img[self.top:self.bottom, self.left:self.right])
            
        
    def roi_callback(self, event, x, y, flags, param):
        # Fill or unfill contour
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.namedWindow='zoomed_in'
            padh = 500
            padv = 200
            self.top = max(0, 4*y-padv)
            self.bottom = min(self.img.shape[0], 4*y+padv)
            self.left = max(0, 4*x-padh)
            self.right = min(self.img.shape[1], 4*x+padh)
            
            cv.imshow('zoomed_in', self.img[self.top:self.bottom, self.left:self.right])
            cv.setMouseCallback('zoomed_in', self.selection_callback)
        return
    
    def save_to_database(self):
        # Compute selection masks
        selected_mask = np.where(self.img==self.filled_value, 255, 0).astype('uint8')
        unselected_mask = np.where(self.img==255, 255, 0).astype('uint8')

        # Update training data
        if os.path.exists(self.database):
            contour_data = pd.read_csv(self.database)
        else:
            contour_data = pd.DataFrame(columns = data_columns)
        get_contour_data(selected_mask, contour_data, self.line_sep, is_selected=True)
        get_contour_data(unselected_mask, contour_data, self.line_sep, is_selected=False)
        contour_data.to_csv(self.database, index=False)
        
        # Train model
        ytrain = contour_data.state.astype(int)
        Xtrain = contour_data[training_columns].values
        model = RandomForestClassifier()
        model.fit(Xtrain, ytrain)
        pickle.dump(model, open(self.model_file, 'wb'))
        
        # Print data for contours that were different?
        changes_mask = np.where(self.img==self.orig, 0, 255).astype('uint8')
        
class Identifying_Model_Validation():
    def __init__(self, orig, img, line_sep):
        self.orig = orig
        self.img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        self.line_sep = line_sep
        self.database = 'training_data_blobs.csv'
        self.model_file = 'model_to_identify_blobs.pkl'
        
        # Label image
        self.blobs = identify_blobs_using_model(img, line_sep)
        self.labeled_img = label_blobs_on_music(orig, self.blobs, line_sep)

        
        # Main loop
        cv.namedWindow='zoomed_out'
        cv.imshow('zoomed_out', self.zoomed_out_image())
        cv.setMouseCallback('zoomed_out', self.roi_callback)
        
        while True:
            cv.imshow('zoomed_out', self.zoomed_out_image())
            k = cv.waitKey(20)
            if k==ord('s'):
                self.save_to_database()
                break
            elif k==ord('q'):
                # Quit without saving
                break
        cv.destroyWindow('zoomed_out')
    
    def zoomed_out_image(self):
        return cv.pyrDown(cv.pyrDown(self.labeled_img))


    def roi_callback(self, event, x, y, flags, param):
        # Fill or unfill contour
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.namedWindow='zoomed_in'
            
            # Find blob with closest centroid
            idx = np.argmin((self.blobs.cx-4*x)**2+(self.blobs.cy-4*y)**2)
            x, y, cx, cy, height, width = self.blobs.loc[idx, ['x','y','cx','cy','height','width']]
            pad = 1
            top = int(y)-pad
            bottom = int(y + height*self.line_sep)+pad
            left = int(x)-pad
            right = int(x + width*self.line_sep)+pad
            
            # Show closeup, respond with keypress
            cv.imshow('zoomed_in', self.orig[top:bottom, left:right])
            key = cv.waitKey(0)
            cv.destroyWindow('zoomed_in')
            
            # Update blob and color label
            if key in list(map(ord, '1234fsn')) or key==13: # 13==Enter
                state = 'none' if key==13 else chr(key)
                self.blobs.loc[idx, 'state'] = state
                cv.circle(self.labeled_img, (int(cx), int(cy)), 15, blob_colors[state], -1)
        return
    
    def save_to_database(self):
        # Update training dataq
        if os.path.exists(self.database):
            blob_data = pd.read_csv(self.database)
        else:
            blob_data = pd.DataFrame(columns=data_columns)
         
        blob_data = pd.concat([blob_data, self.blobs])
        blob_data.to_csv(self.database)
        
        # Train model
        ytrain = blob_data.state
        Xtrain = blob_data[training_columns].values
        model = RandomForestClassifier()
        model.fit(Xtrain, ytrain)
        pickle.dump(model, open(self.model_file, 'wb'))
