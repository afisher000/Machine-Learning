# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 18:39:26 2022

@author: afisher
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_ellipse_goodness, convert_to_jpg
import pickle


# To implement 
# Scale distances to staff_line_separation



jpg_file = 'testline_cleaned.jpg'

# Create grayscale and color images
_, orig = cv.threshold(cv.imread(jpg_file, 0), 127, 255, 0)
gray = orig.copy()
filled_value = 100
mouse_clicks = pickle.load(open('saved_mouse_clicks.pkl','rb'))


# Get line_separation
is_line = (gray.sum(axis=1)/gray.shape[1]) < 127
is_line_start = np.logical_and(~is_line[:-1], is_line[1:])
line_starts = np.where(is_line_start)[0]
line_sep = np.diff(line_starts).mean()




# Mouse callback
def select_contour(event, x, y, flags, param):
    # Return if clicked on black
    if gray[y,x]==0:
        return
        
    # Fill or unfill contour
    if event == cv.EVENT_LBUTTONDOWN:
        fill_value = filled_value if gray[y,x]==255 else 255
        cv.floodFill(gray, None, (x,y), fill_value)
        mouse_clicks.append([x,y])

    # Update image
    cv.imshow('image', gray)
    return


# Main loop
cv.namedWindow='image'
cv.imshow('image', gray)
cv.setMouseCallback('image', select_contour)

# Flood fill initial mouseclicks
for (x,y) in mouse_clicks:
    cv.floodFill(gray, None, (x,y), filled_value)


while True:
    cv.imshow('image', gray)
    k = cv.waitKey(20)
    if k==ord('q'):
        break
    
pickle.dump(mouse_clicks, open('saved_mouse_clicks.pkl','wb'))
cv.destroyAllWindows()


# Create table of contour data by selection state
selected_mask = np.where(gray==filled_value, 255, 0).astype('uint8')
unselected_mask = 255-gray

selected_contours, _ = cv.findContours(selected_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
unselected_contours, _ = cv.findContours(unselected_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

def append_contour_data(contour_data, contours, is_selected=True):
    for c in contours:
        # Skip if too simple
        if len(c)<5:
            continue
        
        # Computations
        area = cv.contourArea(c)
        x,y,w,h = cv.boundingRect(c)
        hull = cv.convexHull(c)
        hull_area = cv.contourArea(hull)
        (x,y),(MA,ma),angle = cv.fitEllipse(c)
        
        # Scale distances by line_sep
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
        
        # Append to dataframe
        contour_data.loc[len(contour_data)] = [is_selected, x, y, area, w, h, aspect_ratio, extent, solidity, angle]
        

contour_data = pd.DataFrame(columns=['state', 'x', 'y', 'area','width','height','aspectratio','extent','solidity', 'angle'])
append_contour_data(contour_data, selected_contours, is_selected=True)
append_contour_data(contour_data, unselected_contours, is_selected=False)
contour_data0 = contour_data.copy()

# Remove uninteresting contours for plotting
# contour_data = contour_data[contour_data.area<2]

# Plot results
# cmap = plt.get_cmap('RdYlGn')
# contour_data.plot.scatter('extent','angle', c=contour_data.state, cmap=cmap)
    

# Create model
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

y = contour_data.state.astype(int)
X = contour_data.drop(columns=['state']).values
svc = SVC(gamma='auto')
svc.fit(X, y)
y0 = y
ypred = svc.predict(X)

# # PCA decomposition
# pca = PCA(n_components=2)
# X_r = pca.fit(X).transform(X)
# print(f'Explained variance (first two components): {str(pca.explained_variance_ratio_)}')
# fig, ax = plt.subplots()
# for color, j, target_name in zip(['red','green'],[0,1],['Unfilled','Filled']):
#     ax.scatter(X_r[y==j, 0], X_r[y==j, 1], color=color, alpha=0.8, lw=2, label=target_name)



# Check model fills contours
contours, _ = cv.findContours(255-orig, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour_data = pd.DataFrame(columns=['state', 'x', 'y','area','width','height','aspectratio','extent','solidity', 'angle'])
append_contour_data(contour_data, contours)
X = contour_data.drop(columns=['state']).values
contour_data.state = svc.predict(X)
gray = orig.copy()
for index, row in contour_data.iterrows():
    if row.state==1:
        cv.floodFill(gray, None, (int(row.x), int(row.y)), 0)
    
cv.imshow('image', gray)
