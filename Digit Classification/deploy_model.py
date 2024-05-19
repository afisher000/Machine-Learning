# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:11:32 2024

@author: afisher
"""

import cv2 as cv
import pickle
import numpy as np

import Utils.opencv as ucv


# %% Read image
gray = cv.imread( 'Images/pi.jpg', cv.IMREAD_GRAYSCALE)
_, binary_image = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV)


# Remove noise
image = ucv.remove_noise(binary_image, min_contour_area = 50)
color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)


# Find contours
contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

model = pickle.load(open('Models/randomforest.pkl', 'rb'))
# pixels = np.zeros((len(contours), 256))
for j, contour in enumerate(contours):

    # Find bounding rect
    x, y, w, h = cv.boundingRect(contour)
    xc, yc, L = round(x+w//2), round(y+h//2), max(w, h)//2+1
    
    # Resize
    resized_image = cv.resize(image[yc-L:yc+L, xc-L:xc+L], (16, 16), interpolation=cv.INTER_AREA)
    
    # Reshape pixels
    pixels = resized_image.reshape(1, -1)/255
    
    # Load model
    label = model.predict(pixels)
    
    cv.rectangle(color, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv.putText(color, str(label), (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
ucv.show_image(color)
