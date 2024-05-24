# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:11:32 2024

@author: afisher
"""
import cv2 as cv
import pickle
import numpy as np
import torch
import Utils.opencv as ucv
from Utils.pytorch import CNN
import matplotlib.pyplot as plt

import decimal
import math

gray = cv.imread( 'Images/pi_200.jpg', cv.IMREAD_GRAYSCALE)
_, binary_image = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV)


# Remove noise
image = ucv.remove_noise(binary_image, min_contour_area = 50)
# %%
def compute_pi_digits_chudnovsky(n):
    decimal.getcontext().prec = n + 10  # Set precision, with a little extra for intermediate calculations
    
    C = 426880 * decimal.Decimal(10005).sqrt()
    M = 1
    L = 13591409
    X = 1
    K = 6
    S = L

    for i in range(1, n):
        M = (M * (K**3 - 16*K)) // (i**3)
        L += 545140134
        X *= -262537412640768000
        S += decimal.Decimal(M * L) / X
        K += 12

    pi = C / S
    pi_digits = list(map(int, list(str(+pi).replace('.',''))))
    return pi_digits[:n]

# %% Load model
model_type = 'pytorch' #sklearn or pytorch

if model_type=='sklearn':
    model = pickle.load(open('Models/randomforest.pkl', 'rb'))
elif model_type=='pytorch':
    model = torch.load('Models/cnn.pkl')
else:
    raise ValueError('Model_type not recognized.')
        

# %% Read image
gray = cv.imread( 'Images/pi_200.jpg', cv.IMREAD_GRAYSCALE)
_, binary_image = cv.threshold(gray, 130, 255, cv.THRESH_BINARY_INV)


# Remove noise
image = ucv.remove_noise(binary_image, min_contour_area = 50)
color_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)


# Loop over contours (sorted by x)
contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

row_sum = image.sum(axis=1)
linestarts = np.where( (row_sum[:-1]==0)*(row_sum[1:]!=0) )[0]
contours = sorted(contours, key=lambda x: ( (x[0][0][1]>=linestarts).sum(), x[0][0][0]) )

labels = compute_pi_digits_chudnovsky(len(contours))
for j, contour in enumerate(contours):

    # Find bounding rect
    x, y, w, h = cv.boundingRect(contour)
    xc, yc, L = round(x+w//2), round(y+h//2), max(w, h)//2+1
    
    # Resize
    resized_image = cv.resize(image[yc-L:yc+L, xc-L:xc+L], (16, 16), interpolation=cv.INTER_AREA)
    
    # Apply model
    if model_type=='sklearn':
        pixels = resized_image.reshape(1, -1)/255
        prediction = model.predict(pixels)[0]
    else:
        pixels = torch.tensor(resized_image.reshape(1,16,16), dtype=torch.float32)
        outputs = model(pixels)
        _, tensor_pred = torch.max(outputs, 1)
        prediction = tensor_pred.numpy()[0]
    
    
    color = (0, 255, 0) if prediction==labels[j] else (0, 0, 255)
    cv.rectangle(color_image, (x,y), (x+w, y+h), color, 2)
    if prediction!=labels[j]:
        cv.putText(color_image, str(prediction), (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 3, color, 2)

ucv.show_image(color_image, scale=.35)
