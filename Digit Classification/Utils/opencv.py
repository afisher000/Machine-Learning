# -*- coding: utf-8 -*-
"""
Created on Sat May 18 16:17:30 2024

@author: afisher
"""
import cv2 as cv
import numpy as np

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)  
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return rotated_image

def xshift_image(image, xshift):
    # Apply circular shift
    xshift = int(xshift)
    if xshift==0:
        return image
    
    shifted_image = np.roll(image, xshift, axis=1)
    
    # Remove spill over
    if xshift>0:
        shifted_image[:, :xshift] = 0
    else:
        shifted_image[:, xshift:] = 0
    return shifted_image
    
    
    
    
def show_image(image, scale=1):
    h, w = image.shape[:2]
    scaled_image = cv.resize(image, (round(w*scale), round(h*scale)), interpolation=cv.INTER_AREA)
    
    cv.imshow('Label', scaled_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def flood_fill_edges(image):
    h, w = image.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    for row in range(h):
        if image[row, 0] == 255:
            cv.floodFill(image, mask, (0, row), 0)
        if image[row, w-1] == 255:
            cv.floodFill(image, mask, (w-1, row), 0)
    for col in range(w):
        if image[0, col] == 255:
            cv.floodFill(image, mask, (col, 0), 0)
        if image[h-1, col] == 255:
            cv.floodFill(image, mask, (col, h-1), 0)
    return image
        
def remove_noise(image, min_contour_area = 50):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv.contourArea(contour)<min_contour_area:
            cv.drawContours(image, [contour], -1, (0, 0, 0), thickness=cv.FILLED)
    return image
                
def scale_image(image, N=16):
    # Should only be one contour
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    assert(len(contours)==1)
    
    # Find bounding rect
    x, y, w, h = cv.boundingRect(contours[0])
    xc, yc, L = round(x+w//2), round(y+h//2), max(w, h)//2+1
    
    # Resize
    resized_image = cv.resize(image[yc-L:yc+L, xc-L:xc+L], (N, N), interpolation=cv.INTER_AREA)
    
    return resized_image