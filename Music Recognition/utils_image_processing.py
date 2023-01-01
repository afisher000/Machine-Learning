# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:41:56 2022

@author: afisher
"""
import numpy as np
import cv2 as cv

# Add 'make_copy' as possible kwarg?

def get_cluster_centers(rect, img, n_clusters):
    # Ensure rect points are integers, get sub image
    x,y,w,h = list(map(int, rect))
    subimg = img[y:y+h, x:x+w]
    
    # Get points in clusters
    ypoints, xpoints = np.where(subimg==0)
    points = np.float32(np.vstack([ypoints, xpoints]).T)
    
    # Apply kmeans clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv.kmeans(points, n_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # Return the centers for img
    return centers + [y,x]



def morphology_operation(img, element, operation):
    kernel = np.ones((int(element[0]), int(element[1])), dtype=np.uint8)
    morphed_img = cv.morphologyEx(img, operation, kernel)
    return morphed_img
    