# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 15:41:56 2022

@author: afisher
"""
import numpy as np
import cv2 as cv

# Add 'make_copy' as possible kwarg?

def show_image(image, scale=1):
    h, w = image.shape[:2]
    scaled_image = cv.resize(image, (round(w*scale), round(h*scale)), interpolation=cv.INTER_AREA)
    
    cv.imshow('Label', scaled_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def get_clusters(rect, img, n_clusters):
    # Ensure rect points are integers, get sub image
    x0,y0,w0,h0 = list(map(int, rect))
    subimg = img[y0:y0+h0, x0:x0+w0]
    
    # Get points in clusters
    ypoints, xpoints = np.where(subimg==0)
    points = np.float32(np.vstack([ypoints, xpoints]).T)
    
    # Apply kmeans clustering
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, rel_centers = cv.kmeans(points, n_clusters, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    centers = (rel_centers + [y0,x0]).astype(int)
    
    # Compute distances to centers
    dists = np.zeros((points.shape[0], n_clusters))
    for j in range(n_clusters):
        dists[:,j] = np.linalg.norm(points-rel_centers[j,:], axis=1)
    nearest_center = dists.argmin(axis=1) 
    
    # Compute rect boundary of adjacent points
    boundingRects = []
    for j in range(n_clusters):
        adj_points = points[nearest_center==j, :]
        y, x = (adj_points.min(axis=0) + [y0,x0]).astype(int)
        h, w = adj_points.ptp(axis=0).astype(int)
        boundingRects.append([x,y,w,h])    
    
    # Return the center and bounding rectangle of each cluster
    return centers, boundingRects


def check_rectangle_overlap(rect0, rect):
    x0,y0,w0,h0 = rect0
    x,y,w,h = rect
    
    # Check if x overlaps
    x_not_overlap = (x+w<=x0)|(x>=x0+w0)
    y_not_overlap = (y+h<=y0)|(y>=y0+h0)
    not_overlap = x_not_overlap | y_not_overlap
    overlap = np.logical_not(not_overlap)
    return overlap

def morphology_operation(img, element, operation):
    kernel = np.ones((int(element[0]), int(element[1])), dtype=np.uint8)
    morphed_img = cv.morphologyEx(img, operation, kernel)
    return morphed_img
    