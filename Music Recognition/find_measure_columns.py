# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd


# Use cleaned version without words!!!
# Measures also includes a place where stem only and note just barely
# Need to also check maybe 10 pix before and after measure

line_spacing = 30

img = cv.imread('line0_cleaned.jpg')
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_scale = gray_scale[:, 2500:3500]
ret, gray = cv.threshold(gray_scale, 125, 255, 0)
cond = (gray.sum(axis=1)/gray.shape[1]) < 127
line_starts = ((~cond[:-1])&cond[1:]).nonzero()[0]
line_ends = ((cond[:-1])&(~cond[1:])).nonzero()[0]+1

# Create patterns
vert_pattern = 255*np.ones((gray.shape[0], 1), np.uint8)
horz_pattern = 255*np.ones((gray.shape[0], 1), np.uint8)
horz_pattern[cond, 0] = 0 # Turn staff lines black
vert_pattern[line_starts[0]+1:line_ends[-1], 0] = 0


# Check patterns against gray columns
is_vert_pattern = (~gray^vert_pattern).all(axis=0) 
is_horz_pattern = (~gray^horz_pattern).all(axis=0)

is_poss_measure = np.logical_and(is_vert_pattern[:-1], ~is_vert_pattern[1:])

measure_cols = []
dcol = line_spacing//2
for col in np.nonzero(is_poss_measure)[0]:
    if is_horz_pattern[col-dcol] and is_horz_pattern[col+dcol]:
        measure_cols.append(col)
        

print(f'{measure_cols} confirmed measure lines')

cv.imshow('test', gray_scale)
cv.waitKey()
cv.destroyAllWindows()


# %%
