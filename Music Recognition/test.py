# %%
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd


# Use cleaned version without words!!!
# Measures also includes a place where stem only and note just barely
# Need to also check maybe 10 pix before and after measure
ret, gray = cv.threshold(cv.imread('line0.jpg',0), 125, 255, 0)
cond = (gray.sum(axis=1)/gray.shape[1]) < 127
line_starts = ((~cond[:-1])&cond[1:]).nonzero()[0]
line_ends = ((cond[:-1])&(~cond[1:])).nonzero()[0]+1

measure_line = 255*np.ones((gray.shape[0],1), np.uint8)
measure_line[line_starts.min()+1:line_ends.max()] = 0


truth = (~gray^measure_line)
lines = truth.all(axis=0)
measures = ((~lines[:-1])&lines[1:]).nonzero()[0]
cv.imshow('test', truth)
cv.waitKey()
cv.destroyAllWindows()


# %%
