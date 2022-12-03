# %% 
import numpy as np
import cv2 as cv
from PIL import Image

img = cv.imread('test_staff.jpg')
gray = img.mean(axis=2)
gray = np.where(gray>240, 255, 0).astype('uint8')

# TODO 
# Use horizontal erosion to determine larger clumps
# Then you can tell where vertical lines are due to clumped notes or accidentals


def dilate_and_erode(img, shape):
    kernel = np.ones(shape, np.uint8)
    dilated = cv.dilate(img, kernel, iterations=1)
    return cv.erode(dilated, kernel, iterations=1)

def erode_and_dilate(img, shape):
    kernel = np.ones(shape, np.uint8)
    eroded = cv.erode(img, kernel, iterations=1)
    return cv.dilate(eroded, kernel, iterations=1)

no_vert_lines = dilate_and_erode(gray, (1,4))
no_horz_lines = dilate_and_erode(gray, (5,1))
# no_lines = dilate_and_erode(no_vert_lines, (5,1))
# all_vert_lines = dilate_and_erode(gray, (35,1))
# long_vert_lines = dilate_and_erode(gray, (50, 1))
# short_vert_lines = cv.bitwise_or(all_vert_lines, cv.bitwise_not(long_vert_lines))
# acc_lines = cv.bitwise_or(short_vert_lines, cv.bitwise_not(no_vert_lines))
# acc_lines_dilated = erode_and_dilate(acc_lines, (10, 1))

contours = cv.findContours(no_horz_lines, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
if len(contours)==2:
    contours=contours[0]
else:
    raise ValueError

for c in contours:
    print(c)
    x,y,w,h = cv.boundingRect(c)
    cv.rectangle(no_horz_lines, (x,y), (x+w,y+h), (0,255,0), 2)

Image.fromarray(no_horz_lines).show()



# %%
