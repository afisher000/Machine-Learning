# %%
import cv2 as cv
import numpy as np
from PIL import Image

# Read image file
img = cv.imread('test_staff.jpg')

# Split into rgb or merge
b,g,r = cv.split(img) #Slower than numpy indexing
img = cv.merge((b,g,r))

# Add border
top, bottom, left, right = [5,4,3,6]
constant_border = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, [255,0,0])

Image.fromarray(b).show()



# %%
