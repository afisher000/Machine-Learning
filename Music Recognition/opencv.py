# %%
import numpy as np
import cv2 as cv


class Test():
    def __init__(self):
        self.threshold = 127
        self.kernel_h = 3
        self.kernel_w = 3
        self.img = cv.imread('test_staff.jpg', 0)

        cv.namedWindow('contours')
        cv.createTrackbar('KernelH', 'contours', 1, 10, self.update)
        cv.createTrackbar('KernelW', 'contours', 1, 10, self.update)
        cv.createTrackbar('Threshold', 'contours', 0, 255, self.update)

        cv.setTrackbarPos("KernelH", "contours", self.kernel_h)
        cv.setTrackbarPos("KernelW", "contours", self.kernel_w)
        cv.setTrackbarPos("Threshold", "contours", self.threshold)
        self.update()
        
        cv.waitKey()
        cv.destroyAllWindows()



    def update(self, slider_value=0):
        # Update parameters
        self.threshold = cv.getTrackbarPos('Threshold','contours')
        self.kernel_h = cv.getTrackbarPos('KernelH','contours')
        self.kernel_w = cv.getTrackbarPos('KernelW', 'contours')


        # Apply thresholding
        ret, thresh = cv.threshold(self.img, self.threshold, 255, 0)

        # Get closed image
        kernel = np.ones((self.kernel_h, self.kernel_w), dtype=np.uint8)
        img_opened = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        # Find and draw contours
        green = [0, 255, 0]
        contours, hierarchy = cv.findContours(img_opened, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        color = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        cv.drawContours(color, contours, -1, green, 2)
        cv.imshow('contours', color)

test = Test()

# %%
