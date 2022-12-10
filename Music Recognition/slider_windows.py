# %%
import numpy as np
import cv2 as cv
from functools import partial
from utils import get_ellipse_goodness

class SliderBaseClass():
    def __init__(self, img_file, sliders=None, window='default'):
        self.img = cv.imread(img_file, 0)
        self.window = window
        cv.namedWindow(self.window)

        # Universal trackbars
        self.scaling=2
        if sliders is None:
            sliders = ['scaling',self.scaling,5]
        else:
            sliders.append(['scaling',self.scaling,5])

        # Create sliders
        cv.namedWindow('sliders')
        for slider_name, slider_val, slider_max in sliders:

            # Create slider if valid attribute
            if hasattr(self, slider_name):
                setattr(self, slider_name, slider_val)
                setattr(self, 'slider_'+slider_name, partial(self.slider_event, slider_name))
                cv.createTrackbar(slider_name, 'sliders', 
                    getattr(self, slider_name), slider_max, 
                    getattr(self, 'slider_'+slider_name)
                )
            else:
                print(f'Warning: {slider_name} is not a valid slider attribute.')


        # Update window
        self.update()
        cv.resizeWindow("sliders", 500, 700)
        cv.waitKey()
        cv.destroyAllWindows()

    # Template for slider_event
    def slider_event(self, slider_name, val):
        if val != getattr(self, slider_name):
            setattr(self, slider_name, val)
            self.update()

    # Default update method
    def update(self):
        # Do something here
        self.show(self.img)
        

    def show(self, img):
        for _ in range(self.scaling):
            img = cv.pyrDown(img)
        cv.imshow(self.window, img)

class ContourDetection(SliderBaseClass):
    def __init__(self, *args, **kwargs):
        # Define attributes
        self.threshold = 127
        self.kernel_h = 5
        self.kernel_w = 5
        super().__init__(*args, **kwargs)

    def update(self):
        # Apply thresholding
        ret, thresh = cv.threshold(self.img, self.threshold, 255, 0)

        # Get closed image
        kernel = np.ones((self.kernel_h, self.kernel_w), dtype=np.uint8)
        img_opened = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

        # Find and draw contours
        green = [0, 255, 0]
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        color = cv.cvtColor(self.img, cv.COLOR_GRAY2BGR)
        cv.drawContours(color, contours, -1, green, 2)
        cv.imshow(self.window, color)




class LineDetection(SliderBaseClass):
    def __init__(self, *args, **kwargs):
        # Define attributes
        self.threshold = 127
        self.canny_min = 100
        self.canny_max = 200
        self.rho = 100
        self.theta = 10
        self.line_threshold = 20
        self.min_line_length = 20
        self.max_line_length = 30
        self.max_line_gap = 0
        super().__init__(*args, **kwargs)
        
    def update(self):
        # Apply thresholding
        ret, thresh = cv.threshold(self.img, self.threshold, 255, 0)
        color = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)

        # Get canny edges
        edges = cv.Canny(thresh, 
            threshold1=self.canny_min, threshold2=self.canny_max
        )

        # Find and draw lines
        green = (0, 255, 0)
        red = (0, 0, 255)
        lines = cv.HoughLinesP(255-thresh, self.rho/100, self.theta/10*np.pi/180, self.line_threshold, 
            minLineLength=self.min_line_length, maxLineGap=self.max_line_gap
        )
        for x1, y1, x2, y2 in lines.squeeze(1):
            line_length = np.sqrt((x2-x1)**2+(y2-y1)**2)
            if line_length<self.max_line_length:
                cv.line(color, (x1, y1), (x2, y2), green, 2)
            else:
                pass
                # cv.line(color, (x1,y1), (x2,y2), red, 2)
        
        self.show(color)

class Thresholding(SliderBaseClass):
    def __init__(self, *args, **kwargs):
        self.threshold = 127
        self.junk = 5
        super().__init__(*args, **kwargs)
        self.parent = super()

    def update(self):
        ret, thresh = cv.threshold(self.img, self.threshold, 255, 0)
        cv.imshow(self.window, thresh)


class MusicSliders(SliderBaseClass):
    def __init__(self, *args, **kwargs):
        # Define attributes
        self.threshold = 127
        self.min_area = 10
        self.max_area = 20
        self.min_e = 0
        self.max_e = 30
        super().__init__(*args, **kwargs)

    def update(self):
        green = (0, 255, 0)
        red = (0,0,255)     

        ret, gray = cv.threshold(self.img, self.threshold, 255, 0)
        contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        # Fill small contours
        for c in contours:
            area = cv.contourArea(c)
            ellipse = cv.fitEllipse(c)
            e = get_ellipse_goodness(c, ellipse)
            
            if (area>self.min_area) and (area<self.max_area) and (e>self.min_e) and (e<self.max_e):
                cv.drawContours(color, [c], -1, red, 5)
     
        self.show(color)
 


img_file = 'cleaned music.jpg'
sliders = [ 
    ['min_area', 1000, 4500],
    ['max_area', 2000, 4500],
    ['min_e', 0, 30],
    ['max_e', 5, 50]
]
test = MusicSliders(img_file, sliders=sliders)
# %%
