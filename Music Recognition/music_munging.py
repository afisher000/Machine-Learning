# %%


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_ellipse_goodness, convert_to_jpg

# # # To be implemented
# Separate pages into lines (simple as how close first staff line is to top of page?)
# Fix categorize_shapes, save staff dimensions/mapping from inspect_line


class AnalyzeLine():
    def __init__(self, file, is_treble):
            _, gray = cv.threshold(cv.imread('testline.jpg', 0), 127, 255, 0)
            self.is_treble = is_treble
            self.staff_margin_factor = 3.2
            self.staff_spacing = None
            self.scale = None
            self.notes = pd.DataFrame(columns=['type','line', 'col', 'pitch'])
            self.blue, self.green, self.red = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
            self.cyan, self.magenta, self.yellow = [(255,255,0), (255,0,255), (0, 255, 255)]
            self.unit_length = None #pixel distance between staff lines

            self.data = pd.DataFrame(columns=['meas','col','pitch','object'])
            
            # Inspect line
            self.inspect_line(gray)
            
            # Fill elliptical notes
            gray = self.clean_music_line(gray)
            cv.imwrite('testline_cleaned.jpg', gray)
            gray = self.fill_contours(gray, min_area=400, max_area=1200, max_e=16.0)
            cv.imwrite('testline_elliptic_filled.jpg', gray)

            # Remove staff lines, OPEN to connect some flats
            gray = self.remove_staff_lines(gray)
            gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel=np.ones((1,3), np.uint8))
            cv.imwrite('testline_removed_staff.jpg', gray)
            
            # Fill remaining notes, CLOSE to remove all lines
            gray = self.fill_contours(gray, min_area=100, max_area=1000)
            gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, np.ones((8,8), dtype=np.uint8))
            cv.imwrite('testline_final.jpg', gray)
            
            # Categorize
            self.categorize_shapes()
    

    def inspect_line(self, gray):
        is_line = (gray.sum(axis=1)/gray.shape[1]) < 127
        is_line_start = np.logical_and(~is_line[:-1], is_line[1:])
        is_line_end = np.logical_and(is_line[:-1], ~is_line[1:])
        start_rows = np.where(is_line_start)[0] + 1
        end_rows = np.where(is_line_end)[0] + 1
        
        self.unit_length = np.mean(np.diff(start_rows))
        self.start_rows = start_rows
        self.end_rows = end_rows
        self.line_rows = self.start_rows/2 + self.end_rows/2
        
        
    def clean_music_line(self, gray):       
        # Build new image from staff contour rect and stem_rows
        new_gray = 255*np.ones(gray.shape, dtype=np.uint8)

        # Get bounding box or largest contour
        contours, _ = cv.findContours(255-gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv.contourArea)
        x,y,w,h = cv.boundingRect(c)

        # Get masks of vertical lines
        short_line_kernel = np.ones((140,1), dtype=np.uint8)
        short_lines_mask = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel=short_line_kernel)
        stem_rows = np.where(short_lines_mask.sum(axis=1)/short_lines_mask.shape[1] < 255)[0]

        # Find largest of union
        top = min(stem_rows.min(), y)
        bottom = min(stem_rows.max(), y+h)
        new_gray[top:bottom] = gray[top:bottom]
        return new_gray
    
    def remove_staff_lines(self, gray):
        # If pixels above and below are white, whiten column
        for start, end in zip(self.start_rows, self.end_rows):
            mask = gray[start-1,:]&gray[end+1,:]
            gray[start:end,:] = mask
        return gray


    def fill_contours(self, gray, min_area=None, max_area=None, max_e=None):
        contours, _ = cv.findContours(255-gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for c in contours:
            # Length of c must be >5 to fit ellipse
            if len(c)<5:
                continue

            # Check contour area
            area = cv.contourArea(c)
            if (min_area is not None) and (area<min_area):
                continue
            if (max_area is not None) and (area>max_area):
                continue
            
            # Check contour ellipticity
            if max_e is not None:
                ellipse = cv.fitEllipse(c)
                e = get_ellipse_goodness(c, ellipse)
                if e>max_e:
                    continue

            # Fill with black
            cv.drawContours(gray, [c], -1, 0, thickness=-1)
        return gray

    def save_note(self, _type, row, col):
        pitch = np.polyval(self.pitch_map, row).round()
        self.notes.loc[len(self.notes),:] = [_type, self.curline, col, pitch]

    def get_pitch_map(self):
        ret, gray = cv.threshold(cv.imread(f'line{self.curline}.jpg', 0), 127, 255, 0)
        line_starts, line_ends = self.get_line_rows_by_cond(img=gray, avg_pixel=127)
        lines = line_starts/2 + line_ends/2
        pitch_vector = [10,8,6,4,2] if self.curline%2==0 else [-2,-4,-6,-8,-10]
        return np.polyfit(lines, pitch_vector, 1)

    def categorize_shapes(self, gray):
        color = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        self.pitch_map = self.get_pitch_map()

        for c in contours:
            area = cv.contourArea(c)
            e = get_ellipse_goodness(c, cv.fitEllipse(c))
            x,y,w,h = cv.boundingRect(c)

            if (area<800) or (area>10000) or (e>40):
                continue

            # Assign to shapes
            s2 = self.scale**2 #Scale areas when staff_spacing varies from nominal
            if (area>1500*s2) and (area<3000*s2) and (e<10): #Single Notes
                cv.drawContours(color, [c], -1, self.red, 5)
                self.save_note('single', y+h/2, x+w/2)

            elif (area>3000*s2) and (area<4500*s2) and (e<30): #Double Notes
                cv.drawContours(color, [c], -1, self.blue, 5)
                self.save_note('single', y+h/4, x+w/2)
                self.save_note('single', y+3*h/4, x+w/2)

            elif (area>900*s2) and (area<1200*s2) and (e<10): #Flats
                cv.drawContours(color, [c], -1, self.green, 5)
                self.save_note('flat', y+h/2, x+w/2)

            elif (area>1300*s2) and (area<1500*s2) and (e<10): #Naturals
                cv.drawContours(color, [c], -1, self.yellow, 5)
                self.save_note('natural', y+h/2, x+w/2)

            elif (area>1500*s2) and (area<2000*s2) and (e>10) and (e<20): #Sharps
                cv.drawContours(color, [c], -1, self.cyan, 5)
                self.save_note('sharp', y+h/2, x+w/2)
            else:
                cv.drawContours(color, [c], -1, self.magenta, 5)

        cv.imwrite('highlighted notes.jpg', color)
        return
    
class AnalyzeSong():
    def __init__(self, file, songname):
        # Define attributes
        self.songname = songname

        # Read file, threshold, and munge
        jpg_file = convert_to_jpg(file)
        _, gray_page = cv.threshold(cv.imread(jpg_file, 0), 127, 255, 0)

        # Split into lines
        num_lines = self.split_music_into_lines(gray_page)
        line_objects = []
        for line in range(num_lines):
            line_file = f'line{line}.jpg'
            is_treble = (line%2==0)
            line_objects.append(AnalyzeLine(line_file, is_treble))


    def split_music_into_lines(self, gray):
        # Remove all-white columns
        non_white_cols = gray.sum(axis=0)<255*gray.shape[0]
        gray = gray[:, non_white_cols]

        # Compute staff_spacing and scale
        is_line = (gray.sum(axis=1)/gray.shape[1]) < 127
        is_line_start = np.logical_and(~is_line[:-1], is_line[1:])
        is_line_end = np.logical_and(is_line[:-1], ~is_line[1:])
        start_rows = np.where(is_line_start)[0] + 1
        end_rows = np.where(is_line_end)[0] + 1
        
        unit_length = np.mean(np.diff(start_rows[:5]))
        num_lines = len(start_rows)//5
        

        # Create jpgs for each line
        tops = list(map(int, start_rows[0::5] - 4*unit_length))
        bottoms = list(map(int, end_rows[4::5] + 4*unit_length))
        left = int(3.5*unit_length)
        right = int(-1.0*unit_length)
        cv.imwrite(f'testline.jpg', gray[tops[0]:bottoms[0],left:right])
        # [cv.imwrite(f'line{j}.jpg', gray[tops[j]:bottoms[j], left:right]) for j in range(num_lines)]
        return num_lines





test = AnalyzeSong('test_song_1.jpg', 'test_song')
test.notes.to_csv('note_data.csv', index=False)
# %%











    


# %%
