# %%


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_ellipse_goodness, convert_to_jpg


class MusicMunging():
    def __init__(self, file):
        # Define attributes
        self.blue, self.green, self.red = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.cyan, self.magenta, self.yellow = [(255,255,0), (255,0,255), (0, 255, 255)]
        self.staff_margin_factor = 3.2
        self.staff_spacing = None
        self.scale = None
        self.notes = pd.DataFrame(columns=['type','line', 'col', 'pitch'])

        # Read file, threshold, and munge
        jpg_file = convert_to_jpg(file)
        _, gray_page = cv.threshold(cv.imread(jpg_file, 0), 127, 255, 0)

        # Split into lines
        num_lines = self.split_music_into_lines(gray_page)
        for line in range(num_lines):
            self.analyze_line(f'line{line}.jpg')

    def analyze_line(self, file):
            _, gray = cv.threshold(cv.imread(file, 0), 127, 255, 0)

            # Fill elliptical notes
            gray = self.clean_music_line(gray)
            gray = self.fill_contours(gray, min_area=600, max_area=900, max_e=6.0)

            # Remove staff lines, OPEN to connect some flats
            gray = self.remove_staff_lines(gray)
            gray = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel=np.ones((1,3), np.uint8))

            # Fill remaining notes, CLOSE to remove all lines
            gray = self.fill_contours(gray, min_area=100, max_area=1000)
            gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, np.ones((8,8), dtype=np.uint))

            # Categorize
            self.categorize_shapes()


    def get_line_rows_by_cond(self, gray, avg_pixel=127):
        # Return starts and ends of lines where row average is less than avg_pixel.
        cond = (gray.sum(axis=1)/gray.shape[1]) < avg_pixel
        line_starts = ((~cond[:-1])&cond[1:]).nonzero()[0]
        line_ends = ((cond[:-1])&(~cond[1:])).nonzero()[0]+1
        return line_starts, line_ends


    def split_music_into_lines(self, gray):
        # Remove all-white columns
        non_white_cols = gray.sum(axis=0)<255*gray.shape[0]
        gray = gray[:, non_white_cols]

        # Compute staff_spacing and scale
        line_starts, line_ends = self.get_line_rows_by_cond(gray)
        line_sep = np.diff(line_starts[:4]).mean()
        num_lines = len(line_starts)//5

        # Could be moved
        self.staff_spacing = line_sep
        self.scale = self.staff_spacing/38.7
        

        # Create jpgs for each line
        tops = list(map(int, line_starts[0::5] - 4*line_sep))
        bottoms = list(map(int, line_ends[4::5] + 4*line_sep))
        left = int(3.5*line_sep)
        right = int(-1.0*line_sep)
        [cv.imwrite(f'line{j}.jpg', gray[tops[j]:bottoms[j], left:right]) for j in range(num_lines)]
        return num_lines

    def clean_music_line(self, gray):       
        # Build new image from staff contour rect and stem_rows
        new_gray = 255*np.ones(gray.shape, dtype=np.uint8)

        # Get bounding box or largest contour
        contours, _ = cv.findContours(255-gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        c = max(contours, key = cv.contourArea)
        x,y,w,h = cv.boundingRect(c)

        # Get masks of vertical lines
        short_lines = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel=np.ones((140,1), dtype=np.uint8))
        stem_rows = np.where(short_lines.sum(axis=1)/short_lines.shape[1] < 255)[0]

        # Find largest of union
        top = min(stem_rows.min(), y)
        bottom = min(stem_rows.max(), y+h)
        new_gray[top:bottom] = gray[top:bottom]
        return new_gray


    def remove_staff_lines(self, gray):
        line_starts, line_ends = self.get_line_rows_by_cond(gray)

        # If pixels above and below are white, whiten column
        for start, end in zip(line_starts, line_ends):
            mask = gray[start,:]&gray[end,:]
            gray[start+1:end,:] = mask
        return gray


    def fill_contours(self, gray, min_area=None, max_area=None, max_e=None):
        contours, _ = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
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

test = MusicMunging('test_song_1.pdf')
test.notes.to_csv('note_data.csv', index=False)
# %%











    


# %%
