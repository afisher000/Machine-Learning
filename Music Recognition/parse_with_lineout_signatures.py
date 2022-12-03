# %%
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

# Ideas
# Scan +/- 1 staff line diameter about line. Look at patterns of 
# "on"/"off" for lineouts

class MUSIC():
    def __init__(self, file):
        self.RGB = np.array(Image.open(file))
        self.GRAY = self.RGB.mean(axis=2)<220
        self.characterize_staff()
        self.QTR_LENGTH = 1.20*self.STAFF_LINE_SEP

        horz_sum = self.GRAY.sum(axis=1)/self.GRAY.shape[1]
        cond = horz_sum>0.9
        starts = self.get_false_to_true(cond)
        ends = self.get_true_to_false(cond)
        for start, end in zip(starts, ends):
            print(start)
            print(end)
            for col in range(self.GRAY.shape[1]):
                # print(self.GRAY[start,col] and self.GRAY[end,col])
                if ~self.GRAY[start,col] and ~self.GRAY[end,col]:
                    self.GRAY[start+1:end, col] = False

        
        Image.fromarray(self.GRAY).show()

        # self.inspect_music_element(self.STAFF_LINES[3], (100,250))

        # # # Annotate notes
        # [self.annotate(line) for line in self.STAFF_LINES]
        # Image.fromarray(self.RGB).show()

    def annotate(self, line):
        mkr = 3 #marker size
        low, mlow, mhigh, high = self.get_signals_for_line(line)

        centers, widths = self.parse_signal(mlow&mhigh)
        for j, (center, width) in enumerate(zip(centers, widths)):
            # Hardcoded assignment
            # Check if quarter note
            if (width>=0.9*self.QTR_LENGTH) and (width<=1.1*self.QTR_LENGTH):
                self.RGB[line-mkr:line+mkr, center-mkr:center+mkr, 1] = 255

    def inspect_music_element(self, line, xlim):
        low, mlow, mhigh, high = self.get_signals_for_line(line)

        fig, ax = plt.subplots()
        ax.plot(low-1.1)
        ax.plot(mlow)
        ax.plot(mhigh+1.1)
        ax.plot(high+2.2)
        ax.set_xlim(xlim)
        
        dy = self.STAFF_LINE_SEP
        sub_RGB = self.RGB[line-dy:line+dy,xlim[0]:xlim[1]]
        Image.fromarray(sub_RGB).show()

    def parse_signal(self, cond):
        starts = self.get_false_to_true(cond)
        ends = self.get_true_to_false(cond)

        # Compute centers with ceil as starts/ends used left index
        centers = np.ceil(starts/2 + ends/2).astype('int')
        widths = (ends - starts).astype('int')
        return centers, widths

    def characterize_staff(self):
        horz_sum = self.GRAY.sum(axis=1)/self.GRAY.shape[1]
        centers, widths = self.parse_signal(horz_sum>0.9)
        line_fit = np.polyfit(np.arange(10,1,-2), centers, 1)
        self.STAFF_LINES = np.ceil(np.polyval(line_fit, np.arange(0, 14))).astype('int')
        self.STAFF_WIDTH = widths.max()
        self.STAFF_LINE_SEP = np.diff(centers).mean().astype('int')


    def get_false_to_true(self, cond):
        is_false_to_true = (~cond[:-1])&cond[1:] 
        return is_false_to_true.nonzero()[0]

    def get_true_to_false(self, cond):
        is_true_to_false = cond[:-1]&(~cond[1:])
        return is_true_to_false.nonzero()[0]+1

    def get_lineout(self, line):
        try:
            signal = self.GRAY[line, :]
        except:
            signal = np.full((self.GRAY.shape[1],), False)
        return signal

    def get_signals_for_line(self, line):
        low = self.get_lineout(line+self.STAFF_WIDTH+self.STAFF_LINE_SEP)
        mlow = self.get_lineout(line+self.STAFF_WIDTH)
        mhigh = self.get_lineout(line-self.STAFF_WIDTH)
        high = self.get_lineout(line-self.STAFF_WIDTH-self.STAFF_LINE_SEP)
        return low, mlow, mhigh, high





# Read image
file = 'test_line2.jpg'
test = MUSIC(file)


    



# %%
