# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:54:35 2022

@author: afisher
"""

class AnalyzeSong():
    def __init__(self, file, songname):
    
        
        # Define attributes
        self.blue, self.green, self.red = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.cyan, self.magenta, self.yellow = [(255,255,0), (255,0,255), (0, 255, 255)]
        self.songname = songname
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