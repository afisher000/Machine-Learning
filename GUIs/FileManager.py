from FeatureEngr import FeatureEngr
from SlideNotes import Notes
from PipeSettings import Pipe
import pandas as pd
import numpy as np
import os

class FileManager():
    def __init__(self):
        self.data_gui = None
        self.model_gui = None
        self.featengr = FeatureEngr(self)
        self.notes = Notes(self)
        self.pipe = Pipe(self)
        self.directory = '.' # initialize to current directory
        return


    def load_directory(self, directory):
        self.directory = directory
        
        # Get target and ID features
        train_df = pd.read_csv(os.path.join(directory, 'train.csv'))
        test_df = pd.read_csv(os.path.join(directory, 'test.csv'))
        self.id_feature = test_df.columns[0] #Can be done better
        self.target_feature = (set(train_df.columns) - set(test_df.columns)).pop()
        
        # Load data files
        self.featengr.load_files()
        self.pipe.load_files()