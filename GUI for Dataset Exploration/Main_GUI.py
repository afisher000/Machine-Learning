# %%
import sys
import os
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Utils.feature_engineering import FeatureEngr
from Utils.slide_notes import Notes
from Utils.pipe_settings import Pipe

from GUIs.model_gui import ModelGUI
from GUIs.data_gui import DataGUI

# PyQt5 imports
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import Qt as qt
from PyQt5 import uic


# Figure Imports
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # to automatically fit long axis ticklabels

# To IMPLEMENT
# Add time of run and circle of death to see that it is running
# Add default clip values when only one is specified (min or max are default)
# Replace 'none' with '' in menus
# Simplify update_hyperparam_widgets, have input that specifies single/multimode


mw_Ui, mw_Base = uic.loadUiType('UIs/master_window.ui')
class MainGUI(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        # Initialization
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.show()

        # Attributes
        self.featengr = FeatureEngr(self)
        self.notes = Notes(self)
        self.pipe = Pipe(self)
        
        self.directory = '.' # initialize to current directory
        
        # Connect signals and slots
        self.browseDirectory_button.clicked.connect(self.select_directory)
        self.featureEngineering_button.clicked.connect(self.show_data_gui)
        self.modelAnalysis_button.clicked.connect(self.show_model_gui)

    def select_directory(self):
        ''' Select directory of dataset. Must contain a "train.csv" and "test.csv" '''
        TESTING = True
        if TESTING:
            directory  = 'Datasets/Titanic'
        else:
            directory = str(qtw.QFileDialog.getExistingDirectory(self, 'Select Directory'))
            if not directory:
                return
    
            if ('train.csv' not in os.listdir(directory)) or ('test.csv' not in os.listdir(directory)):
                qtw.QMessageBox.critical(self, 'Invalid Directory', 'Directory must include "test.csv" and "train.csv" files.')
                return
        
       
        self.directory = directory
    
        # Read data
        train_df = pd.read_csv(os.path.join(directory, 'train.csv'))
        test_df = pd.read_csv(os.path.join(directory, 'test.csv'))
        
        # Infer ID (first column) and target variables
        self.id_feature = test_df.columns[0] 
        self.id_test = test_df[self.id_feature]
        self.target_feature = (set(train_df.columns) - set(test_df.columns)).pop()
        
        # Determine model type based on number of unique target values
        if train_df[self.target_feature].nunique()<10:
            self.model_type = 'classifier'
        else:
            self.model_type = 'regressor'
        
        # Load data files
        self.featengr.load_files()
        self.pipe.load_files()

        # Update gui with filepath
        self.directory_label.setText(os.path.relpath(directory))

        
    def show_data_gui(self):
        ''' Open data (feature engineering) window '''
        
        if self.directory_label.text()=='':
            qtw.QMessageBox.critical(self, 'Choose Directory', 'Directory must include "test.csv" and "train.csv" files.')
            return
        self.data_gui = DataGUI(self)
        
    def show_model_gui(self):
        # Show model GUI
        if self.directory_label.text()=='':
            qtw.QMessageBox.critical(self, 'Choose Directory', 'Directory must include "test.csv" and "train.csv" files.')
            return
        self.model_gui = ModelGUI(self)
        

if __name__ =='__main__':
    app = qtw.QApplication(sys.argv)
    gui = MainGUI()
    app.exec()

# %%
