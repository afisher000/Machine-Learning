# %%
import sys
import os
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from FeatureEngr import FeatureEngr
from SlideNotes import Notes
from PipeSettings import Pipe

# PyQt5 imports
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import Qt as qt
from PyQt5 import uic
from ModelGUI import ModelGUI
from DataGUI import DataGUI


# Figure Imports
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # to automatically fit long axis ticklabels

# To IMPLEMENT
# Make so you can only use one window at a time
# Can you start working with the pipeline window?
# Sort features by name in pipeline window
# Don't write feature engineering with target feature to txt file.
# Add time of run and circle to death to see that it is running
# How to implement bagging with different hyperparaters models?
# Put error if trying to use dummies on categories with too many entries
# Add "current_model" to model_analysis classes
# Soft voting by saving predictions to a dataframe, when fitting on the data frame...

mw_Ui, mw_Base = uic.loadUiType('master_window.ui')
class MainGUI(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()



        # Import Layout
        self.setupUi(self)
        self.featengr = FeatureEngr(self)
        self.notes = Notes(self)
        self.pipe = Pipe(self)
        self.directory = '.' # initialize to current directory
        

        # Connect signals and slots
        self.browseDirectory_button.clicked.connect(self.select_directory)
        self.featureEngineering_button.clicked.connect(self.show_data_gui)
        self.modelAnalysis_button.clicked.connect(self.show_model_gui)

    def select_directory(self):
        # directory = str(qtw.QFileDialog.getExistingDirectory(self, 'Select Directory'))
        # if not directory:
        #     return

        # if ('train.csv' not in os.listdir(directory)) or ('test.csv' not in os.listdir(directory)):
        #     qtw.QMessageBox.critical(self, 'Invalid Directory', 'Directory must include "test.csv" and "train.csv" files.')
        #     return
        
        # self.directory = directory
        self.directory = 'C:/Users/afish/Documents/GitHub/Machine-Learning/Datasets/Titanic'
    
        # Get target and ID features
        train_df = pd.read_csv(os.path.join(directory, 'train.csv'))
        test_df = pd.read_csv(os.path.join(directory, 'test.csv'))
        self.id_feature = test_df.columns[0] 
        self.id_test = test_df[self.id_feature]
        self.target_feature = (set(train_df.columns) - set(test_df.columns)).pop()
        if train_df[self.target_feature].nunique()<10:
            self.model_type = 'classifier'
        else:
            self.model_type = 'regressor'
        
        # Load data files
        self.featengr.load_files()
        self.pipe.load_files()

        self.directory_label.setText(os.path.relpath(directory))

        
    def show_data_gui(self):
        if self.directory_label.text()=='':
            qtw.QMessageBox.critical(self, 'Choose Directory', 'Directory must include "test.csv" and "train.csv" files.')
            return
        self.data_gui = DataGUI(self)
        
    def show_model_gui(self):
        if self.directory_label.text()=='':
            qtw.QMessageBox.critical(self, 'Choose Directory', 'Directory must include "test.csv" and "train.csv" files.')
            return
        self.model_gui = ModelGUI(self)

    

if __name__ =='__main__':
    app = qtw.QApplication(sys.argv)
    gui = MainGUI()
    app.exec()
# %%
