# %%
import sys
import os
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from FileManager import FileManager

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
# Only allow one of each window to be open
# Update pipeline features from data analysis


mw_Ui, mw_Base = uic.loadUiType('master_window.ui')
class MainGUI(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()

        # Import Layout
        self.setupUi(self)
        self.file_manager = FileManager()
        

        # Connect signals and slots
        self.browseDirectory_button.clicked.connect(self.select_directory)
        self.featureEngineering_button.clicked.connect(self.show_data_gui)
        self.modelAnalysis_button.clicked.connect(self.show_model_gui)

    def select_directory(self):
        directory = str(qtw.QFileDialog.getExistingDirectory(self, 'Select Directory'))
        if ('train.csv' not in os.listdir(directory)) or ('test.csv' not in os.listdir(directory)):
            qtw.QMessageBox.critical(self, 'Invalid Directory', 'Directory must include "test.csv" and "train.csv" files.')
            return
        
        self.directory_label.setText(os.path.relpath(directory))
        self.file_manager.load_directory(directory)
        
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
