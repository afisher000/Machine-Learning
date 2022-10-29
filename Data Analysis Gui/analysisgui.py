# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 08:40:32 2022

@author: afisher
"""

from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5.Qt import Qt as qt
from PyQt5 import uic


mw_Ui, mw_Base = uic.loadUiType('inspection.ui')
class Main_Window(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Import Layout
        self.setupUi(self)
        
        # Connect signals and slots
        pass