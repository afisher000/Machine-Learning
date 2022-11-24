
from GUI_Figure import GUI_Figure
import numpy as np
import pandas as pd

# PyQt5 imports
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import Qt as qt
from PyQt5 import uic



mw_Ui, mw_Base = uic.loadUiType('data_analysis_gui.ui')
class DataGUI(mw_Base, mw_Ui):
    def __init__(self, main_gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()
        
        # Import Layout
        self.setupUi(self)
        self.main_gui = main_gui
        self.figure = GUI_Figure(self, self.plot_layout)
        self.update_menus()
        self.initialize_fixed_menus()

        # Connect signals and slots
        self.log_button.clicked.connect(self.log_transform_feature)
        self.bin_button.clicked.connect(self.bin_transform_feature)
        self.create_button.clicked.connect(self.create_feature_from_code)
        self.create_lineedit.returnPressed.connect(self.create_feature_from_code)
        self.dummies_button.clicked.connect(self.dummies_transform_feature)
        self.encode_button.clicked.connect(self.encode_transform_feature)
        self.encode_lineedit.returnPressed.connect(self.encode_transform_feature)
        self.drop_button.clicked.connect(self.drop_feature)
        self.undrop_button.clicked.connect(self.undrop_feature)
        self.notes_button.clicked.connect(self.save_notes)
        self.update_button.clicked.connect(self.update_figure)
        self.customplot_button.clicked.connect(self.custom_plot)
        self.title_lineedit.returnPressed.connect(self.change_note_focus)
        self.clip_button.clicked.connect(self.clip_transform_feature)

    def change_note_focus(self):
        self.text_textedit.setFocus()

    def initialize_fixed_menus(self):
        # Setup aggfcn_combobox
        agg_fcn_options = {'none':None, 'max':np.max, 'min':np.min, 'mean':np.mean, 'median':np.median, 'sum':np.sum}
        for label, fcn in agg_fcn_options.items():
            self.aggfcn_combobox.addItem(label, fcn)
            
        self.customplot_combobox.addItems(['none','correlation', 'null percentage', 'largest category', 'number of categories'])
        self.encodeby_combobox.addItems(['map', 'mean', 'median'])
        return

    def clip_transform_feature(self):
        feature = self.clip_combobox.currentText()
        if feature == 'none':
            return

        try:
            minval = float(self.minval_lineedit.text())
            maxval = float(self.maxval_lineedit.text())
        except:
            qtw.QMessageBox.critical(self, 'Invalid minval or maxval', 'Both inputs must be entered and numerical values')
            return

        self.main_gui.featengr.new_feature('clip', feature=feature, clip_range = [minval, maxval])
        self.minval_lineedit.setText('')
        self.maxval_lineedit.setText('')
        self.clip_combobox.setCurrentText('none')
        self.update_menus()

    def log_transform_feature(self):
        feature = self.log_combobox.currentText()
        if feature == 'none':
            return
        self.main_gui.featengr.new_feature('log1p', feature=feature)
        self.log_combobox.setCurrentText('none')
        self.update_menus()

    def bin_transform_feature(self):
        feature = self.bin_combobox.currentText()
        if feature == 'none':
            return
        self.main_gui.featengr.new_feature('bin', feature=feature)
        self.bin_combobox.setCurrentText('none')
        self.update_menus()

    def drop_feature(self):
        feature = self.parse_combobox_text(self.drop_combobox)
        if feature is None:
            return
        self.main_gui.featengr.new_feature('drop', feature=feature)
        self.drop_combobox.setCurrentText('none')
        self.update_menus()

    def dummies_transform_feature(self):
        feature = self.dummies_combobox.currentText()
        if feature == 'none':
            return 
        self.main_gui.featengr.new_feature('dummies', feature=feature)
        self.dummies_combobox.setCurrentText('none')
        self.update_menus()

    def undrop_feature(self):
        feature = self.undrop_combobox.currentText()
        if feature == '':
            return
        self.main_gui.featengr.new_feature('undrop', feature=feature)
        self.update_menus()

    def create_feature_from_code(self):
        code = self.create_lineedit.text()
        if not self.main_gui.featengr.new_feature('code', code=code):
            self.create_lineedit.setText('')
            self.update_menus()

    def encode_transform_feature(self):
        encodeby = self.encodeby_combobox.currentText()
        feature = self.encode_combobox.currentText()
        map_str = self.encode_lineedit.text()
        if feature == 'none':
            return
            


        if not self.main_gui.featengr.new_feature('encode', feature=feature, map_str=map_str, encodeby=encodeby):
            self.encode_combobox.setCurrentText('none')
            self.encode_lineedit.setText('')
            self.encodeby_combobox.setCurrentText('map')
            self.update_menus()

    def raise_error(self, title, text):
        qtw.QMessageBox.critical(self, title, text)
        return

    def update_menus(self):
        dropped_features = self.main_gui.featengr.get_features_by_isdropped(True)
        undropped_features = self.main_gui.featengr.get_features_by_isdropped(False, prefix=True)

        undropped_features_plus_none = undropped_features
        undropped_features_plus_none.insert(0, 'none')

        num_pnum_features_plus_none = self.main_gui.featengr.get_features_by_type(['num','pnum'])
        num_pnum_features_plus_none.insert(0, 'none')

        num_features_plus_none = self.main_gui.featengr.get_features_by_type('num')
        num_features_plus_none.insert(0, 'none')

        cat_features_plus_none = self.main_gui.featengr.get_features_by_type('cat')
        cat_features_plus_none.insert(0, 'none')

        # Specify options for each combobox
        options_dict = { 
            'x':undropped_features_plus_none,
            'y':undropped_features_plus_none,
            'hue':undropped_features_plus_none,
            'size':undropped_features_plus_none,
            'style':undropped_features_plus_none,
            'log':num_pnum_features_plus_none,
            'bin':num_pnum_features_plus_none,
            'dummies':cat_features_plus_none,
            'clip':num_pnum_features_plus_none,
            'encode':cat_features_plus_none,
            'drop':undropped_features_plus_none,
            'undrop':dropped_features,
            'corrtarget':num_pnum_features_plus_none
        }

        # Update combobox items
        for menu, options in options_dict.items():
            combobox = getattr(self, menu+'_combobox')
            current_text = combobox.currentText()
            combobox.clear()
            combobox.addItems(options)
            combobox.setCurrentText(current_text)

    def save_notes(self):
        ''' Write comments and figure to notes powerpoint'''
        title = self.title_lineedit.text()
        text = self.text_textedit.toPlainText()
        canvas = self.figure.canvas
        self.main_gui.notes.save_notes(title, text, canvas)
        
        # Reset GUI inputs
        self.title_lineedit.setText('')
        self.text_textedit.setText('')
        
    def parse_combobox_text(self, comboboxes):
        is_not_list = not isinstance(comboboxes, list)
        if is_not_list:
            comboboxes = [comboboxes]
        output = []
        for combobox in comboboxes:
            text = combobox.currentText()
            if text=='none':
                output.append(None)
            else:
                prefix, feature = text.split(' ')
                output.append(feature)
                
        if is_not_list:
            output = output[0]
        return output
        
    def update_figure(self):
        x, y, hue, size, style = self.parse_combobox_text( 
            [self.x_combobox, self.y_combobox, self.hue_combobox, self.size_combobox, self.style_combobox]
        )
        agg_fcn = self.aggfcn_combobox.currentData()
        
        data = self.main_gui.featengr.data
        num_features = self.main_gui.featengr.get_features_by_type('num')
        pnum_features = self.main_gui.featengr.get_features_by_type('pnum')
        cat_features = self.main_gui.featengr.get_features_by_type('cat')
        
        if x is None:
            qtw.QMessageBox.warning(self, 'No X specified', 'The X feature must be specified to plot')
            self.figure.reset_figure(ncols=1)
            self.figure.draw()
            return 
        
        if y is None:
            if x in num_features:
                self.figure.histplot(data, x, hue)
                return
            else:
                self.figure.histplot_and_piechart(data, x, hue)
                return
        
        if x in num_features:
            if y in num_features:
                self.figure.scatterplot(data, x, y, hue, size, style)
            elif y in pnum_features:
                self.figure.boxenplot_and_piechart( 
                    data, x, y, hue, agg_fcn=agg_fcn, switch_axes=True
                )
            elif y in cat_features:
                self.figure.boxenplot_and_piechart( 
                    data, x, y, hue, agg_fcn=agg_fcn, switch_axes=True, median_ordering=True
                )
        elif x in pnum_features:
            if y in num_features:
                self.figure.boxenplot_and_piechart( 
                    data, x, y, hue, agg_fcn=agg_fcn 
                )
            elif y in pnum_features:
                self.figure.boxenplot_and_piechart( 
                    data, x, y, hue, agg_fcn=agg_fcn 
                )
            elif y in cat_features:
                self.figure.boxenplot_and_piechart( 
                    data, x, y, hue, agg_fcn=agg_fcn, switch_axes=True, median_ordering=True 
                )
        elif x in cat_features:
            if y in num_features:
                self.figure.boxenplot_and_piechart( 
                    data, x, y, hue, agg_fcn=agg_fcn, median_ordering=True 
                )
            elif y in pnum_features:
                self.figure.boxenplot_and_piechart( 
                    data, x, y, hue, agg_fcn = agg_fcn
                )     
            elif y in cat_features:
                self.figure.histplot_and_piechart( 
                    data, x, hue=y, stat='count'
                )
        
    def custom_plot(self):
        plot_type = self.customplot_combobox.currentText()
        if plot_type=='none':
            return
        
        corrtarget = self.corrtarget_combobox.currentText()
        if plot_type=='correlation' and corrtarget=='none':
            qtw.QMessageBox.warning(
                self, 'Select Correlation Target', 'You must select a correlation target.'
            )
            return
        n_features = self.customplot_spinbox.value()
        data = self.main_gui.featengr.data
        undropped_features = self.main_gui.featengr.get_features_by_isdropped(False)
        self.figure.custom_plot(data[undropped_features], plot_type, corrtarget, n_features)
        return