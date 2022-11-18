# %%
import collections.abc #needed for pptx import
from pptx import Presentation
import sys
import os
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# PyQt5 imports
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5.Qt import Qt as qt
from PyQt5 import uic

# Figure Imports
from matplotlib.backends.backend_qt5agg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # to automatically fit long axis ticklabels



class FileManagement():
    def __init__(self, MAIN_GUI):
        self.MAIN_GUI = MAIN_GUI
        # target feature?
        self.directory = '.' # initialize to current directory
        self.featengr_csv = 'feature_engineering.csv'
        self.featengr_txt = 'feature_engineering_code.txt'
        self.featsettings_csv = 'feature_engineering_settings.csv'
        self.notes_pptx = 'slide_notes.pptx'
        return

    def create_new_feature(self, transform, feature=None, map_str=None, code=None):
        if transform=='log1p':
            self.featengr['log'+feature] = np.log1p(self.featengr[feature])
            self.featsettings.loc['log'+feature] = ['log'+feature, False, 'num']
            self.save_featengr_code( 
                feature,
                f'df["log{feature}"]=np.log1p(df["{feature}"])'
            )

        elif transform == 'bin':
            self.featengr['bin'+feature] = pd.qcut(self.featengr[feature], 10)
            self.featsettings.loc['bin'+feature] = ['bin'+feature, False, 'cat']
            self.save_featengr_code( 
                feature,
                f'df["bin{feature}"]=pd.qcut(df["{feature}"], 10)'
            )

        elif transform == 'dummies':
            self.featengr = pd.concat([ 
                self.featengr, 
                pd.get_dummies(self.featengr[feature], prefix=feature)], axis=1
            )
            self.save_featengr_code( 
                feature,
                f'df=pd.concat([df, pd.get_dummies(df["{feature}"], prefix="{feature}")], axis=1)' 
            )
            for cat in self.featengr[feature].unique():
                if cat == 'Null':
                    self.featengr = self.featengr.drop(columns=[feature+'_Null'])
                    self.save_featengr_code( 
                        feature, 
                        f'df=df.drop(columns=["{feature}_Null"])'
                    )
                else:
                    self.featsettings.loc[f'{feature}_{cat}'] = [f'{feature}_{cat}', False, 'pnum']

        elif transform == 'drop':
            self.featsettings.loc[feature, 'isdropped'] = True

        elif transform == 'undrop':
            self.featsettings.loc[feature, 'isdropped'] = False

        elif transform == 'encode':
            try:
                encode_map = {}
                for entry in map_str.split(','):
                    key, value = entry.split('=')
                    if key.strip() not in self.featengr[feature].values:
                        raise ValueError(f'Key "{key}" is invalid for given feature')
                    encode_map[key.strip()] = float(value.strip())
            except Exception as e:
                self.MAIN_GUI.DATA_GUI.raise_error('Invalid Map String', f'{e}')
                return 1

            try:
                self.featengr['enc'+feature] = self.featengr[feature].map(encode_map).astype('float')
            except Exception as e:
                self.MAIN_GUI.DATA_GUI.raise_error('Encoding Error', f'{e}')
                return 1

            self.save_featengr_code( 
                feature, 
                f'df["enc_{feature}"] = df["{feature}"].map({encode_map}).astype("float")'
            )
            self.featsettings.loc['enc_'+feature] = ['enc_'+feature, False, 'pnum']

        elif transform == 'code':
            df = self.featengr
            old_features = df.columns
            try:
                exec(code)
            except Exception as e:
                self.MAIN_GUI.DATA_GUI.raise_error('Code Error', f'{e}')
                return 1
            self.save_featengr_code(None, code)

            # Make parsings feature type a function?
            for feature in (set(df.columns)-set(old_features)):
                self.featsettings.loc[feature] = [feature, False, 'none']
                if df[feature].dtype == 'object':
                    df[feature] = df[feature].fillna('Null')
                    df[feature] = pd.Categorical(df[feature], np.sort(df[feature].unique()))
                    self.featsettings.loc[feature, 'feature_type'] = 'cat'
                else:
                    if df[feature].nunique()<11: #TODO make 11 and attribute? Used multiple locations
                        self.featsettings.loc[feature, 'feature_type'] = 'pnum'
                    else:
                        self.featsettings.loc[feature, 'feature_type'] = 'num'

        
        self.MAIN_GUI.DATA_GUI.update_menus()
        return

    def save_featengr_code(self, feature, code):
        # TODO Check if feature is target feature...
        with open(os.path.join(self.directory, self.featengr_txt), 'a') as f:
            f.write(code)
            f.write('\n')

    def get_features_by_type(self, feature_type):
        # Sort?
        feature_list = self.featsettings.name[self.featsettings.feature_type==feature_type].to_list()
        return sorted(feature_list)

    def get_features_by_isdropped(self, boolean, prefix=False):
        # Sort?
        if prefix:
            full_name = self.featsettings.feature_type + '_' + self.featsettings.name
        else:
            full_name = self.featsettings.name
        feature_list = full_name[self.featsettings.isdropped==boolean].to_list()
        return sorted(feature_list)

    def create_featengr_files(self):
        # Read and differentiate train data
        data = pd.read_csv(os.path.join(self.directory, 'train.csv'))
        obj_features = data.select_dtypes('object').columns
        num_features = data.select_dtypes('number').columns
        is_pnum = data[num_features].nunique().lt(11)

        # Convert to object dtypes to Categoricals
        data[obj_features] = data[obj_features].fillna('Null')
        for feature in obj_features.to_series():
            data[feature] = pd.Categorical(data[feature], np.sort(data[feature].unique()))
        
        # Build featsettings
        settings = pd.DataFrame(index=data.columns)
        settings['name'] = data.columns
        settings['isdropped'] = False
        settings.loc[obj_features, 'feature_type'] = 'cat'
        settings.loc[num_features[is_pnum], 'feature_type'] = 'pnum'
        settings.loc[num_features[~is_pnum], 'feature_type'] = 'num'

        # Save to file
        data.to_csv(os.path.join(self.directory, self.featengr_csv), index=False)
        settings.to_csv(os.path.join(self.directory, self.featsettings_csv))
        return

    def load_featengr_files(self):
        # Read or create featengr
        featengr_path = os.path.join(self.directory, self.featengr_csv)
        featsettings_path = os.path.join(self.directory, self.featsettings_csv)

        if not os.path.exists(featengr_path):
            self.create_featengr_files()

        self.featengr = pd.read_csv(featengr_path)
        self.featsettings = pd.read_csv(featsettings_path, index_col=0)
        return

    def select_directory(self):
        self.directory = str(qtw.QFileDialog.getExistingDirectory(self.MAIN_GUI, 'Select Directory'))
        self.MAIN_GUI.directory_label.setText(os.path.relpath(self.directory))
        self.load_featengr_files()
        
    



mw_Ui, mw_Base = uic.loadUiType('master_window.ui')
class Main_GUI(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()

        # Import Layout
        self.setupUi(self)
        self.FILES = FileManagement(self)

        # Connect signals and slots
        self.browseDirectory_button.clicked.connect(self.FILES.select_directory)
        self.featureEngineering_button.clicked.connect(self.show_data_GUI)
        # self.modelAnalysis_button.clicked.connect(call to create window)

    def show_data_GUI(self):
        if ('train.csv' not in os.listdir(self.FILES.directory)) or ('test.csv' not in os.listdir(self.FILES.directory)):
            qtw.QMessageBox.critical(self, 'Invalid Directory', 'Directory must include "test.csv" and "train.csv" files.')
            return
        self.DATA_GUI = Data_GUI(self)





mw_Ui, mw_Base = uic.loadUiType('data_analysis_gui.ui')
class Data_GUI(mw_Base, mw_Ui):
    def __init__(self, MAIN_GUI, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()

        # Import Layout
        self.setupUi(self)
        self.FILES = MAIN_GUI.FILES
        self.MAIN_GUI = MAIN_GUI
        self.update_menus()

        # Connect signals and slots
        self.log_button.clicked.connect(self.log_transform_feature)
        self.bin_button.clicked.connect(self.bin_transform_feature)
        self.create_button.clicked.connect(self.create_feature_from_code)
        self.dummies_button.clicked.connect(self.dummies_transform_feature)
        self.encode_button.clicked.connect(self.encode_transform_feature)
        self.drop_button.clicked.connect(self.drop_feature)
        self.undrop_button.clicked.connect(self.undrop_feature)

    def log_transform_feature(self):
        feature = self.log_combobox.currentText()
        if feature == 'none':
            return
        self.FILES.create_new_feature('log1p', feature=feature)
        self.log_combobox.setCurrentText('none')

    def bin_transform_feature(self):
        feature = self.bin_combobox.currentText()
        if feature == 'none':
            return
        self.FILES.create_new_feature('bin', feature=feature)
        self.bin_combobox.setCurrentText('none')

    def drop_feature(self):
        prefix, feature = self.drop_combobox.currentText().split('_')
        if feature == 'none':
            return
        self.FILES.create_new_feature('drop', feature=feature)
        self.drop_combobox.setCurrentText('none')

    def dummies_transform_feature(self):
        feature = self.dummies_combobox.currentText()
        if feature == 'none':
            return 
        self.FILES.create_new_feature('dummies', feature=feature)
        self.dummies_combobox.setCurrentText('none')

    def undrop_feature(self):
        feature = self.undrop_combobox.currentText()
        if feature == '':
            return
        self.FILES.create_new_feature('undrop', feature=feature)

    def create_feature_from_code(self):
        code = self.create_lineedit.text()
        if not self.FILES.create_new_feature('code', code=code):
            self.create_lineedit.setText('')

    def encode_transform_feature(self):
        map_str = self.encode_lineedit.text()
        feature = self.encode_combobox.currentText()
        if feature == 'none':
            return

        if not self.FILES.create_new_feature('encode', feature=feature, map_str=map_str):
            self.encode_combobox.setCurrentText('none')
            self.encode_lineedit.setText('')

    def raise_error(self, title, text):
        qtw.QMessageBox.critical(self, title, text)
        return

    def update_menus(self):
        dropped_features = self.FILES.get_features_by_isdropped(True)
        undropped_features = self.FILES.get_features_by_isdropped(False, prefix=True)

        undropped_features_plus_none = undropped_features
        undropped_features_plus_none.insert(0, 'none')

        num_features_plus_none = self.FILES.get_features_by_type('num')
        num_features_plus_none.insert(0, 'none')

        cat_features_plus_none = self.FILES.get_features_by_type('cat')
        cat_features_plus_none.insert(0, 'none')

        # Specify options for each combobox
        options_dict = { 
            'x':undropped_features_plus_none,
            'y':undropped_features_plus_none,
            'hue':undropped_features_plus_none,
            'size':undropped_features_plus_none,
            'style':undropped_features_plus_none,
            'target':undropped_features_plus_none,
            'log':num_features_plus_none,
            'bin':num_features_plus_none,
            'dummies':cat_features_plus_none,
            'encode':cat_features_plus_none,
            'drop':undropped_features_plus_none,
            'undrop':dropped_features
        }

        # Update combobox items
        for menu, options in options_dict.items():
            combobox = getattr(self, menu+'_combobox')
            current_text = combobox.currentText()
            combobox.clear()
            combobox.addItems(options)
            combobox.setCurrentText(current_text)

class GUI_Figure():
    def __init__(self, MAIN_GUI, layout):
        self.MAIN_GUI = MAIN_GUI
        self.canvas = FigureCanvas(Figure(figsize=(3,3)))
        layout.addWidget(NavigationToolbar(self.canvas, self.MAIN_GUI.DATA_GUI))
        layout.addWidget(self.canvas)
    
    def reset_figure(self, ncols):
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.subplots(ncols=ncols)
        return

    

if __name__ =='__main__':
    app = qtw.QApplication(sys.argv)
    GUI = Main_GUI()
    app.exec()
# %%
