# %%  -*- coding: utf-8 -*-

"""
Created on Mon Oct 24 08:40:32 2022

@author: afisher
"""

import collections.abc #needed for pptx import
from pptx import Presentation
import sys
import os
import pandas as pd
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

## To implement
# Filtering options (only data from a given category or numeric range)
# Create one-hot and dummy encoders
# Create encodings through gui

mw_Ui, mw_Base = uic.loadUiType('data_analysis_gui.ui')
class Main_Window(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()

        # Initialize attributes
        self.file_directory = None

        # Import data and setup layout
        self.setupUi(self)
        self.initialize_figure()

        # Connect signals and slots
        quit = qtw.QAction('Quit', self)
        quit.triggered.connect(self.closeEvent)
        self.browse_button.clicked.connect(self.browse_file)
        self.update_button.clicked.connect(self.update_figure)
        self.drop_button.clicked.connect(self.drop_feature)
        self.undrop_button.clicked.connect(self.undrop_feature)
        self.corr_button.clicked.connect(self.plot_correlations)
        self.nullpct_button.clicked.connect(self.plot_nullpct)
        self.log_button.clicked.connect(self.log_transform_feature)
        self.bin_button.clicked.connect(self.bin_feature)
        self.saveas_button.clicked.connect(self.save_as)
        self.save_button.clicked.connect(self.save)
        self.notes_button.clicked.connect(self.save_notes_to_pptx)
        self.create_button.clicked.connect(self.create_feature_from_code)

        # Reset estimator menu
        aggregate_fcn_options = {' none':None, ' max':np.max, ' min':np.min, ' mean':np.mean, ' median':np.median, ' sum':np.sum}
        for label, fcn in aggregate_fcn_options.items():
            self.aggregate_combobox.addItem(label, fcn)

    def load_pptx(self):
        # Open relevant slideshow
        filename = self.save_lineedit.text().removesuffix('.csv') + '_slides.pptx'
        if os.path.exists(filename):
            ppt = Presentation(filename)
        else:
            ppt = Presentation()
            features_slide = ppt.slides.add_slide(ppt.slide_layouts[1])
            features_slide.shapes[0].text_frame.paragraphs[0].text = 'Custom features'
        return ppt, filename

    def save_notes_to_pptx(self):
        # Read inputs
        tempfile = 'temp.png'
        title = self.title_lineedit.text()
        text = self.text_textedit.toPlainText()
        self.canvas.print_figure(tempfile, bbox_inches='tight')


        # Add slide with text and picture
        ppt, fullpath = self.load_pptx()
        slide = ppt.slides.add_slide(ppt.slide_layouts[8])
        slide.shapes[0].text_frame.paragraphs[0].text = ' ' if title=='' else title
        pic = slide.placeholders[1].insert_picture(tempfile)
        pic.crop_top, pic.crop_left, pic.crop_bottom, pic.crop_right = 0,0,0,0
        slide.shapes[2].text_frame.paragraphs[0].text = text
        ppt.save(fullpath)
        os.remove(tempfile)

        # Clear text inputs
        self.title_lineedit.setText('')
        self.text_textedit.setText('')

        pass

    def bin_feature(self):
        feature = self.bin_combobox.currentText()
        self.df['bin'+feature] = pd.qcut(self.df[feature], 10)
        self.cat_features.loc['bin'+feature] = 'bin'+feature
        self.update_menus()

    def log_transform_feature(self):
        feature = self.log_combobox.currentText()
        self.df['log'+feature] = np.log1p(self.df[feature])
        self.num_features.loc['log'+feature] = 'log'+feature
        self.update_menus()

    def reset_figure(self, ncols):
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.subplots(ncols=ncols)
        return

    def plot_nullpct(self):
        # Nulls are strings in category features
        self.reset_figure(ncols=1)
        valid_features = self.get_valid_features()
        prefix, target = self.target_combobox.currentText().split(' ')
        (  
            (self.df[valid_features].isna().sum() + self.df.apply(lambda x: sum(x=='Null')))
            .sort_values(ascending=False).divide(len(self.df))[:self.nullpct_spinbox.value()]
            .plot.bar(xlabel='Feature', ylabel='Null Percentage', ax=self.ax)
        )
        self.canvas.draw()

    def get_valid_features(self, prefix=False):
        if prefix:
            valid_features = pd.concat([ 
                'num '+self.num_features,
                'pnum '+self.pnum_features,
                'cat '+self.cat_features
            ]).sort_values().to_list()
        else:
            valid_features = pd.concat([ 
                self.num_features,
                self.pnum_features,
                self.cat_features
            ]).sort_values().to_list()
        return valid_features

    def plot_correlations(self):
        valid_features = self.get_valid_features()
        prefix, target = self.target_combobox.currentText().split(' ')
        if target == 'none':
            qtw.QMessageBox.warning(
                self, 'No Target Specified', 'Correlations must be computed against some target feature.'
            )
            return

        self.reset_figure(ncols=1)     
        (
            self.df[valid_features].corrwith(self.df[target], numeric_only=True)
            .abs().sort_values(ascending=False)[1:self.corr_spinbox.value()+1] #skip correlation with self
            .plot.bar(xlabel='Feature', ylabel='Correlation', ax=self.ax)
        )
        self.canvas.draw()

    def update_menus(self):
        # Ensure features are sorted. 
        self.dropped_features.sort()
        self.num_features.sort_values()

        # Combine all valid features with identifying prefix
        valid_features = self.get_valid_features(prefix=True)
        valid_features.insert(0, ' none')


        # Option dictionary
        options_dict = { 
            'x':valid_features,
            'y':valid_features,
            'hue':valid_features,
            'size':valid_features,
            'style':valid_features,
            'target':valid_features,
            'log':self.num_features,
            'bin':self.num_features,
            'drop':valid_features,
            'undrop':self.dropped_features
        }

        for menu, options in options_dict.items():
            combobox = getattr(self, menu+'_combobox')
            current_text = combobox.currentText()
            combobox.clear()
            combobox.addItems(options)
            combobox.setCurrentText(current_text)

        return

    def drop_feature(self):
        # Drop feature
        prefix, feature = self.drop_combobox.currentText().split(' ')
        if feature=='none':
            return

        self.dropped_features.extend([feature])
        if prefix=='num':
            self.num_features = self.num_features.drop(feature)
        elif prefix=='pnum':
            self.pnum_features = self.pnum_features.drop(feature)
        elif prefix=='cat':
            self.cat_features = self.cat_features.drop(feature)
        self.update_menus()
        return 

    def undrop_feature(self):
        # Undrop feature
        feature = self.undrop_combobox.currentText()
        self.dropped_features.remove(feature)

        if self.df[feature].dtype=='object':
            self.cat_features.loc[feature] = feature
        else:
            if self.df[feature].nunique()<11:
                self.pnum_features.loc[feature] = feature
            else:
                self.num_features.loc[feature] = feature

        self.update_menus()
        return 

    def create_feature_from_code(self):
        df = self.df
        old_features = df.columns

        python_code = self.create_lineedit.text()
        
        try:
            exec(python_code)
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Code Error', f'Error message returned: {e}')
            return            
        
        for feature in (set(df.columns)-set(old_features)):
            if df[feature].dtype=='object':
                self.cat_features.loc[feature] = feature
                df[feature] = self.df[feature].fillna('Null') #Make null own category
                df[feature] = pd.Categorical(self.df[feature], np.sort(df[feature].unique()))
            else:
                if df[feature].nunique()<11:
                    self.pnum_features.loc[feature] = feature
                else:
                    self.num_features.loc[feature] = feature

        # Add feature engineering to slides
        ppt, fullpath = self.load_pptx()
        slide = ppt.slides[0]
        slide_text = slide.shapes[1].text_frame.paragraphs[0].text
        slide_text += python_code if slide_text=='' else ('\n' + python_code)
        slide.shapes[1].text_frame.paragraphs[0].text = slide_text
        ppt.save(fullpath)

        self.create_lineedit.setText('')
        self.update_menus()

    def save_as(self):
        if not hasattr(self, 'df'):
            qtw.QMessageBox.warning(self, 'Missing Data', 'No data available to save.')
            return

        # Encode dropped columns with '_dropped' suffix
        self.df = self.df.rename(columns={c:c+'_dropped' for c in self.dropped_features})
        filename = self.saveas_lineedit.text()

        if filename=='':
            qtw.QMessageBox.warning(self, 'No Filename', 'Add a filename to save to.')
            return

        if not filename.endswith('.csv'):
            filename += '.csv'

        full_path = os.path.join(self.file_directory, filename)
        self.df.to_csv(full_path, index=False)
        print(f'Saved as {full_path}')
        self.saveas_lineedit.setText('')
        self.save_lineedit.setText(os.path.relpath(full_path))

    def save(self):
        # Encode dropped columns with '_dropped' suffix
        self.df = self.df.rename(columns={c:c+'_dropped' for c in self.dropped_features})
        filename = self.save_lineedit.text()

        if filename=='':
            qtw.QMessageBox.warning(self, 'No Filename', 'Add a filename to save to.')
            return

        if not filename.endswith('.csv'):
            filename += '.csv'

        fullpath = os.path.join(self.file_directory, filename)
        self.df.to_csv(fullpath, index=False)
        print(f'Saved to {fullpath}')

    def browse_file(self):
        # Read data from new file, return if error
        abs_filename, _ = qtw.QFileDialog.getOpenFileName()
        if not abs_filename:
            return

        self.file_directory = os.path.dirname(abs_filename)
        self.save_lineedit.setText(os.path.relpath(abs_filename))
        self.df = pd.read_csv(abs_filename)

        # Identify dropped features by suffix and remove suffix
        self.dropped_features = [c.removesuffix('_dropped') 
            for c in self.df.columns if c.endswith('_dropped')]
        self.df = self.df.rename(columns={c:c.removesuffix('_dropped') 
            for c in self.df.columns if c.endswith('_dropped')})

        # Categorize the valid features as numerical (num), pseudonumerical (pnum), and cateogrical (cat)
        valid_features = self.df.drop(columns=self.dropped_features).columns.tolist()

        # save 'object' dtypes as categoricals
        obj_dtype_features = self.df[valid_features].select_dtypes('object').columns
        self.df[obj_dtype_features] = self.df[obj_dtype_features].fillna('Null') #Make null own category
        self.cat_features = obj_dtype_features.to_series()
        for cat_feature in self.cat_features:
            self.df[cat_feature] = pd.Categorical(self.df[cat_feature], np.sort(self.df[cat_feature].unique()))

        # numerical and pseudonumeric
        num_dtype_features = self.df.select_dtypes('number').columns
        is_pseudo = self.df[num_dtype_features].nunique().lt(11)
        self.pnum_features = num_dtype_features[is_pseudo].to_series()
        self.num_features = num_dtype_features[~is_pseudo].to_series()

        self.update_menus()
        
    def closeEvent(self, event):
        save_confirm = qtw.QMessageBox.question( 
            self, 'Save on Exit?', 'Do you want to save current feature engineering?',
            qtw.QMessageBox.Yes | qtw.QMessageBox.No
            )
        if save_confirm==qtw.QMessageBox.Yes:
            self.save()
        event.accept()

        return
    
    def initialize_figure(self):
        # Create matplotlib figure
        self.canvas = FigureCanvas(Figure(figsize=(3,3)))
        self.plot_layout.addWidget(NavigationToolbar(self.canvas, self))
        self.plot_layout.addWidget(self.canvas)

    def update_figure(self):
        df = self.df
        
        def parse_combobox_text(combobox):
            prefix, feature = combobox.currentText().split(' ')
            return None if feature=='none' else feature

        def boxen_and_pie_plot(data, x, y, hue=None, agg_fcn=None):
            self.reset_figure(ncols=2)
            if agg_fcn is None:
                sns.boxenplot(data=data, x=x, y=y, hue=hue, ax=self.ax[0])
            else:
                sns.barplot(data=data.dropna(subset=[x,y]), x=x, y=y, hue=hue, estimator=agg_fcn, ax=self.ax[0], errorbar=None)
            self.ax[0].tick_params(axis='x', rotation=90)
            df[x].value_counts(dropna=False).sort_index().plot.pie(autopct='%.0f%%', ax=self.ax[1])

        x = parse_combobox_text(self.x_combobox)
        y = parse_combobox_text(self.y_combobox)
        hue = parse_combobox_text(self.hue_combobox)
        size = parse_combobox_text(self.size_combobox)
        style = parse_combobox_text(self.style_combobox)
        agg_fcn = self.aggregate_combobox.currentData()

        # Check that x is specified
        if x is None: 
            qtw.QMessageBox.warning(self, 'No X specified', 'The X feature must be specified to plot')
            self.reset_figure(ncols=1)
            self.canvas.draw()
            return

        # X input only
        if y is None:
            if x in self.num_features:
                self.reset_figure(ncols=1)
                sns.histplot(data=df, x=x, hue=hue, multiple='stack', kde=True, ax=self.ax)
            else:
                if df[x].nunique()>30:
                    qtw.QMessageBox.warning(self, 'Axis Overflow', 'Too many unique entried to plot.')
                    self.reset_figure(ncols=1)
                    self.canvas.draw()
                    return

                self.reset_figure(ncols=2)
                sns.histplot(data=df, x=x, hue=hue, multiple='stack', kde=False, ax=self.ax[0])
                self.ax[0].tick_params(axis='x', rotation=90)
                self.df[x].value_counts(dropna=False).sort_index().plot.pie(autopct='%.0f%%', ax=self.ax[1])

        # X-Y Inputs
        if (x in self.num_features):
            if (y in self.num_features):
                self.reset_figure(ncols=1)
                sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, style=style, ax=self.ax)
                # Add regression?
            elif (y in self.pnum_features):
                x, y = y, x #Switch pnum to x axis
                boxen_and_pie_plot(data=df, x=x, y=y, hue=hue)

            elif (y in self.cat_features):
                x, y = y, x #Switch category to x axis
                # Sort according to median of numeric
                cat_ordering = df[x].cat.categories.tolist()
                median_ordering = df.groupby(by=x)[y].median().sort_values().index.tolist()
                df[x] = df[x].cat.reorder_categories(median_ordering)

                boxen_and_pie_plot(data=df, x=x, y=y, hue=hue)

                # Revert to categorical ordering
                df[x] = df[x].cat.reorder_categories(cat_ordering)

        elif (x in self.pnum_features):
            if (y in self.num_features):
                boxen_and_pie_plot(data=df, x=x, y=y, hue=hue, agg_fcn=agg_fcn)
            elif (y in self.pnum_features):
                boxen_and_pie_plot(data=df, x=x, y=y, hue=hue, agg_fcn=agg_fcn)
            elif (y in self.cat_features):
                x, y = y, x #Switch category to x axis
                # Sort according to median of numeric
                cat_ordering = df[x].cat.categories.tolist()
                median_ordering = df.groupby(by=x)[y].median().sort_values().index.tolist()
                df[x] = df[x].cat.reorder_categories(median_ordering)
                boxen_and_pie_plot(data=df, x=x, y=y, hue=hue)

                # Revert to categorical ordering
                df[x] = df[x].cat.reorder_categories(cat_ordering)
        elif (x in self.cat_features):
            if (y in self.num_features):
                # Sort according to median of numeric
                cat_ordering = df[x].cat.categories.tolist()
                median_ordering = df.groupby(by=x)[y].median().sort_values().index.tolist()
                df[x] = df[x].cat.reorder_categories(median_ordering)

                boxen_and_pie_plot(data=df, x=x, y=y, hue=hue, agg_fcn=agg_fcn)

                # Revert to categorical ordering
                df[x] = df[x].cat.reorder_categories(cat_ordering)
            elif (y in self.pnum_features):
                boxen_and_pie_plot(data=df, x=x, y=y, hue=hue, agg_fcn=agg_fcn)

            elif (y in self.cat_features):
                x_noise, y_noise = np.random.random(len(df))/2, np.random.random(len(df))/2
                self.reset_figure(ncols=2)
                sns.histplot(data=df, x=x, hue=y, stat='count', multiple='stack', ax=self.ax[0])
                self.ax[0].tick_params(axis='x', rotation=90)
                df[x].value_counts(dropna=False).sort_index().plot.pie(autopct='%.0f%%', ax=self.ax[1])

        self.canvas.draw()
        return

if __name__ =='__main__':
    app = qtw.QApplication(sys.argv)
    mw = Main_Window()
    app.exec()


# %%
