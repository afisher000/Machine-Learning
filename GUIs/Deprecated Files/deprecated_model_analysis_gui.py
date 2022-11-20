# %%  -*- coding: utf-8 -*-

"""
Created on Mon Oct 24 08:40:32 2022

@author: afisher
"""
import sys
sys.path.append('../Datasets')
from ClassifierAnalysis import ClassifierModels
from RegressorAnalysis import RegressorModels
from Pipelines import Imputation, Scaling

import collections.abc #needed for pptx import
from pptx import Presentation
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

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

## To Implement
# Error will occur if saleprice used in feature engineering .txt
# Add error handling to output



# Feature Engineering
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, mw):
        self.main_window = mw
        self.txt_file = mw.txt_file

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        with open(self.txt_file,'r') as f:
            code = f.read()
        try:
            exec(code)
        except Exception as e:
            qtw.QMessageBox.critical(self.main_window, 'Feature Engineering Error', f'Raised following error: {e}')
        return df

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, pipe_df):
        self.pipe_df = pipe_df

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        selected_features = self.pipe_df.index[self.pipe_df.feature_status]
        df  = df[selected_features]
        return df


mw_Ui, mw_Base = uic.loadUiType('model_analysis_gui.ui')
class Main_Window(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()

        # Import data and setup layout
        self.txt_file = None
        self.pipe_file = None
        self.setupUi(self)
        self.initialize_figure()

        # Connect signals and slots
        quit = qtw.QAction('Quit', self)
        quit.triggered.connect(self.closeEvent)
        self.browse_button.clicked.connect(self.browse_file)
        self.applypipeline_button.clicked.connect(self.apply_pipeline)
        self.fitmodel_button.clicked.connect(self.fit_model)
        self.scoring_combobox.currentTextChanged.connect(self.set_model_scoring)
        self.bagging_checkbox.stateChanged.connect(self.toggle_bagging)
        self.boosting_checkbox.stateChanged.connect(self.toggle_boosting)
        self.learningcurve_button.clicked.connect(self.plot_learning_curve)
        self.model_combobox.currentTextChanged.connect(self.update_hyper_parameters)
        self.submission_button.clicked.connect(self.create_submission)
        self.validation_button.clicked.connect(self.plot_validation_curve)


    def create_submission(self):
        model = self.get_model().model

        # Convert to actual target 
        target_transform = self.targettransform_combobox.currentText()
        if target_transform == 'log1p':
            y_pred = np.exp(model.predict(self.X_test))-1
        else:
            y_pred = model.predict(self.X_test)

        predictions = pd.DataFrame({self.ID_column:self.pred_ID, self.target_feature:y_pred})
        predictions.to_csv(os.path.join(self.directory, 'submission.csv'), index=False)
        return

    def get_model_string(self):
        model_str = self.model_combobox.currentText()
        if self.bagging_checkbox.isChecked():
            model_str += '_bagging'
        if self.boosting_checkbox.isChecked():
            model_str += '_boosting'
        return model_str

    def get_model(self):
        model_str = self.get_model_string()
        if '_' in model_str:
            base_model, strategy = model_str.split('_')
            model = getattr(self.model_analysis.models[base_model], strategy)
        else:
            model = self.model_analysis.models[model_str]
        return model


    def get_param_grid(self):
        param_grid = {}
        for row in range(self.paramgrid_layout.rowCount()):
            checkbox = self.paramgrid_layout.itemAt(row, 0).widget()
            lineedit = self.paramgrid_layout.itemAt(row, 1).widget()
            if checkbox.isChecked():
                try:
                    exec(f'param_grid["{checkbox.text()}"] = {lineedit.text()}')
                except Exception as e:
                    qtw.QMessageBox.critical(self, 'Parameter Grid Error', f'Error raised: {e}')
        return param_grid

    def fit_model(self):
        self.apply_pipeline()
        self.results_label.setText('')
        param_grid = self.get_param_grid()
        model_str = self.get_model_string()
        cv = self.cvfold_spinbox.value()
        scores = self.model_analysis.gridsearch(model_str, param_grid=param_grid, cv=cv)

        # Print results
        self.results_label.setText(f'{model_str}: {scores.mean():.3f} ({scores.std():.3f})')
        opt_params = self.get_model().model.get_params()
        self.optparam_label.setText(', '.join([f'{key}={opt_params[key]}' for key in param_grid.keys()]))

    def plot_learning_curve(self):
        model_str = self.get_model_string()
        cv = self.cvfold_spinbox.value()

        self.reset_figure(ncols=1)
        self.model_analysis.learning_curve(model_str, samples=10, cv=cv, ax=self.ax)
        self.canvas.draw()
        return

    def plot_validation_curve(self):
        model_str = self.get_model_string()
        cv = self.cvfold_spinbox.value()
        param_grid = self.get_param_grid()
        if not param_grid:
            qtw.QMessageBox.critical(self, 'No Parameter Selected', 'Select a hyperparameter to plot against.')
            return
        
        param_name, param_range = list(param_grid.items())[0]

        self.reset_figure(ncols=1)
        self.model_analysis.validation_curve(model_str, param_name, param_range, cv=cv, ax=self.ax)
        self.canvas.draw()
        return

    def update_hyper_parameters(self):
        param_grid = self.get_model().param_grid

        # Clear paramgrid QFormLayout
        for row in reversed(range(self.paramgrid_layout.rowCount())):
            self.paramgrid_layout.removeRow(row)

        # Populate paramgrid QFromLayout
        for param, paramrange in param_grid.items():
            self.paramgrid_layout.addRow( 
                qtw.QCheckBox(param, self), qtw.QLineEdit(str(paramrange), self)
            )

    def toggle_bagging(self):
        if self.bagging_checkbox.isChecked() and self.boosting_checkbox.isChecked():
            self.boosting_checkbox.setChecked(False)
        self.update_hyper_parameters()

    def toggle_boosting(self):
        if self.bagging_checkbox.isChecked() and self.boosting_checkbox.isChecked():
            self.bagging_checkbox.setChecked(False)
        self.update_hyper_parameters()

    def set_model_scoring(self):
        self.model_analysis.scoring = self.scoring_combobox.currentText()
        return

    def initialize_models(self):
        # Infer target feature and model type (regressor vs classifier)
        train_df = pd.read_csv(os.path.join(self.directory, 'train.csv'))
        train_features = train_df.columns
        test_features = pd.read_csv(os.path.join(self.directory, 'test.csv')).columns
        self.target_feature = (set(train_features) - set(test_features)).pop()
        if train_df[self.target_feature].nunique()<10:
            self.model_analysis = ClassifierModels()
            scoring_options = ['accuracy','f1','neg_log_loss','precision','recall']

        else:
            self.model_analysis = RegressorModels()
            scoring_options = [ 
                'neg_root_mean_squared_error','neg_mean_absolute_error',
                'neg_mean_squared_log_error'
            ]

        self.scoring_combobox.addItems(scoring_options)
        self.model_combobox.addItems(self.model_analysis.models.keys())
 
    def save_pipe_settings(self):
        # Update settings in self.pipe_df from pointers
        for feature, row in self.pipe_df.iterrows():
            self.pipe_df.loc[feature, 'feature_status'] = self.pipe_df.loc[feature, 'checkbox_ptr'].isChecked()
            self.pipe_df.loc[feature, 'scale_strat'] = self.pipe_df.loc[feature, 'scaling_menu_ptr'].currentText()
        
            impute_strat = self.pipe_df.loc[feature, 'imputing_menu_ptr'].currentText()
            self.pipe_df.loc[feature, 'impute_strat'] = impute_strat
            if impute_strat=='bycategory':
                self.pipe_df.loc[feature,'impute_by'] = self.pipe_df.loc[feature,'imputeby_menu_ptr'].currentText()
            elif impute_strat=='byvalue':
                self.pipe_df.loc[feature,'impute_by'] = self.pipe_df.loc[feature,'fillna_input_ptr'].text()
            else:
                self.pipe_df.loc[feature,'impute_by'] = 'none'

        self.pipe_df.to_csv(self.pipe_file)

    def apply_pipeline(self):
        # Save pipe_settings to file
        self.save_pipe_settings()

        # Read in test and train data
        train_df = pd.read_csv(os.path.join(self.directory,'train.csv'))
        test_df = pd.read_csv(os.path.join(self.directory, 'test.csv'))
        self.ID_column = test_df.columns[0]
        self.pred_ID = test_df[self.ID_column]
        self.y = train_df[self.target_feature]

        scaling_data = self.get_scaling_data()
        imputation_data = self.get_imputation_data()

        transformations = [ 
            ('Feature Engineering', FeatureEngineering(self)),
            ('Imputation', Imputation(imputation_data)),
            ('Scaling', Scaling(scaling_data)),
            ('DropFeatures', DropFeatures(self.pipe_df))
        ]

        # Polynomial Features
        if self.polyfeatures_checkbox.isChecked():
            polydeg = self.polydeg_spinbox.value()
            interaction_only = self.polyinteraction_checkbox.isChecked()
            transformations.append(('PolyFeatures', PolynomialFeatures(polydeg, interaction_only=interaction_only)))

        # Build and apply pipeline
        pipeline = Pipeline(transformations)
        self.X = pipeline.fit_transform(train_df)
        self.X_test = pipeline.transform(test_df)

        self.model_analysis.X = self.X

        # Apply target transform
        target_transform = self.targettransform_combobox.currentText()
        if target_transform == 'log1p':
            self.model_analysis.y = np.log1p(self.y)
        else:
            self.model_analysis.y = self.y

        # Output Nans
        if not self.polyfeatures_checkbox.isChecked():
            nansums = mw.X_test.isna().sum() + mw.X.isna().sum()
            nan_features = nansums.index[nansums>0].to_list()
            if nan_features:
                self.output_lineedit.setText(f'Warning: Null values in {nan_features}')
            else:
                self.output_lineedit.setText('')


        return

    def get_scaling_data(self):
        # Read pipe settings for selected features
        pipe_df = pd.read_csv(self.pipe_file, index_col=0)
        pipe_df = pipe_df[pipe_df.feature_status]

        # Build dictionary for scaling implementation
        data = {}
        for feature, strat in pipe_df.scale_strat.items():
            if strat == 'none':
                continue
            if strat in data:
                data[strat].append(feature)
            else:
                data[strat] = [feature]
        return data

    def get_imputation_data(self):
        # Read pipe settings for selected features
        pipe_df = pd.read_csv(self.pipe_file, index_col=0)
        pipe_df = pipe_df[pipe_df.feature_status]

        # Build dictionary for imputation implementation
        data = {}
        for feature, strat in pipe_df.impute_strat.items():
            if strat == 'none':
                continue
            
            # Entry in dictinoary depends on strategy
            if strat in ['bycategory', 'byvalue']:
                by_value = pipe_df.impute_by[feature]
                entry = (feature, by_value)
            else:
                entry = feature
                
            if strat in data:
                data[strat].append(entry)
            else:
                data[strat] = [entry]
        return data


    def load_pipe_settings(self, numeric_features):
        if os.path.exists(self.pipe_file):
            pipe_df = pd.read_csv(self.pipe_file, index_col = 0)
        else:
            pipe_df = pd.DataFrame( 
                index=np.sort(numeric_features), 
                columns=[ 
                    'checkbox_ptr',
                    'scaling_menu_ptr',
                    'imputing_menu_ptr',
                    'imputeby_menu_ptr',
                    'fillna_input_ptr',
                    'feature_status', 
                    'scale_strat', 
                    'impute_strat', 
                    'impute_by'
                ]
            )

            # Remove target
            if self.target_feature in pipe_df.index:
                pipe_df = pipe_df.drop(self.target_feature, axis=0)

            # Initialize columns
            pipe_df['feature_status'] = False
            pipe_df['scale_strat'] = 'none'
            pipe_df['impute_strat'] = 'none'
            pipe_df['impute_by'] = 'none'
        return pipe_df

    def browse_file(self):
        # Read data from new file, return if error
        self.abs_filename, _ = qtw.QFileDialog.getOpenFileName()
        if self.abs_filename=='':
            return
            
        self.directory = os.path.dirname(self.abs_filename)
        self.txt_file = self.abs_filename.removesuffix('.csv') + '_feature_engineering.txt'
        self.pipe_file = self.abs_filename.removesuffix('.csv') + '_pipe_settings.csv'

        self.data_df = pd.read_csv(self.abs_filename)
        numeric_features = self.data_df.select_dtypes('number') #only numeric
        discrete_features = self.data_df.columns[self.data_df.nunique()<0.05*len(self.data_df)]

        # Initialize Models
        self.initialize_models()
        self.update_hyper_parameters()

        self.pipe_df = self.load_pipe_settings(numeric_features)
        self.populate_pipeline_widgets(discrete_features)

        # Setup rest of form
        self.targettransform_combobox.addItems(['none', 'log1p'])



    def populate_pipeline_widgets(self, discrete_features):
        grid = self.pipeline.layout()

        # Get discrete features for bycategory menu
        discrete_features_plus_none = discrete_features.to_list()
        discrete_features_plus_none.insert(0, 'none')

        # Create title row
        titles = ['Feature','Scaling','Imputing','Category','Value']
        for icol, title in enumerate(titles):
            grid.addWidget(qtw.QLabel(title, self), 0, icol)

        # Populate by row
        for irow, (feature, row) in enumerate(self.pipe_df.iterrows()):

            # Create widgets
            box = qtw.QCheckBox(feature, self)
            scaling_menu = qtw.QComboBox(self)
            scaling_menu.addItems(['none','minmax','standard','quartile'])
            imputing_menu = qtw.QComboBox(self)
            imputing_menu.addItems(['none','mean','median','most_frequent','bycategory','byvalue'])
            imputeby_menu = qtw.QComboBox(self)
            imputeby_menu.addItems(discrete_features_plus_none) 
            fillna_input = qtw.QLineEdit(self)

            # Link pointers in table
            ptrs = [ 
                'checkbox_ptr',
                'scaling_menu_ptr',
                'imputing_menu_ptr',
                'imputeby_menu_ptr',
                'fillna_input_ptr'
            ]
            self.pipe_df.loc[feature, ptrs] = box, scaling_menu, imputing_menu, imputeby_menu, fillna_input

            # Populate row
            grid.addWidget(box, irow+1, 0)
            grid.addWidget(scaling_menu, irow+1, 1)
            grid.addWidget(imputing_menu, irow+1, 2)
            grid.addWidget(imputeby_menu, irow+1, 3)
            grid.addWidget(fillna_input, irow+1, 4)

            # Set current values
            box.setChecked(row.feature_status)
            scaling_menu.setCurrentText(row.scale_strat)
            imputing_menu.setCurrentText(row.impute_strat)
            if row.impute_strat=='bycategory':
                imputeby_menu.setCurrentText(row.impute_by)
            elif row.impute_strat=='byvalue':
                fillna_input.setText(row.impute_by)

        grid.setRowStretch(grid.rowCount(), 1)
        return        

    def reset_figure(self, ncols):
        self.canvas.figure.clear()
        self.ax = self.canvas.figure.subplots(ncols=ncols)
        return

    def initialize_figure(self):
        # Create matplotlib figure
        self.canvas = FigureCanvas(Figure(figsize=(3,3)))
        self.plot_layout.addWidget(NavigationToolbar(self.canvas, self))
        self.plot_layout.addWidget(self.canvas)
        self.reset_figure(ncols=1)

    def closeEvent(self, event):
        self.pipe_df.to_csv(self.pipe_file)
        event.accept()
        return

if __name__ =='__main__':
    app = qtw.QApplication(sys.argv)
    mw = Main_Window()
    app.exec()


# %%
