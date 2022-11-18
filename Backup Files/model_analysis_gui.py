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
# !!!!have grid resembling the dataframe pipe_settings. You can visually see everything immediately!
# Error will occur if saleprice used in feature engineering .txt
# Add error handling to output


# Feature Engineering
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, mw):
        self.main_window = mw
        self.txt_file = mw.txt_file
        pipe_df = pd.read_csv(mw.pipe_file, index_col=0)
        self.valid_features = pipe_df.index[pipe_df.checkbox_status].to_list()

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        with open(self.txt_file,'r') as f:
            code = f.read()
        try:
            exec(code)
        except Exception as e:
            qtw.QMessageBox.critical(self.main_window, 'Feature Engineering Error', f'Raised following error: {e}')

        df = df[self.valid_features]
        return df



mw_Ui, mw_Base = uic.loadUiType('model_analysis_gui.ui')
class Main_Window(mw_Base, mw_Ui):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()

        # Import data and setup layout
        self.valid_features = None  # Contain selected features
        self.txt_file = None
        self.pipe_file = None
        self.setupUi(self)
        self.initialize_figure()

        # Connect signals and slots
        quit = qtw.QAction('Quit', self)
        quit.triggered.connect(self.closeEvent)
        self.browse_button.clicked.connect(self.browse_file)
        self.update_features_button.clicked.connect(self.update_features)
        self.minmax_button.clicked.connect(self.add_to_minmax)
        self.standard_button.clicked.connect(self.add_to_standard)
        self.quartile_button.clicked.connect(self.add_to_quartile)
        self.noscaling_button.clicked.connect(self.remove_scaling)
        self.scalingsettings_button.clicked.connect(self.scaling_settings)
        self.mean_button.clicked.connect(self.add_to_mean)
        self.median_button.clicked.connect(self.add_to_median)
        self.mode_button.clicked.connect(self.add_to_mode)
        self.bycategory_button.clicked.connect(self.add_to_bycategory)
        self.byvalue_button.clicked.connect(self.add_to_byvalue)
        self.noimputing_button.clicked.connect(self.remove_imputation)
        self.imputesettings_button.clicked.connect(self.imputation_settings)
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
        if self.model_analysis.X is None:
            qtw.QMessageBox.critical(self, 'No Training Data', 'Run the pipeline first.')
            return

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
 

    def apply_pipeline(self, target):
        # Save pipe_settings to file
        self.pipe_df.to_csv(self.pipe_file)

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
            ('Scaling', Scaling(scaling_data))
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
        pipe_df = pd.read_csv(self.pipe_file, index_col=0)

        data = {}
        for feature, strat in pipe_df.loc[self.valid_features].scale_strat.items():
            if strat == 'none':
                continue
            if strat in data:
                data[strat].append(feature)
            else:
                data[strat] = [feature]
        return data

    def get_imputation_data(self):
        pipe_df = pd.read_csv(self.pipe_file, index_col=0)
        data = {}
        for feature, strat in pipe_df.loc[self.valid_features].impute_strat.items():
            if strat == 'none':
                continue
            
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

    def add_to_mean(self):
        feature = self.mean_combobox.currentText()
        if feature == 'none':
            return
        self.pipe_df.loc[feature, 'impute_strat'] = 'mean'
        self.update_menus()
        self.mean_combobox.setCurrentText('none')

    def add_to_median(self):
        feature = self.median_combobox.currentText()
        if feature == 'none':
            return
        self.pipe_df.loc[feature, 'impute_strat'] = 'median'
        self.update_menus()
        self.median_combobox.setCurrentText('none')


    def add_to_mode(self):
        feature = self.mode_combobox.currentText()
        if feature == 'none':
            return
        self.pipe_df.loc[feature, 'impute_strat'] = 'most_frequent'
        self.update_menus()
        self.mode_combobox.setCurrentText('none')

    def add_to_bycategory(self):
        feature = self.bycategory_combobox.currentText()
        if feature == 'none':
            return

        cat = self.bycategory_category_combobox.currentText()
        if cat == 'none':
            qtw.QMessageBox.critical(self, 'No Category selected', 'First select a category to impute by')
        
        self.pipe_df.loc[feature, 'impute_strat'] = 'bycategory'
        self.pipe_df.loc[feature, 'impute_by'] = cat
        self.update_menus()
        self.bycategory_combobox.setCurrentText('none')
        self.bycategory_category_combobox.setCurrentText('none')

    def add_to_byvalue(self):
        feature = self.byvalue_combobox.currentText()
        if feature == 'none':
            return

        value = self.byvalue_lineedit.text()
        if value == '':
            qtw.QMessageBox.critical(self, 'No Value Entered', 'First enter a value to impute with.')

        try:
            float(value)
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Code Error', 'Unable to cast input as float')

        self.pipe_df.loc[feature, 'impute_strat'] = 'byvalue'
        self.pipe_df.loc[feature, 'impute_by'] = value
        self.update_menus()
        self.byvalue_combobox.setCurrentText('none')
        self.byvalue_lineedit.setText('')

    def remove_imputation(self):
        feature = self.noimpute_combobox.currentText()
        if feature == 'none':
            return
        self.pipe_df.loc[feature, 'impute_strat'] = 'none'
        self.pipe_df.loc[feature, 'impute_by'] = 'none'
        self.update_menus()
        self.noimpute_combobox.setCurrentText('none')
        

    def imputation_settings(self):
        impute_strats = self.pipe_df.impute_strat[self.pipe_df.impute_strat!='none']
        message = ''
        for strat in np.sort(impute_strats.unique()):
            message += f'{strat.upper()}:\n'
            for feature in impute_strats.index[impute_strats==strat]:
                if feature in self.valid_features:
                    impute_by = self.pipe_df.loc[feature, 'impute_by']
                    if impute_by=='none':
                        message += feature+'\n'
                    else:
                        message += feature + f' by {impute_by}\n'

        qtw.QMessageBox.information(self, 'Scaling Settings', message)

    def scaling_settings(self):
        scale_strats = self.pipe_df.scale_strat[self.pipe_df.scale_strat!='none']
        message = ''
        for strat in np.sort(scale_strats.unique()):
            message += f'{strat.upper()}:\n'
            for feature in scale_strats.index[scale_strats==strat]:
                if feature in self.valid_features:
                    message += feature+'\n'

        qtw.QMessageBox.information(self, 'Scaling Settings', message)


    def add_to_minmax(self):
        feature = self.minmax_combobox.currentText()
        if feature=='none':
            return
        self.pipe_df.loc[feature, 'scale_strat'] = 'minmax'
        self.update_menus()
        self.minmax_combobox.setCurrentText('none')

    def add_to_standard(self):
        feature = self.standard_combobox.currentText()
        if feature=='none':
            return
        self.pipe_df.loc[feature, 'scale_strat'] = 'standard'
        self.update_menus()
        self.standard_combobox.setCurrentText('none')

    def add_to_quartile(self):
        feature = self.quartile_combobox.currentText()
        if feature=='none':
            return
        self.pipe_df.loc[feature, 'scale_strat'] = 'quartile'
        self.update_menus()
        self.quartile_combobox.setCurrentText('none')

    def remove_scaling(self):
        feature = self.noscaling_combobox.currentText()
        if feature=='none':
            return
        self.pipe_df.loc[feature, 'scale_strat'] = 'none'
        self.update_menus()
        self.noscaling_combobox.setCurrentText('none')

    def initialize_target_transform_menu(self):
        target_transforms = ['none', 'log1p']
        self.targettransform_combobox.addItems(target_transforms)
        self.targettransform_combobox.setCurrentText('none')
        return

    def browse_file(self):
        # Read data from new file, return if error
        abs_filename, _ = qtw.QFileDialog.getOpenFileName()
        if not abs_filename:
            return

        self.directory = os.path.dirname(abs_filename)
        self.abs_filename = abs_filename
        self.txt_file = self.abs_filename.removesuffix('.csv') + '_feature_engineering.txt'
        self.pipe_file = self.abs_filename.removesuffix('.csv') + '_pipe_settings.csv'
        self.data_df = pd.read_csv(abs_filename).select_dtypes('number') #only numeric

        # Initialize Models
        self.initialize_models()
        self.initialize_target_transform_menu()
        self.update_hyper_parameters()

        if os.path.exists(self.pipe_file):
            self.pipe_df = pd.read_csv(self.pipe_file, index_col = 0)
        else:
            self.pipe_df = pd.DataFrame( 
                index=np.sort(self.data_df.columns), 
                columns=['checkbox_ptr', 'checkbox_status', 'scale_strat', 'impute_strat', 'impute_by']
            )

            # Remove target
            if self.target_feature in self.pipe_df.index:
                self.pipe_df = self.pipe_df.drop(self.target_feature, axis=0)

            # Initialize columns
            self.pipe_df['checkbox_status'] = False
            self.pipe_df['scale_strat'] = 'none'
            self.pipe_df['impute_strat'] = 'none'
            self.pipe_df['impute_by'] = 'none'

        # Produce checkboxes and check according to pipe_df
        for feature, status in self.pipe_df.checkbox_status.items():
            box = qtw.QCheckBox(feature, self)
            self.pipe_df.loc[feature, 'checkbox_ptr'] = box
            self.scrollarea.layout().addWidget(box)
            box.setChecked(status)
        self.scrollarea.layout().addStretch(1)
        self.update_menus()


    def update_features(self):
        self.pipe_df['checkbox_status'] = self.pipe_df.checkbox_ptr.apply(lambda x: x.isChecked())
        self.update_menus()

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

    def update_menus(self):

        self.valid_features = self.pipe_df.index[self.pipe_df.checkbox_status].to_list()

        valid_data_features = self.data_df[self.valid_features]
        numeric_features = valid_data_features.select_dtypes('number').columns.to_list()
        numeric_features.insert(0, 'none')

        discrete_features = valid_data_features.columns[valid_data_features.nunique()<15].to_list()
        discrete_features.insert(0, 'none')

        scaled_features = self.pipe_df.index[self.pipe_df.scale_strat != 'none'].to_list()
        scaled_features.insert(0, 'none')

        imputed_features = self.pipe_df.index[self.pipe_df.impute_strat != 'none'].to_list()
        imputed_features.insert(0, 'none')

        valid_features_plus_none = self.valid_features.copy()
        valid_features_plus_none.insert(0, 'none')

        
        # Option dictionary
        options_dict = { 
            'minmax':valid_features_plus_none,
            'standard':valid_features_plus_none,
            'quartile':valid_features_plus_none,
            'noscaling':scaled_features,
            'mean':numeric_features,
            'median':numeric_features,
            'mode':discrete_features,
            'bycategory': numeric_features,
            'byvalue': numeric_features,
            'bycategory_category': discrete_features,
            'noimputing':imputed_features,
        }

        for menu, options in options_dict.items():
            combobox = getattr(self, menu+'_combobox')
            current_text = combobox.currentText()
            if current_text not in options:
                current_text = 'none'
            combobox.clear()
            combobox.addItems(options)
            combobox.setCurrentText(current_text)

    def closeEvent(self, event):
        self.pipe_df.to_csv(self.pipe_file)
        event.accept()
        return

if __name__ =='__main__':
    app = qtw.QApplication(sys.argv)
    mw = Main_Window()
    app.exec()


# %%
