# %%
from ClassifierAnalysis import ClassifierModels
from RegressorAnalysis import RegressorModels
from Pipelines import FeatureTransforms, Imputation, Scaling, KeepSelectedFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from GUI_Figure import GUI_Figure
import pandas as pd
import numpy as np
import os

# PyQt5 imports
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import Qt as qt
from PyQt5 import uic


mw_Ui, mw_Base = uic.loadUiType('model_analysis_gui.ui')
class ModelGUI(mw_Base, mw_Ui):
    def __init__(self, main_gui, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.show()
        
        # Import Layout
        self.setupUi(self)
        self.file_manager = main_gui.file_manager
        self.file_manager.pipe.load_files()
        self.file_manager.model_gui = self
        self.main_gui = main_gui
        self.figure = GUI_Figure(self, self.plot_layout)
        self.populate_pipeline_widgets()
        self.initialize_model()
        self.update_hyper_parameters()
        self.targettransform_combobox.addItems(['none', 'log1p', 'encoding'])


        
        # Signals and Slots
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
        y_pred = self.transform_target( 
            self.get_model().model.predict(self.X_test), invert=True
        )
        predictions = pd.DataFrame({
            self.file_manager.id_feature:self.id_test, 
            self.file_manager.target_feature:y_pred
        })
        submission_path = os.path.join(self.file_manager.directory, 'submission.csv')
        predictions.to_csv(submission_path, index=False)
        return

    def fit_model(self):
        if self.apply_pipeline():
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

        # Can not due when target is category dtype
        if self.model_type=='regressor' or self.targettransform_combobox.currentText()=='encoding':
            residues = self.get_model().model.predict(self.X) - self.transform_target(self.y)
            self.file_manager.featengr.add_residues(residues)




    def plot_learning_curve(self):
        model_str = self.get_model_string()
        cv = self.cvfold_spinbox.value()

        self.figure.reset_figure(ncols=1)
        self.model_analysis.learning_curve(model_str, samples=10, cv=cv, ax=self.figure.ax)
        self.figure.canvas.draw()
        return

    def plot_validation_curve(self):
        model_str = self.get_model_string()
        cv = self.cvfold_spinbox.value()
        param_grid = self.get_param_grid()
        if not param_grid:
            qtw.QMessageBox.critical(self, 'No Parameter Selected', 'Select a hyperparameter to plot against.')
            return
        
        param_name, param_range = list(param_grid.items())[0]

        self.figure.reset_figure(ncols=1)
        self.model_analysis.validation_curve(model_str, param_name, param_range, cv=cv, ax=self.figure.ax)
        self.figure.canvas.draw()
        return


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

    def initialize_model(self):
        train_path = os.path.join(self.file_manager.directory, 'train.csv')
        train = self.file_manager.featengr.open_as_categoricals(train_path)
        target_feature = self.file_manager.target_feature
        if train[target_feature].nunique()<10:
            self.model_type = 'classifier'
            self.model_analysis = ClassifierModels()
            scoring_options = ['accuracy','f1','neg_log_loss','precision','recall']
            self.model_analysis.scoring = scoring_options[0]
            
        else:
            self.model_type = 'regressor'
            self.model_analysis = RegressorModels()
            scoring_options = [ 
                'neg_root_mean_squared_error','neg_mean_absolute_error',
                'neg_mean_squared_log_error'
            ]
            self.model_analysis.scoring = scoring_options[0]
        self.scoring_combobox.addItems(scoring_options)
        self.model_combobox.addItems(self.model_analysis.models.keys())

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
    def apply_pipeline(self):
        # Save pipe settings, check for new features, and update pipeline widgets
        self.file_manager.pipe.save_settings()
        self.file_manager.pipe.load_files()
        self.populate_pipeline_widgets()

        # Read data, get target
        train_path = os.path.join(self.file_manager.directory, 'train.csv')
        test_path = os.path.join(self.file_manager.directory, 'test.csv')
        train = self.file_manager.featengr.open_as_categoricals(train_path)
        test = self.file_manager.featengr.open_as_categoricals(test_path)

        id_feature = self.file_manager.id_feature
        target_feature = self.file_manager.target_feature
        self.y = train[target_feature]
        self.id_test = test[id_feature]
        
        # Build transformations
        scaling_dict = self.file_manager.pipe.get_scaling_dict()
        imputation_dict = self.file_manager.pipe.get_imputation_dict()
        featengr_path = os.path.join(self.file_manager.directory, self.file_manager.featengr.featengr_file)
        selected_features = self.file_manager.pipe.get_selected_features()

        transformations = [ 
            ('Feature Engineering', FeatureTransforms(self, featengr_path)),
            ('Imputation', Imputation(imputation_dict)),
            ('Scaling', Scaling(scaling_dict)),
            ('DropFeatures', KeepSelectedFeatures(selected_features))
        ]
        
        # Optionally add polynomial features
        if self.polyfeatures_checkbox.isChecked():
            polydeg = self.polydeg_spinbox.value()
            interaction_only = self.polyinteraction_checkbox.isChecked()
            transformations.append(('PolyFeatures', PolynomialFeatures(polydeg, interaction_only=interaction_only)))

        # Apply pipeline
        pipeline = Pipeline(transformations)
        try:
            self.X = pipeline.fit_transform(train)
            self.X_test = pipeline.transform(test)
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Pipeline Error', f'Raised following error: {e}')
            return 1

        self.model_analysis.X = self.X
        self.X_test = self.X_test
        self.model_analysis.y = self.transform_target(self.y)
        return 0

    def transform_target(self, y, invert=False, allow_encoding=False):
        transform = self.targettransform_combobox.currentText()
        if not invert:
            if transform == 'none':
                return y
            elif transform == 'log1p':
                return np.log1p(y)
            elif transform == 'encoding':
                map_str = self.encode_lineedit.text()
                try:
                    encoding_dict = self.file_manager.featengr.get_encoding_dict(map_str)
                except Exception as e:
                    qtw.QMessageBox.critical(self, 'Encoding Error', f'Raised following error: {e}\nApplying no transform')
                    return y
                return  pd.Series(y).map(encoding_dict).astype('float')
                    
        else:
            if transform == 'none':
                return y
            elif transform == 'log1p':
                return np.exp(y)-1
            elif transform == 'encoding':
                map_str = self.encode_lineedit.text()
                try:
                    encoding_dict = self.file_manager.featengr.get_encoding_dict(map_str)
                except Exception as e:
                    qtw.QMessageBox.critical(self, 'Encoding Error', f'Raised following error: {e}\nApplying no transform')
                    return y
                inverted_encoding_dict = {v:k for k,v in encoding_dict.items()}
                return pd.Series(y).map(inverted_encoding_dict).astype('float')

        return



    def populate_pipeline_widgets(self):
        # Clear grid layout
        grid = self.pipeline.layout()
        for i in reversed(range(grid.count())): 
            grid.itemAt(i).widget().setParent(None)

        
        # Get discrete features for bycategory menu
        discrete_features_plus_none = self.file_manager.featengr.get_discrete_features()
        discrete_features_plus_none.insert(0, 'none')

        
        # Create title row
        titles = ['Feature','Scaling','Imputing','Category','Value']
        for icol, title in enumerate(titles):
            grid.addWidget(qtw.QLabel(title, self), 0, icol)

        # Populate by row
        for irow, (feature, row) in enumerate(self.file_manager.pipe.settings.iterrows()):

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
            self.file_manager.pipe.widget_ptrs.loc[feature] = [box, scaling_menu, imputing_menu, imputeby_menu, fillna_input]

            # Populate row
            grid.addWidget(box, irow+1, 0)
            grid.addWidget(scaling_menu, irow+1, 1)
            grid.addWidget(imputing_menu, irow+1, 2)
            grid.addWidget(imputeby_menu, irow+1, 3)
            grid.addWidget(fillna_input, irow+1, 4)

            
        grid.setRowStretch(grid.rowCount(), 1)
        self.file_manager.pipe.update_widgets()
        return  
# %%
