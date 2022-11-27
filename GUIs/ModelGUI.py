# %%
from ModelAnalysis import ModelAnalysis
from Pipelines import FeatureTransforms, Imputation, Scaling, KeepSelectedFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from GUI_Figure import GUI_Figure
import pandas as pd
import numpy as np
import os
import numbers

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
        self.main_gui = main_gui
        self.main_gui.pipe.load_files()
        
        self.figure = GUI_Figure(self, self.plot_layout)
        self.populate_pipeline_widgets()
        self.initialize_model()
        self.targettransform_combobox.addItems(['none', 'log1p', 'encoding'])

        # Signals and Slots
        self.applypipeline_button.clicked.connect(self.apply_pipeline)
        self.fitsinglemodel_button.clicked.connect(self.fit_singlemodel)
        self.scoring_combobox.currentTextChanged.connect(self.set_model_scoring)
        self.learningcurve_button.clicked.connect(self.plot_learning_curve)
        self.singlemodels_combobox.currentTextChanged.connect(self.update_hyperparam_widgets)
        self.multimodels_combobox.currentTextChanged.connect(self.update_hyperparam_widgets)
        self.submission_button.clicked.connect(self.create_submission)
        self.validation_button.clicked.connect(self.plot_validation_curve)
        self.savemodel_button.clicked.connect(self.save_model)
        self.modelname_lineedit.returnPressed.connect(self.save_model)
        self.dropmodel_button.clicked.connect(self.drop_selected_models)
        self.cvfold_spinbox.valueChanged.connect(self.set_model_cv)
        self.loadmodel_button.clicked.connect(self.load_model)
        self.displaymodel_button.clicked.connect(self.display_model_parameters)
        self.fitmultimodel_button.clicked.connect(self.fit_multimodel)


    def set_model_cv(self):
        self.model_analysis.cv = self.cvfold_spinbox.value()
        return
    
    def set_model_scoring(self):
        self.model_analysis.scoring = self.scoring_combobox.currentText()
        return

    def drop_selected_models(self):
        self.model_analysis.drop_selected_models()
        self.update_saved_models()

    def load_model(self):
        try:
            self.model_analysis.load_model()
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Error in Loading Model', f'Following error raised: {e}')
            return

    def display_model_parameters(self):
        try:
            title, text = self.model_analysis.display_model_parameters()
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Error in Displaying Model Parameters', f'Following error raised: {e}')
            return

        qtw.QMessageBox.information(self, title, text)

    def save_model(self):
        if self.model_analysis.get_current_model_object() is None:
            qtw.QMessageBox.critical(self, 'No Fit Model', 'You must fit a model first')
            return

        model_name = self.modelname_lineedit.text()
        if model_name=='':
            qtw.QMessageBox.critical(self, 'No Model Label', 'Type a model label.')
            return

        self.model_analysis.save_model(model_name)
        self.update_saved_models()
        self.modelname_lineedit.setText('')


    def update_saved_models(self):
        # Define and clear grid
        grid = self.savedmodel_grid
        for i in reversed(range(grid.count())): 
            grid.itemAt(i).widget().setParent(None)
        
        titles = ['Model','Score']
        irow = 0
        for icol, title in enumerate(titles):
            grid.addWidget(qtw.QLabel(title, self), irow, icol)

        for model_label, row in self.model_analysis.saved_models.iterrows():
            box = qtw.QCheckBox(model_label, self)
            self.model_analysis.saved_models.loc[model_label, 'checkbox'] = box
            label = qtw.QLabel(f'{row.model_object.mean_score:.3f}', self)
            
            irow += 1
            grid.addWidget(box, irow, 0)
            grid.addWidget(label, irow, 1)

    def create_submission(self):
        model_object = self.model_analysis.get_current_model_object()
        if model_object is None:
            qtw.QMessageBox.critical(self, 'Fit Model', 'No model has been fit')
            return

        y_pred = self.transform_target(model_object.test_predictions, invert=True)
        predictions = pd.DataFrame({
            self.main_gui.id_feature:self.main_gui.id_test, 
            self.main_gui.target_feature:y_pred
        })
        submission_path = os.path.join(self.main_gui.directory, 'submission.csv')
        predictions.to_csv(submission_path, index=False)
        return


    def fit_multimodel(self):
        # Apply pipeline
        if self.apply_pipeline():
            return

        self.multimodelresults_label.setText('')

        # Create new model object
        model_string = self.multimodels_combobox.currentText()
        self.model_analysis.create_model_object(model_string)

        param_grid = self.get_param_grid(self.multimodel_paramgrid_layout)
        self.model_analysis.gridsearch(param_grid=param_grid)

        # Print Results
        model_object = self.model_analysis.get_current_model_object()
        self.multimodelresults_label.setText(
            f'{model_string}: {model_object.mean_score:.3f} ({model_object.scores.std():.3f})'
        )
        optimized_params = model_object.estimator.get_params()
        self.multimodelparams_label.setText(', '.join([f'{key}={optimized_params[key]}' for key in param_grid.keys()]))

        # Compute residue when target type is numeric
        train_predictions = model_object.train_predictions
        if isinstance(train_predictions[0], numbers.Number):
            residues = train_predictions - self.transform_target(self.y)
            self.main_gui.featengr.add_residues(residues)


    def fit_singlemodel(self):
        # Apply pipeline
        if self.apply_pipeline():
            return

        self.singlemodelresults_label.setText('')

        # Create new model object
        model_string = self.singlemodels_combobox.currentText()
        self.model_analysis.create_model_object(model_string)

        # Run GridSearch 
        param_grid = self.get_param_grid(self.singlemodel_paramgrid_layout)
        self.model_analysis.gridsearch(param_grid=param_grid)

        # Print Results
        model_object = self.model_analysis.get_current_model_object()
        self.singlemodelresults_label.setText(
            f'{model_string}: {model_object.mean_score:.3f} ({model_object.scores.std():.3f})'
        )
        optimized_params = model_object.estimator.get_params()
        self.singlemodelparams_label.setText(', '.join([f'{key}={optimized_params[key]}' for key in param_grid.keys()]))

        # Compute residue when target type is numeric
        train_predictions = model_object.train_predictions
        if isinstance(train_predictions[0], numbers.Number):
            residues = train_predictions - self.transform_target(self.y)
            self.main_gui.featengr.add_residues(residues)


    def plot_learning_curve(self):
        self.figure.reset_figure(ncols=1)
        self.model_analysis.learning_curve(samples=10, ax=self.figure.ax)
        self.figure.canvas.draw()
        return

    def plot_validation_curve(self):
        self.figure.reset_figure(ncols=1)
        try:
            self.model_analysis.validation_curve(ax=self.figure.ax)
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Plotting Error', f'{e}')
        self.figure.canvas.draw()
        return


    def initialize_model(self):
        scoring_options_dict = { 
            'classifier':['accuracy','f1','neg_log_loss','precision','recall'],
            'regressor':[ 
                'neg_root_mean_squared_error','neg_mean_absolute_error',
                'neg_mean_squared_log_error'
            ]
        }

        self.model_analysis = ModelAnalysis(self.main_gui)
        self.scoring_combobox.addItems(scoring_options_dict[self.main_gui.model_type])
        self.singlemodels_combobox.addItems(self.model_analysis.estimator_dict.keys())
        self.multimodels_combobox.addItems(['bagging','boosting','stacking','voting'])
        self.set_model_scoring()
        self.set_model_cv()
        self.apply_pipeline()
        self.update_hyperparam_widgets()
        self.update_saved_models()

    def get_param_grid(self, layout):
        param_grid = {}
        for row in range(layout.rowCount()):
            checkbox = layout.itemAt(row, 0).widget()
            lineedit = layout.itemAt(row, 1).widget()
            if checkbox.isChecked():
                try:
                    exec(f'param_grid["{checkbox.text()}"] = {lineedit.text()}')
                except Exception as e:
                    qtw.QMessageBox.critical(self, 'Parameter Grid Error', f'Error raised: {e}')
        return param_grid

    def update_hyperparam_widgets(self):
        ## Single Model Hyperparams
        model_string = self.singlemodels_combobox.currentText()
        param_grid = self.model_analysis.param_grid_dict[model_string]

        # Clear paramgrid QFormLayout
        for row in reversed(range(self.singlemodel_paramgrid_layout.rowCount())):
            self.singlemodel_paramgrid_layout.removeRow(row)

        # Populate paramgrid QFromLayout
        for param, paramrange in param_grid.items():
            self.singlemodel_paramgrid_layout.addRow( 
                qtw.QCheckBox(param, self), qtw.QLineEdit(str(paramrange), self)
            )

        ## Multi Model Hyperparams
        model_string = self.multimodels_combobox.currentText()
        param_grid = self.model_analysis.param_grid_dict[model_string]

        # Clear paramgrid QFormLayout
        for row in reversed(range(self.multimodel_paramgrid_layout.rowCount())):
            self.multimodel_paramgrid_layout.removeRow(row)

        # Populate paramgrid QFromLayout
        for param, paramrange in param_grid.items():
            self.multimodel_paramgrid_layout.addRow( 
                qtw.QCheckBox(param, self), qtw.QLineEdit(str(paramrange), self)
            )

        

    def apply_pipeline(self):
        # Save pipe settings, check for new features, and update pipeline widgets
        self.main_gui.pipe.save_settings()
        self.main_gui.pipe.load_files()
        self.populate_pipeline_widgets()

        # Read data, get target
        train_path = os.path.join(self.main_gui.directory, 'train.csv')
        test_path = os.path.join(self.main_gui.directory, 'test.csv')
        train = self.main_gui.featengr.open_as_categoricals(train_path)
        test = self.main_gui.featengr.open_as_categoricals(test_path)
        self.y = train[self.main_gui.target_feature]
        
        # Build transformations
        scaling_dict = self.main_gui.pipe.get_scaling_dict()
        imputation_dict = self.main_gui.pipe.get_imputation_dict()
        featengr_path = os.path.join(self.main_gui.directory, self.main_gui.featengr.featengr_file)
        selected_features = self.main_gui.pipe.get_selected_features()

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
            X = pipeline.fit_transform(train)
            X_test = pipeline.transform(test)
        except Exception as e:
            qtw.QMessageBox.critical(self, 'Pipeline Error', f'Raised following error: {e}')
            return 1

        self.model_analysis.X = X
        self.model_analysis.X_test = X_test
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
                    encoding_dict = self.main_gui.featengr.get_encoding_dict(map_str)
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
                    encoding_dict = self.main_gui.featengr.get_encoding_dict(map_str)
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
        discrete_features_plus_none = self.main_gui.featengr.get_discrete_features()
        discrete_features_plus_none.insert(0, 'none')

        
        # Create title row
        titles = ['Feature','Scaling','Imputing','Category','Value']
        for icol, title in enumerate(titles):
            grid.addWidget(qtw.QLabel(title, self), 0, icol)

        # Populate by row
        for irow, (feature, row) in enumerate(self.main_gui.pipe.settings.iterrows()):

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
            self.main_gui.pipe.widget_ptrs.loc[feature] = [box, scaling_menu, imputing_menu, imputeby_menu, fillna_input]

            # Populate row
            grid.addWidget(box, irow+1, 0)
            grid.addWidget(scaling_menu, irow+1, 1)
            grid.addWidget(imputing_menu, irow+1, 2)
            grid.addWidget(imputeby_menu, irow+1, 3)
            grid.addWidget(fillna_input, irow+1, 4)

            
        grid.setRowStretch(grid.rowCount(), 1)
        self.main_gui.pipe.update_widgets()
        return  
# %%
