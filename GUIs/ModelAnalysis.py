# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, SGDClassifier, SGDRegressor, RidgeClassifier
from sklearn.ensemble import ( 
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score, learning_curve, validation_curve
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
import pickle
import os


class BaseModel():
    def __init__(self, model_string, base_estimator, default_param_grid):
        self.label = model_string
        self.estimator = base_estimator
        self.default_param_grid = default_param_grid
        self.param_grid = None
        return

    def get_label(self):
        return self.label

class ModelAnalysis():
    def __init__(self, main_gui):
        self.main_gui = main_gui
        self.model_type = main_gui.model_type #classifier or regressor
        self.current_model_object = None
        self.cv = None

        self.X = None
        self.X_test = None
        self.y = None
        self.scoring = None

        self.models_folder = 'SavedModels'
        self.saved_models = self.load_saved_models()
        self.hardcoded_model_dicts()

    def hardcoded_model_dicts(self):
        # Define base_estimators and paramgrids
        self.param_grid_dict = { 
            'ridge':{ 
                'alpha':[0.01, 0.1, 1, 10, 100]
            },
            'supportvector':{ 
                'C':[.01, .1, 1, 10, 100],
                'kernel':['linear','poly','rbf'],
                'gamma':[.01, .1, 1, 10, 100]
            },
            'sgd':{ 
                'loss':['hinge','log_loss','squared_error'],
                'max_iter':[500, 1000],
                'alpha':[.01, .1, 1, 10, 100],
                'l1_ratio':[.01, .1, 1, 10]
            },
            'randomforest':{ 
                'n_estimators':[50, 100, 200],
                'criterion':['gini', 'entropy'],
                'max_depth':[3,5,10],
                'min_samples_split':[5, 10, 20]
            },
            'extraforest':{ 
                'n_estimators':[50, 100, 200],
                'criterion':['gini', 'entropy'],
                'max_depth':[3,5,10],
                'min_samples_split':[5, 10, 20]
            },
            'kneighbors':{ 
                'n_neighbors':[5, 10, 15],
                'weights':['distance','uniform'],
                'algorithm':['auto','ball_tree','kd_tree'],
                'leaf_size':[15, 30, 45]
            },
            'neural_MLP':{ 
                'solver':['lbfgs','sgd','adam'],
                'hidden_layer_sizes':[(100,), (100,100)],
                'activation':['logistic','tanh','relu'],
                'alpha':[1e-5, 1e-4, 1e-3],
                'learning_rate_init':[1e-4, 1e-3, 1e-2]
            },
            'stacking':{
                'cv':[3,5,10],
                'passthrough':[True, False]
            },
            'voting':{
                'voting':['soft','hard']
            },
            'boosting':{ 
                'n_estimators':[25, 50, 100],
                'learning_rate':[.1, .5, 1, 5, 10],
                'algorithm':['SAMME']
            },
            'bagging':{ 
                'n_estimators':[10, 30, 50], 
                'max_samples':[0.5, 0.75, 1.0],
                'max_features':[0.5, 0.75, 1.0],
                'bootstrap':[True, False]
            }
        }

        self.estimator_dict = { 
            'ridge':{'regressor':Ridge(), 'classifier':RidgeClassifier()},
            'supportvector':{'regressor':SVR(cache_size=1000), 'classifier':SVC(cache_size=1000)},
            'sgd':{'regressor':SGDRegressor(), 'classifier':SGDClassifier()},
            'randomforest':{'regressor':RandomForestRegressor(), 'classifier':RandomForestClassifier()},
            'extraforest':{'regressor':ExtraTreesRegressor(), 'classifier':ExtraTreesClassifier()},
            'kneighbors':{'regressor':KNeighborsRegressor(), 'classifier':KNeighborsClassifier()},
            'neural_MLP':{'regressor':MLPRegressor(), 'classifier':MLPClassifier()},
        }
        return

    def get_current_model_object(self):
        return self.current_model_object


    def create_model_object(self, model_string):
        # Get param_grid and consider selected saved models
        param_grid = self.param_grid_dict[model_string]
        isSelected = self.saved_models.checkbox.apply(lambda x: x.isChecked())  
        
        # Only one model for bagging and boosting
        if model_string in ['bagging','boosting']:
            if sum(isSelected)!=1:
                raise ValueError('Exactly one saved model must be selected')
            base_estimator = self.saved_models.model_object[isSelected][0]

        # Greater than two models for voting and stacking
        if model_string in ['voting','stacking']:
            if sum(isSelected)==0:
                raise ValueError('One or more models must be selected')
            estimator_list = list(zip( 
                self.saved_models.model_name[isSelected].tolist(),
                self.saved_models.model_object[isSelected].apply(lambda x: x.estimator).tolist()
            ))

        # Specify model label and estimator
        if model_string == 'bagging':
            estimator = BaggingRegressor(base_estimator) if self.model_type == 'regressor' else BaggingClassifier(base_estimator)
            model_label = str(base_estimator) + ' bagging'
        elif model_string == 'boosting':
            estimator = AdaBoostRegressor(base_estimator) if self.model_type == 'regressor' else AdaBoostClassifier(base_estimator)
            model_label = str(base_estimator)+' boosting'
        elif model_string == 'voting':
            estimator = VotingRegressor(estimator_list) if self.model_type == 'regressor' else VotingClassifier(estimator_list)
            model_label = str(estimator_list)+' voting'
        elif model_string == 'stacking':
            estimator = StackingRegressor(estimator_list) if self.model_type == 'regressor' else StackingClassifier(estimator_list)
            model_label = str(estimator_list) + ' stacking'
        else:
            estimator = self.estimator_dict[model_string][self.model_type]
            model_label = model_string
        
        # Create model object from label, estimator, and param_grid
        self.current_model_object = BaseModel(model_label, estimator, param_grid)
        return


    def gridsearch(self, param_grid=None):
        model_object = self.current_model_object

        # If param_grid not specified, use single default hyperparameter
        if param_grid is None:
            default_item = list(model_object.default_param_grid.items())[0]
            param_grid = {default_item[0]:default_item[1]}
        else:
            model_object.param_grid = param_grid

        # Fit grid
        grid = GridSearchCV(model_object.estimator, n_jobs=-1, cv=self.cv, param_grid=param_grid, scoring=self.scoring)
        grid.fit(self.X, self.y)
        scores = cross_val_score(grid.best_estimator_, self.X, self.y, cv=self.cv, n_jobs=-1, scoring=self.scoring)
       
        # Update model object
        model_object.mean_score = scores.mean()
        model_object.scores = scores
        model_object.estimator = grid.best_estimator_
        model_object.estimator.fit(self.X, self.y)
        model_object.train_predictions = model_object.estimator.predict(self.X)
        model_object.test_predictions = model_object.estimator.predict(self.X_test)
        return 


    def learning_curve(self, samples=10, ax=None):
        ''' Uses best model of class specified by model string to create learning curve.'''
        model_object = self.current_model_object

        train_sizes, train_scores, valid_scores = learning_curve( 
            model_object.estimator, self.X, self.y, train_sizes = np.linspace(.1, 1, samples), cv=self.cv, n_jobs=-1, scoring=self.scoring
        )

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        ax.plot(train_sizes, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_object.get_label())
        ax.set_xlabel('Training Size')
        ax.set_ylabel(self.scoring)
        ax.legend()
        return


    def validation_curve(self, ax=None):
        model_object = self.current_model_object
        if model_object.param_grid is None:
            raise ValueError('No hyperparameters used in current model')

        param_name, param_range = list(model_object.param_grid.items())[0]

        train_scores, valid_scores = validation_curve( 
            model_object.estimator, self.X, self.y, param_name=param_name, param_range=param_range, cv=self.cv, n_jobs=-1, scoring=self.scoring
        )

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(param_range, train_scores.mean(axis=1), label='Training')
        ax.plot(param_range, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_object.get_label())
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.scoring)
        if max(param_range)/min(param_range)>99:
            ax.set_xscale('log')
        ax.legend()
        return

    def load_saved_models(self):
        saved_models = pd.DataFrame(columns=['model_name','model_object','checkbox'])
        dirpath = os.path.join(self.main_gui.directory, self.models_folder)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        files = os.listdir(dirpath)
        for file in files:
            if file.endswith('.pkl'):
                model_name, model_object = pickle.load(open(os.path.join(dirpath, file), 'rb'))
                saved_models.loc[model_name] = [model_name, model_object, None]
        return saved_models

    def save_model(self, model_name):
        model_object = self.current_model_object
        model_object.estimator.fit(self.X, self.y)

        self.saved_models.loc[model_name] = [model_name, model_object, None]
        path = os.path.join(self.main_gui.directory, self.models_folder, model_name + '.pkl')
        pickle.dump([model_name, model_object], open(path, 'wb'))

    def load_model(self):
        selected_models = self.saved_models.checkbox.apply(lambda x: x.isChecked())
        if sum(selected_models)!=1:
            raise ValueError('Exactly one saved model must be selected')

        model_name = self.saved_models.index[selected_models][0]
        self.current_model_object = self.saved_models.loc[model_name, 'model_object']
        return
        
    def display_model_parameters(self):
        selected_models = self.saved_models.checkbox.apply(lambda x: x.isChecked())
        if sum(selected_models)!=1:
            raise ValueError('Exactly one saved model must be selected')

        model_name = self.saved_models.index[selected_models][0]
        parameters = self.saved_models.loc[model_name, 'model_object'].estimator.get_params()
        title = self.saved_models.loc[model_name, 'model_name']
        text = '\n'.join([f'{k}={str(v)}' for k,v in parameters.items()])
        return title, text


    def drop_selected_models(self):
        selected_models = self.saved_models.checkbox.apply(lambda x: x.isChecked())
        drop_models = self.saved_models.model_name[selected_models]
        for model_name in drop_models:
            self.saved_models = self.saved_models.drop(model_name)
            path = os.path.join(self.main_gui.directory, self.models_folder, model_name + '.pkl')
            os.remove(path)
        