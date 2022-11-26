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


class BaggingModel():
    def __init__(self, base_estimator, model_type):
        if model_type == 'regressor':
            self.current_estimator = BaggingRegressor(base_estimator)
        elif model_type == 'classifier':
            self.current_estimator = BaggingClassifier(base_estimator)

        self.param_grid = { 
            'n_estimators':[10, 30, 50], 
            'max_samples':[0.5, 0.75, 1.0],
            'max_features':[0.5, 0.75, 1.0],
            'bootstrap':[True, False]
        }

class BoostingModel():
    def __init__(self, base_estimator, model_type):
        if model_type == 'regressor':
            self.current_estimator = AdaBoostRegressor(base_estimator)
        elif model_type == 'classifier':
            self.current_estimator = AdaBoostClassifier(base_estimator)

        self.param_grid = { 
            'n_estimators':[25, 50, 100],
            'learning_rate':[.1, .5, 1, 5, 10],
            'algorithm':['SAMME']
        }

class StackingModel():
    def __init__(self, estimator_list, model_type):
        if model_type == 'regressor':
            self.current_estimator = StackingRegressor(estimator_list)
        elif model_type == 'classifier':
            self.current_estimator = StackingClassifier(estimator_list)

        self.param_grid = {
            'cv':[3,5,10],
            'passthrough':[True, False]
        }



class VotingModel():
    def __init__(self, estimator_list, model_type):
        if model_type == 'regressor':
            self.current_estimator = VotingRegressor(estimator_list)
        elif model_type == 'classifier':
            self.current_estimator = VotingClassifier(estimator_list)

        self.param_grid = {'voting':['hard','soft']}


class BaseModel():
    def __init__(self, model, param_grid, model_type):
        self.model_type = model_type
        self.current_estimator = model
        self.param_grid = param_grid
        self.score = None
        self.train_predictions = None
        self.test_predictions = None
        self.update_bagging_model()
        self.update_boosting_model()
        return

    def update_bagging_model(self):
        self.bagging = BaggingModel(self.current_estimator, self.model_type)
        return

    def update_boosting_model(self):
        self.boosting = BoostingModel(self.current_estimator, self.model_type)


class ModelAnalysis():
    def __init__(self, main_gui, X=None, X_test=None, y=None, scoring=None):
        self.main_gui = main_gui
        self.model_type = main_gui.model_type #classifier or regressor
        self.current_model_object = None
        self.X = X
        self.X_test = X_test
        self.y = y
        self.scoring = scoring
        self.models_folder = 'SavedModels'
        self.saved_models = self.load_saved_models()
        self.defined_models = self.define_models()

    def load_saved_models(self):
        saved_models = pd.DataFrame(columns=['model_label','model_object','checkbox'])
        dirpath = os.path.join(self.main_gui.directory, self.models_folder)

        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        files = os.listdir(dirpath)
        for file in files:
            if file.endswith('.pkl'):
                model_label, model_object = pickle.load(open(os.path.join(dirpath, file), 'rb'))
                saved_models.loc[model_label] = [model_label, model_object, None]
        return saved_models

    def save_model(self, model_label, model_string, bagging, boosting, cv=None):
        # Test whether this saves a reference to the model or a copy of the model
        # distinct from the model_string.
        model_object, _ = self.get_model_object(model_string, bagging, boosting)
        model_object.current_estimator.fit(self.X, self.y)
        model_object.train_predictions = model_object.current_estimator.predict(self.X)
        if model_object.train_predictions is None:
            raise ValueError('Model has not been fit yet.')
            return 1
        

        self.saved_models.loc[model_label] = [model_label, model_object, None]
        path = os.path.join(self.main_gui.directory, self.models_folder, model_label + '.pkl')
        pickle.dump([model_label, model_object], open(path, 'wb'))

    def drop_selected_models(self):
        selected_models = self.saved_models.checkbox.apply(lambda x: x.isChecked())
        drop_models = self.saved_models.model_label[selected_models]
        for model_label in drop_models:
            self.saved_models = self.saved_models.drop(model_label)
            path = os.path.join(self.main_gui.directory, self.models_folder, model_label + '.pkl')
            os.remove(path)
        

    def define_models(self):
        # Define base_estimators and paramgrids
        paramgrid_dict = { 
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
            } 
        }
        base_estimator_dict = { 
            'ridge':{'regressor':Ridge(), 'classifier':RidgeClassifier()},
            'supportvector':{'regressor':SVR(cache_size=1000), 'classifier':SVC(cache_size=1000)},
            'sgd':{'regressor':SGDRegressor(), 'classifier':SGDClassifier()},
            'randomforest':{'regressor':RandomForestRegressor(), 'classifier':RandomForestClassifier()},
            'extraforest':{'regressor':ExtraTreesRegressor(), 'classifier':ExtraTreesClassifier()},
            'kneighbors':{'regressor':KNeighborsRegressor(), 'classifier':KNeighborsClassifier()},
            'neural_MLP':{'regressor':MLPRegressor(), 'classifier':MLPClassifier()}
        }

        defined_models = {}
        for model_name in base_estimator_dict.keys():
            defined_models[model_name] = BaseModel( 
                base_estimator_dict[model_name][self.model_type],
                paramgrid_dict[model_name],
                self.model_type
            )
        return defined_models

    def get_model_object(self, model_string, bagging, boosting):
        base_estimator = self.defined_models[model_string]
        if bagging:
            model_object = base_estimator.bagging
            model_label = model_string + ' with bagging'
        elif boosting:
            model_object = base_estimator.boosting
            model_label = model_string + ' with boosting'
        else:
            model_object = base_estimator
            model_label = model_string
        return model_object, model_label

    def gridsearch(self, model_string, cv=10, param_grid=None, bagging=False, boosting=False):
        '''Specify model by string.'''
        if model_string not in self.defined_models.keys():
            raise ValueError('Model not found in defined models')
            return 1
        model_object, _ = self.get_model_object(model_string, bagging, boosting)


        # If param_grid not specified, use single default hyperparameter
        if param_grid is None:
            default_hyperparam = list(model_object.param_grid.keys())[0]
            param_grid = {default_hyperparam:model_object.param_grid[default_hyperparam]}

        # Fit grid
        grid = GridSearchCV(model_object.current_estimator, n_jobs=-1, cv=cv, param_grid=param_grid, scoring=self.scoring)
        grid.fit(self.X, self.y)
        scores = cross_val_score(grid.best_estimator_, self.X, self.y, cv=10, n_jobs=-1, scoring=self.scoring)
       
        # Update model
        model_object.score = scores.mean()
        model_object.current_estimator = grid.best_estimator_
        model_object.current_estimator.fit(self.X, self.y)
        model_object.train_predictions = model_object.current_estimator.predict(self.X)
        model_object.test_predictions = model_object.current_estimator.predict(self.X_test)

        return scores


    def learning_curve(self, model_string, samples=10, cv=10, ax=None, bagging=False, boosting=False):
        ''' Uses best model of class specified by model string to create learning curve.'''
        model_object, model_label = self.get_model_object(model_string, bagging, boosting)

        train_sizes, train_scores, valid_scores = learning_curve( 
            model_object.current_estimator, self.X, self.y, train_sizes = np.linspace(.1, 1, samples), cv=cv, n_jobs=-1, scoring=self.scoring
        )
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        ax.plot(train_sizes, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_label)
        ax.set_xlabel('Training Size')
        ax.set_ylabel(self.scoring)
        ax.legend()
        return


    def validation_curve(self, param_name, param_range, model_string=None, cv=10, ax=None, bagging=False, boosting=False):
        model_object, model_label = self.get_model_object(model_string, bagging, boosting)
        
        train_scores, valid_scores = validation_curve( 
            model_object.current_estimator, self.X, self.y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=-1, scoring=self.scoring
        )

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(param_range, train_scores.mean(axis=1), label='Training')
        ax.plot(param_range, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_label)
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.scoring)
        if max(param_range)/min(param_range)>99:
            ax.set_xscale('log')
        ax.legend()
        return

