# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, SGDClassifier, SGDRegressor, RidgeClassifier
from sklearn.ensemble import ( 
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score, learning_curve, validation_curve
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor



# To Implement
    # Use SAMME when probabilities not allowed
    # Implement voting ensemble strategies

# Ensemble techniques
class BaggingRegressorModel():
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.model = BaggingRegressor(self.base_estimator)
        self.score = None
        self.hyper_params = ['n_estimators', 'max_samples', 'max_features', 'bootstrap']
        self.param_grid = { 
            'n_estimators':[10, 30, 50], 
            'max_samples':[0.5, 0.75, 1.0],
            'max_features':[0.5, 0.75, 1.0],
            'bootstrap':[True, False]
        }

class AdaBoostRegressorModel():
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.model = AdaBoostRegressor(self.base_estimator)
        self.score = None
        self.hyper_params = ['n_estimators', 'learning_rate','algorithm']
        self.param_grid = { 
            'n_estimators':[25, 50, 100],
            'learning_rate':[.1, .5, 1, 5, 10],
            'algorithm':['SAMME']
        }

# Base Model Inheritance Class
class BaseModel():
    def __init__(self):
        return

    def get_score(self):
        if hasattr(self, 'score'):
            return self.score
        else:
            print('No score. Model has yet to be fit.')
            return

    def hyper_params(self):
        return list(self.param_grid.keys())

    def update_bagging_model(self):
        self.bagging = BaggingRegressorModel(self.model)
        return

    def update_boosting_model(self):
        self.boosting = AdaBoostRegressorModel(self.model)

# Base Estimatorsr
class RidgeRegressorModel(BaseModel):
    def __init__(self):
        self.model = Ridge()
        self.param_grid = {'alpha':[0.01, 0.1, 1, 10, 100]}
        self.update_bagging_model()
        self.update_boosting_model()

class SVRModel(BaseModel):
    def __init__(self):
        self.model = SVR(cache_size=1000)
        self.param_grid = { 
            'C':[.01, .1, 1, 10, 100],
            'kernel':['linear','poly','rbf'],
            'gamma':[.01, .1, 1, 10, 100]
        }
        self.update_bagging_model()
        self.update_boosting_model()
        
class SGDRegressorModel(BaseModel):
    def __init__(self):
        self.model = SGDRegressor()
        self.param_grid = { 
            'loss':['hinge','log_loss','squared_error'],
            'max_iter':[500, 1000],
            'alpha':[.01, .1, 1, 10, 100],
            'l1_ratio':[.01, .1, 1, 10]
        }
        self.update_bagging_model()
        self.update_boosting_model()

class RandomForestRegressorModel(BaseModel): # Technically ensemble, but easier to run here
    def __init__(self):
        self.model = RandomForestRegressor()
        self.param_grid = { 
            'n_estimators':[50, 100, 200],
            'criterion':['gini', 'entropy'],
            'max_depth':[3,5,10],
            'min_samples_split':[5, 10, 20]
        }
        self.update_bagging_model()
        self.update_boosting_model()

class ExtraTreesRegressorModel(BaseModel):
    def __init__(self):
        self.model = ExtraTreesRegressor()
        self.param_grid = { 
            'n_estimators':[50, 100, 200],
            'criterion':['gini', 'entropy'],
            'max_depth':[3, 5, 10], 
            'min_samples_split':[5, 10, 20]
        }
        self.update_bagging_model()
        self.update_boosting_model()

class KNeighborsRegressorModel(BaseModel):
    def __init__(self):
        self.model = KNeighborsRegressor()
        self.param_grid = { 
            'n_neighbors':[5, 10, 15],
            'weights':['distance','uniform'],
            'algorithm':['auto','ball_tree','kd_tree'],
            'leaf_size':[15, 30, 45]
        }
        self.update_bagging_model()
        self.update_boosting_model()


class RegressorModels():
    def __init__(self, X=None, y=None, scoring=None):
        self.X = X
        self.y = y
        self.scoring = scoring

        # Add models
        self.models = { 
            'ridge':RidgeRegressorModel(),
            'svr':SVRModel(),
            'randomforest':RandomForestRegressorModel(),
            'kneighbors':KNeighborsRegressorModel(),
            'extraforest':ExtraTreesRegressorModel()
        }


    def parse_model(self, model_string):
        bagging = False
        boosting = False

        if model_string.endswith('_bagging'):
            model_string = model_string.removesuffix('_bagging')
            bagging = True
        elif model_string.endswith('_boosting'):
            model_string = model_string.removesuffix('_boosting')
            boosting = True

        if model_string in self.models.keys():
            base_estimator = self.models[model_string]
            if bagging:
                if not hasattr(base_estimator, 'bagging'):
                    base_estimator.update_bagging_model()
                model = base_estimator.bagging
            elif boosting:
                if not hasattr(base_estimator, 'boosting'):
                    base_estimator.update_boosting_model()
                model = base_estimator.boosting
            else:
                model = base_estimator
        else:
            raise ValueError(f'{model_string} not a valid model identifier.')
        return model, bagging, boosting

    def gridsearch(self, model_string, cv=10, param_grid=None, hyper_params=None):
        '''Specify model by string. Can specify specific hyperparameters or override param_grid'''
        model, bagging, boosting = self.parse_model(model_string)

        # Specificy param_grid
        param_grid = model.param_grid if param_grid is None else param_grid
        if hyper_params is not None:
            param_grid = [{k:param_grid[k] for k in hyper_params}]

        grid = GridSearchCV(model.model, n_jobs=-1, cv=cv, param_grid=param_grid, scoring=self.scoring)
        grid.fit(self.X, self.y)
        self.grid = grid
        model.model = grid.best_estimator_
        scores = cross_val_score(grid.best_estimator_, self.X, self.y, cv=10, n_jobs=-1, scoring=self.scoring)
        model.score = scores.mean()

        return scores


    def learning_curve(self, model_string, samples=10, cv=10, ax=None):
        ''' Uses best model of class specified by model string to create learning curve.'''
        model, bagging, boosting = self.parse_model(model_string)

        train_sizes, train_scores, valid_scores = learning_curve( 
            model.model, self.X, self.y, train_sizes = np.linspace(.1, 1, samples), cv=cv, n_jobs=-1, scoring=self.scoring
        )
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        ax.plot(train_sizes, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_string)
        ax.set_xlabel('Training Size')
        ax.set_ylabel(self.scoring)
        ax.legend()
        return


    def validation_curve(self, model_string, param_name, param_range, cv=10, ax=None):
        model, bagging, boosting = self.parse_model(model_string)
        
        train_scores, valid_scores = validation_curve( 
            model.model, self.X, self.y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=-1, scoring=self.scoring
        )

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(param_range, train_scores.mean(axis=1), label='Training')
        ax.plot(param_range, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_string)
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.scoring)
        if max(param_range)/min(param_range)>100:
            ax.set_xscale('log')
        ax.legend()
        return


    # def compare_model_predictions(self, model_strings):
    #     predictions = pd.DataFrame(columns = model_strings)
    #     for model_string in model_strings:
    #         model = self.parse_model(model_string)[0]
    #         predictions[model_string] = model.model.predict(self.X)
        
    #     self.model_correlations = predictions.corr()
    #     print(self.model_correlations)

