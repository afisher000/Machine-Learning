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


# To Implement
    # Use SAMME when probabilities not allowed
    # Implement voting ensemble strategies

# Ensemble techniques
class BaggingClassifierModel():
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.model = BaggingClassifier(self.base_estimator)
        self.score = None
        self.hyper_params = ['n_estimators', 'max_samples', 'max_features', 'bootstrap']
        self.param_grid = [{ 
            'n_estimators':[10, 30, 50], 
            'max_samples':[0.5, 0.75, 1.0],
            'max_features':[0.5, 0.75, 1.0],
            'bootstrap':[True, False]
        }]

class AdaBoostClassifierModel():
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.model = AdaBoostClassifier(self.base_estimator)
        self.score = None
        self.hyper_params = ['n_estimators', 'learning_rate','algorithm']
        self.param_grid = [{ 
            'n_estimators':[25, 50, 100],
            'learning_rate':np.logspace(-1, 1, 3),
            'algorithm':['SAMME']
        }]

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
        return list(self.param_grid[0].keys())

    def update_bagging_model(self):
        self.bagging = BaggingClassifierModel(self.model)
        return

    def update_boosting_model(self):
        self.boosting = AdaBoostClassifierModel(self.model)

# Base Estimatorsr
class RidgeClassifierModel(BaseModel):
    def __init__(self):
        self.model = RidgeClassifier()
        self.param_grid = [{'alpha':np.logspace(-2,2,5)}]


class SVCModel(BaseModel):
    def __init__(self):
        self.model = SVC(cache_size=1000)
        self.param_grid = [{ 
            'C':np.logspace(-2, 2, 5),
            'kernel':['linear','poly','rbf'],
            'gamma':np.logspace(-2,2,5)
        }]
        
class SGDClassifierModel(BaseModel):
    def __init__(self):
        self.model = SGDClassifier()
        self.param_grid = [{ 
            'loss':['hinge','log_loss','squared_error'],
            'max_iter':[500, 1000],
            'alpha':np.logspace(-5,2,5),
            'l1_ratio':np.logspace(-2,1,3)
        }]

class RandomForestClassifierModel(BaseModel): # Technically ensemble, but easier to run here
    def __init__(self):
        self.model = RandomForestClassifier()
        self.param_grid = [{ 
            'n_estimators':[50, 100, 200],
            'criterion':['gini', 'entropy'],
            'max_depth':[3,5,10],
            'min_samples_split':[5, 10, 20]
        }]

class ExtraTreesClassifierModel(BaseModel):
    def __init__(self):
        self.model = ExtraTreesClassifier()
        self.param_grid = [{ 
            'n_estimators':[50, 100, 200],
            'criterion':['gini', 'entropy'],
            'max_depth':[3, 5, 10], 
            'min_samples_split':[5, 10, 20]
        }]


class KNeighborsClassifierModel(BaseModel):
    def __init__(self):
        self.model = KNeighborsClassifier()
        self.param_grid = [{ 
            'n_neighbors':[5, 10, 15],
            'weights':['distance','uniform'],
            'algorithm':['auto','ball_tree','kd_tree'],
            'leaf_size':[15, 30, 45]
        }]


class ClassifierModels():
    def __init__(self, X, y, scoring):
        self.X = X
        self.y = y
        self.scoring=scoring

        # Add models
        self.models = { 
            'ridge':RidgeClassifierModel(),
            'svc':SVCModel(),
            'randomforest':RandomForestClassifierModel(),
            'kneighbors':KNeighborsClassifierModel(),
            'extraforest':ExtraTreesClassifierModel()
        }


    def parse_model(self, model_string):
        bagging = False
        boosting = False

        if model_string.endswith('_bagging'):
            model_string = model_string.removesuffix('_bagging')
            bagging = True
        elif model_string.endswith('boosting'):
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
            param_grid = [{k:param_grid[0][k] for k in hyper_params}]

        grid = GridSearchCV(model.model, n_jobs=-1, cv=cv, param_grid=param_grid, scoring=self.scoring)
        grid.fit(self.X, self.y)
        self.grid = grid
        model.model = grid.best_estimator_
        scores = cross_val_score(grid.best_estimator_, self.X, self.y, cv=10, n_jobs=-1, scoring=self.scoring)
        model.score = scores.mean()

        # Pring results
        print(f'{model_string}: {scores.mean():.3f} ({scores.std():.3f})')
        return


    def learning_curve(self, model_string, samples=10, cv=10):
        ''' Uses best model of class specified by model string to create learning curve.'''
        model, bagging, boosting = self.parse_model(model_string)

        train_sizes, train_scores, valid_scores = learning_curve( 
            model.model, self.X, self.y, train_sizes = np.linspace(.1, 1, samples), cv=cv, n_jobs=-1, scoring=self.scoring
        )
        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        ax.plot(train_sizes, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_string)
        ax.set_xlabel('Training Size')
        ax.set_ylabel(self.scoring)
        ax.legend()
        return


    def validation_curve(self, model_string, param_name, param_range, cv=10):
        model, bagging, boosting = self.parse_model(model_string)
        
        train_scores, valid_scores = validation_curve( 
            model.model, self.X, self.y, param_name=param_name, param_range=param_range, cv=cv, n_jobs=-1, scoring=self.scoring
        )

        fig, ax = plt.subplots()
        ax.plot(param_range, train_scores.mean(axis=1), label='Training')
        ax.plot(param_range, valid_scores.mean(axis=1), label='Validation')
        ax.set_title(model_string)
        ax.set_xlabel(param_name)
        ax.set_ylabel(self.scoring)
        ax.legend()
        return

    def compare_model_predictions(self, model_strings):
        predictions = pd.DataFrame(columns = model_strings)
        for model_string in model_strings:
            model = self.parse_model(model_string)[0]
            predictions[model_string] = model.model.predict(self.X)
        
        self.model_correlations = predictions.corr()
        print(self.model_correlations)

