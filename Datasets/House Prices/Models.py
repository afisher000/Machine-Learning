

# Steps to train linear model
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA 
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import learning_curve, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class ModelAnalysis():
    def __init__(self, model, X, y, param_grid, scoring='neg_root_mean_squared_error'):
        self.X = X
        self.y = y
        self.scoring = scoring

        # Apply gridsearch
        self.grid = GridSearchCV(model, n_jobs=-1, cv=10, param_grid=param_grid, return_train_score=True,
            scoring='neg_root_mean_squared_error'
        )
        self.grid.fit(X,y)
        self.best_estimator = self.grid.best_estimator_

        # Best results
        scores = cross_val_score(self.grid.best_estimator_, self.X, self.y, cv=10, n_jobs=-1, scoring=self.scoring)
        print(f'Error = {scores.mean():.3f} ({scores.std()})')

    def plot_learningcurve(self, samples=10, cv=10):
        train_sizes, train_scores, valid_scores = learning_curve( 
            self.best_estimator, self.X, self.y, train_sizes=np.linspace(0.1,1,samples), cv=10, n_jobs=-1, 
            scoring=self.scoring
        )
        fig, ax = plt.subplots()
        ax.plot(train_sizes, train_scores.mean(axis=1), label='Training')
        ax.plot(train_sizes, valid_scores.mean(axis=1), label='Validation')
        ax.legend()
        return

    def plot_gridsearch(self, param, logx=False):
        grid_df = pd.DataFrame(self.grid.cv_results_)
        if param in self.grid.param_grid.keys():
            param_col = f'param_{param}'
            param_df = grid_df[[param_col, 'mean_test_score','mean_train_score']]
            param_df.plot(x=param_col, y=['mean_test_score','mean_train_score'],
                style='*', logx=logx)
        else:
            raise Exception(f'{param} was not found in param_grid.')
        return

    def plot_PCA_decisionboundary(self):
        pca = PCA(n_components=2, copy=True)
        X_pca = pca.fit_transform(self.X)
        model = clone(self.grid.best_estimator_)
        model.fit(X_pca, self.y)
        decision_boundary = DecisionBoundaryDisplay.from_estimator( 
            model, X_pca, response_method='predict',
            xlabel='F0', ylabel='F1'
        )
        decision_boundary.ax_.scatter(X_pca[:,0], X_pca[:,1], c=self.y)
        plt.show()
        return