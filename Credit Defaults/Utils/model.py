# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:14:40 2024

@author: afisher
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold, StratefiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc
import matplotlib.pyplot as plt

def model(train, test, id_var, label_var, n_folds = 5):
    
    # Extract ids and labels
    train_ids = train[id_var]
    train_y = train[label_var]
    test_ids = test[id_var]
    
    # Remove from dataframe
    train = train.drop(columns = [id_var, label_var])
    test = test.drop(columns = [id_var])
    
    # Make sure aligned, get feature names
    train, test = train.align(test, join='inner', axis=1)
    features = list(train.columns)
    
    # Convert to np arrays
    train_X = np.array(train)
    test_X = np.array(test)
    
    # Create empty arrays
    feature_importance_values = np.zeros(len(features))
    test_predictions = np.zeros(test.shape[0])
    train_predictions = np.zeros(train.shape[0])   
    validation_scores = []
    fit_scores = []
    
    # Iterate over folds
    kfold = StratefiedKFold(n_splits = n_folds)
    for j, (fit_mask, validate_mask) in enumerate(kfold.split(train_X, train_y)):
        # Split training data into fit and validate
        fit_X, fit_y = train_X[fit_mask], train_y[fit_mask]
        validate_X, validate_y = train_X[validate_mask], train_y[validate_mask]
        
        # Create model
        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1,
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
        # Fit, record results for fit and validate sets
        model.fit(fit_X, fit_y,
                  eval_metric = 'auc', eval_set = [(fit_X, fit_y), (validate_X, validate_y)],
                  eval_names = ['fit', 'validate'], early_stopping_rounds=100, verbose=200)
        
        # Record best iteration
        best_iteration = model.best_iteration_
        
        # Make test predictions (average over kfolds)
        test_predictions += model.predict_proba(test_X, num_iteration=best_iteration)[:,1]/kfold.n_splits
        
        # Compute out-of-fold train predictions
        train_predictions[validate_mask] = model.predict_proba(validate_X, num_iteration=best_iteration)[:,1]
        
        # Record the best scores
        validation_score = model.best_score_['validate']['auc']
        fit_score = model.best_score_['fit']['auc']
        
        validation_scores.append(validation_score)
        fit_scores.append(fit_score)
        
        # Clean up memory
        gc.enable()
        del model, fit_X, validate_X
        gc.collect()
        
        
    # Create submission dataframe
    submission = pd.DataFrame({id_var:test_ids, label_var:test_predictions})
    
    # Make feature importance dataframe
    feature_importances = pd.DataFrame({'feature':features, 'important':feature_importance_values})
    
    # Create fit/validation dataframe for folds
    fold_labels = list(range(n_folds)) + ['overall']
    
    fit_scores.append(fit_score.mean())
    
    out_of_fold_auc = roc_auc_score(train_y, train_predictions)
    validation_scores.append()
    
    metrics = pd.DataFrame({'fold':fold_labels, 'fit':fit_scores, 'validate':validation_scores})
    
    return submission, feature_importances, metrics
    
    
    
    
        
        
    