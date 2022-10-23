
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Pipelines import Imputation, FeatureEngineering, Scaling, SelectFeatures
from Models import ModelAnalysis

from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# %% 
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')

# Save labels and IDs
y_train = df_train.Survived
train_IDs = df_train['PassengerId']
test_IDs = df_test['PassengerId']

# Pipelines
std_scale_features = ['Age', 'Family_Size', 'Pclass']
qnt_scale_features = ['Fare']
keep_features = ['Age','Family_Size','Fare','Sex','Pclass']
pipeline = Pipeline([ 
    ('Imputation', Imputation()),
    ('Feature_Engr', FeatureEngineering()),
    ('Scaling', Scaling(
        std_scale_features=std_scale_features,
        qnt_scale_features=qnt_scale_features
    )),
    ('Features_to_keep', SelectFeatures(keep_features=keep_features))
])

# Prepared data
X_train = pipeline.fit_transform(df_train)
X_test = pipeline.transform(df_test)



# %% Models
model = SVC(cache_size = 1000)
param_grid = {
    'C':np.logspace(-2,2,5), 'gamma':np.logspace(-2,2,5)
}
analysis = ModelAnalysis(model, X_train, y_train, param_grid=param_grid, scoring='accuracy')



# scores = cross_val_score(model, X_train, y_train, cv=10, n_jobs=-1)
# print(f'Score: {scores.mean():.2f} ({scores.std():.2f})')
# model.fit(X_train, y_train)
# y_test = model.predict(X_test)

# thresh = 0.5
# y_train_prob = model.predict_proba(X_train)
# y_pred = model.predict(X_train)

# # Incorrect fits
# wrong_pred = (y_train != y_pred)
# X_wrong = X_train[wrong_pred]
# X_right = X_train[~wrong_pred]

# # Plot right vs wrong
# fig, ax = plt.subplots()
# X_right.plot.scatter(x='Age', y='Fare', c='g', ax=ax, label='right')
# X_wrong.plot.scatter(x='Age', y='Fare', c='r', ax=ax, label='wrong')




# %% Save predictions
predictions = pd.DataFrame({'PassengerId':test_IDs, 'Survived':y_test})
predictions.to_csv('predictions.csv', index=False)
# %%
