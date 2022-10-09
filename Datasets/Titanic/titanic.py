
from random import Random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# %% Munging
# Fare=0 datapoints?

# Read and append data
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
y_train = df_train.Survived
test_IDs = df_test['PassengerId']

data = pd.concat([df_train, df_test])



# Hot Encode Sex
data.Sex = data.Sex.map({'male':0, 'female':1})

# Pipelines to fill NA with median and scale (Fare uses quantile scaling)
std_scale_pipeline = Pipeline([ 
    ('imputer', SimpleImputer(strategy='median')),
    ('qnt_scale', QuantileTransformer(n_quantiles=100))
])

qnt_scale_pipeline = Pipeline([ 
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

drop_columns = ['Cabin','Embarked','Ticket','Name','PassengerId', 'Survived']
std_scale_columns = ['Pclass','Sex','Age','SibSp','Parch']
full_pipe = ColumnTransformer([
    ('drop', 'drop', drop_columns),
    ('non_fare', std_scale_pipeline, std_scale_columns),
    ('fare', qnt_scale_pipeline, ['Fare'])
])

full_pipe.fit(df_train)
X_train = full_pipe.transform(df_train)
X_test = full_pipe.transform(df_test)





# %% Models
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Linear Model, saturates at .79 (.03)
print('Linear Model')
model = RidgeClassifierCV()
scores = cross_val_score(model, X_train, y_train, cv=10)
print(f'Scores = {np.mean(scores):.2f} ({np.std(scores):.2f})')

# Polynomial Model, saturates at .82 (.03)
# print('Poly Model')
# X_train_poly = PolynomialFeatures(degree=2).fit_transform(X_train)
# model = RidgeClassifierCV()
# scores = cross_val_score(model, X_train_poly, y_train, cv=10)
# print(f'Scores = {np.mean(scores):.2f} ({np.std(scores):.2f})')


# SVC with kernel, saturates at .82 (.04)
# print('SVC Kernel Model')
# model = SVC(C=1, kernel='poly', degree=3)
# scores = cross_val_score(model, X_train, y_train, cv=10)
# model.fit(X_train, y_train)
# print(f'Scores = {np.mean(scores):.2f} ({np.std(scores):.2f})')
# y_test = model.predict(X_test)

# Nearest Neighbors
# print('Nearest Neighbors')
# model = KNeighborsClassifier(n_neighbors=5)
# scores = cross_val_score(model, X_train, y_train, cv=10)
# print(f'Scores = {np.mean(scores):.2f} ({np.std(scores):.2f})')

# Random Forest
# print('Random Forest')
# model = RandomForestClassifier(n_estimators = 100, max_depth=4)
# scores = cross_val_score(model, X_train, y_train, cv=10)
# print(f'Scores = {np.mean(scores):.2f} ({np.std(scores):.2f})')


# %%
predictions = np.hstack([ 
    test_IDs.values.reshape(-1,1),
    y_test.reshape(-1,1)
])

result = pd.DataFrame(data=predictions, columns=['PassengerId', 'Survived'])
result.to_csv('predictions.csv', index=False)
# %%
