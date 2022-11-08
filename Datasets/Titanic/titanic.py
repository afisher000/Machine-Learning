
# %% Imports
import sys
sys.path.append('../../Datasets')
from ClassifierAnalysis import ClassifierModels
from Pipelines import Imputation, Scaling


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.regression import algo
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

# Save labels and IDs
y_train = train_df.Survived
train_df = train_df.drop(columns=['Survived'])
train_IDs = train_df['PassengerId']
test_IDs = test_df['PassengerId']

# Feature Engineering
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, df, y=None):

        return self

    def transform(self, df, y=None):
        # # Add features
        df['Family'] = df.SibSp + df.Parch
        df['logFare'] = np.log1p(df.Fare)
        df['isKid'] = np.where(df.Age<=14, 1, 0)
        df['isLargeFamily'] = np.where(df.Family>=4, 1, 0)

        # Encode features
        df.Sex = df.Sex.map({'male':0, 'female':1})
        df = self.convert_to_dummies(df, 'Embarked', prefix='Port', drop=True)
        df = self.convert_to_dummies(df, 'Pclass', prefix='Pclass', drop=False)


        # Features to drop
        drop_features = ['Parch','SibSp','Fare',
            'Ticket','Cabin','PassengerId','Name'
        ]

        df = df.drop(columns=drop_features)
        return df

    def convert_to_dummies(self, df, column, prefix=None, drop=True):
        df = pd.concat(
            [df, pd.get_dummies(df[column], prefix=prefix)], axis=1
        )
        if drop:
            df = df.drop(columns=column)
        return df


# Pipelines
scaling_data = { 
    'standard':['Age','logFare'],
    'minmax':['Family']
}
imputation_data = {
    'bycategory':[('Age','Pclass'), ('logFare','Pclass')]
}
pipeline = Pipeline([ 
    ('Feature Engineering', FeatureEngineering()),
    ('Imputation', Imputation(imputation_data)),
    ('Scaling', Scaling(scaling_data))#,
    #('Interactions', PolynomialFeatures(2, interaction_only=True))
])

X_train = pipeline.fit_transform(train_df)
X_test = pipeline.transform(test_df)



# %% Testing New Model Analysis
cm = ClassifierModels(X_train, y_train, scoring='accuracy')

cm.gridsearch('ridge', param_grid=[{'alpha':np.logspace(-2,2,10)}])
# cm.gridsearch('ridge_bagging', hyper_params = ['n_estimators'])

cm.gridsearch('svc', hyper_params=['C'])
# cm.gridsearch('svc_bagging', hyper_params = ['n_estimators'])
# cm.gridsearch('svc_boosting', hyper_params = ['n_estimators', 'algorithm'])

cm.gridsearch('kneighbors', hyper_params=['n_neighbors', 'algorithm'])

cm.gridsearch('randomforest', hyper_params=['n_estimators', 'min_samples_split'])

cm.gridsearch('extraforest', hyper_params=['n_estimators', 'min_samples_split'])
cm.compare_model_predictions(['ridge','svc', 'kneighbors', 'extraforest','randomforest'])

# %% Save predictions
y_test = best_model.predict(X_test)
predictions = pd.DataFrame({'PassengerId':test_IDs, 'Survived':y_test})
predictions.to_csv('submission.csv', index=False)
print('Updated submission.csv')



# %%
