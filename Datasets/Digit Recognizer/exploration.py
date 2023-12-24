# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:25:53 2023

@author: afisher
"""
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import SGDClassifier

# %% Read data
df = pd.read_csv('train.csv')

y_train = df.label.values
X_train = df.drop(columns='label').values


# %% Plot example image
some_digit = X_train[500, :]
some_digit_image= some_digit.reshape(28,28)

plt.imshow(some_digit_image,
           cmap = matplotlib.cm.binary,
           interpolation='nearest'
           )



# %% Train SGD to identify 5s
y_train_5 = (y_train==5)

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Scoring is not useful in this case since a "never5" classifier will have 90% 
# accuracy. 
cross_val_score(sgd_clf, X_train, y_train_5, cv=5, scoring='accuracy')




