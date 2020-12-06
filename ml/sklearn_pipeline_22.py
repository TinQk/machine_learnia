# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:43:19 2020

@author: Quentin
"""

# Pipeline = chaine de traitement = transformer(s) + estimator

import numpy as np
import seaborn as sns

iris = sns.load_dataset("iris")

print(iris.head())

X = iris.drop('species', axis=1)
y = iris['species']

# Encoder les y
from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()
y = encoder.fit_transform(y) 


# split le jeux de donnée
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Prévoir notre pipeline

# Transformer --> Normalisation
from sklearn.preprocessing import StandardScaler

# Estimator --> classifier
from sklearn.neighbors import KNeighborsClassifier


# Create a pipeline
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(),
                      KNeighborsClassifier())

model.fit(X_train, y_train)
print(model.predict(X_test))