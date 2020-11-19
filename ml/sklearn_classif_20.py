# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:09:38 2020

@author: Quentin
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns


# Retravailler dataset d'exemple
titanic = sns.load_dataset("titanic")
titanic = titanic[['survived', 'pclass', 'sex', 'age']]
titanic.dropna(axis=0, inplace=True) #inplace true pour remplacer la variable titanic
titanic['sex'] = titanic['sex'].astype('category').cat.codes
print(titanic.shape)
print(titanic.head())

X = titanic.drop('survived', axis=1)
y = titanic['survived']



# ML

# les x sont les "features" --> données en entrée, on peut en avoir plusieurs
# les y sont les données en sortie

# 1. sélectionner un model et préciser ses hyperparametres

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

# 2. entrainer un model sur les données X, y via .fit
model.fit(X,y)

# 3. évaluer le model via .score
print(model.score(X,y))


# exo : tester d'autres options pour augmenter le score (attention ! pas bien de tester sur le set d'entrainement !)
for i in range(10):
    m = KNeighborsClassifier(n_neighbors=i+1)
    m.fit(X,y)
    print(str(i+1) + ' --> ' + str(m.score(X,y)))


# 4. utiliser le model via .predict
#predictions = model.predict(X)

def survie(model, pclass, sex, age):
    x = np.array([pclass, sex, age]).reshape(1,3)
    print(model.predict(x))
    print(model.predict_proba(x))
    
survie(model, 3, 1, 28)
    




# graph des predictions
#plt.plot(X, predictions, c='r')
