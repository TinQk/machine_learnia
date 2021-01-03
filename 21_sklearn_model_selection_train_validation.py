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

#print(titanic.shape)
#print(titanic.head())

# Déterminer les entrées et les sorties
# les x sont les "features" --> données en entrée, on peut en avoir plusieurs
# les y sont les données en sortie

X = titanic.drop('survived', axis=1)
y = titanic['survived']

#print('Entrées (features): ', X.shape, '\n', X.head())
#print('Sorties :', y.shape, '\n', y.head())



########## ML

# 1. Créer un train set et un test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#print('train set shape : ', X_train.shape)
#print('test set shape : ', X_test.shape)

plt.figure()
plt.subplot()

plt.scatter(X_train['pclass'], X_train['age'], c='green')
plt.scatter(X_test['pclass'], X_test['age'], c='yellow')



# 2. Sélectionner un model et l'évaluer sans réglage

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier() # n_neighbors=1

# entrainer le model sur le train set via .fit
model.fit(X_train, y_train)

# Evaluer le model via .score sur le test set
#print('Test score sans régler les hyperparamètres: ', model.score(X_test,y_test))


# 3. Régler les hyperparamètres du model 

# Pour chaque valeur de l'hyperparamètre à tester, effetuer une cross validation :
#   - découper le train set en plusieurs sous ensemble, dont un sera le 'validation set'
#   - entrainer le model sur chaque sous ensemble, et évaluer son score sur le validation set
#   - faire la moyenne des scores --> on a le score pour

# param à optimiser pour l'exemple : n_neighbors


# test pour n_neighbors = 1, avec 5 sous ensembles pour la cross validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(KNeighborsClassifier(n_neighbors=1), X_train, y_train, cv=5)
score = scores.mean()
#print('score moyen via cross_validation (sur train set) pour neighbors = 1 : ', score)


# tester plusieurs valeurs de n_neighbors --> automatisé dans sklearn : validation curve
from sklearn.model_selection import validation_curve
# --> nous donne, via cross validation, à la fois les scores du model sur le train_set, mais aussi sur le test_set
# validation_curve(model, X_train, y_train, 'hyperparametre', valeurs, cv=5)
neighbors_val = np.arange(1, 30)
train_scores, test_scores = validation_curve(KNeighborsClassifier(), X_train, y_train, 'n_neighbors', neighbors_val, cv=5)

# on a donc des tableaux avec une ligne pour chaque valeur d'hyperparametre, et une colonne pour chaque ensemble de la cross-validation
# on peut faire la moyenne de chaque ligne pour trouver le meilleur hyperparametre

plt.figure()
plt.plot(neighbors_val, train_scores.mean(axis=1), label='training')
plt.plot(neighbors_val, test_scores.mean(axis=1), label='testing')
plt.ylabel('score')
plt.xlabel('n_neighbors')
plt.legend()


# GRID SEARCH CV --> construire une grille de modèles avec tous les combinaisons d'hyperparametres
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': neighbors_val,
              'metric': ['euclidean', 'manhattan', 'minkowski'],
              'weights': ['uniform', 'distance']}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)

# On peut entrainer la grille comme on entrainerait un seul model
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

model = grid.best_estimator_

print('score sur le test set : ', model.score(X_test, y_test))


# Check les prédictions du model sur le set de test :
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, model.predict(X_test))
print('matrice de confusion : \n', matrix)



# 4. Check sur la courbe d'apprentissage du model pour voir si avoir plus de données serait bénéfique
from sklearn.model_selection import learning_curve
N, train_scores, test_scores = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)

print(N)

plt.figure()
plt.plot(N, train_scores.mean(axis=1), label='training')
plt.plot(N, test_scores.mean(axis=1), label='testing')
plt.ylabel('score')
plt.xlabel('Nb features used')
plt.legend()


# 4. utiliser le model via .predict
#predictions = model.predict(X)

def survie(model, pclass, sex, age):
    x = np.array([pclass, sex, age]).reshape(1,3)
    print(model.predict(x))
    print(model.predict_proba(x))
    
#survie(model, 3, 1, 28)
    




# graph des predictions
#plt.plot(X, predictions, c='r')
