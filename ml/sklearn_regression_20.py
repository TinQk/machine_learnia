# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:09:38 2020

@author: Quentin
"""

import numpy as np
import matplotlib.pyplot as plt


# Créer dataset d'exemple
np.random.seed(0)
m = 100 # creating 100 samples
X = np.linspace(0, 10, m).reshape(m, 1) # 1 colonnes, m features
y = X + np.random.randn(m, 1) # ajouter une distrib normale autour de 0 (du bruit)

plt.scatter(X, y)



# ML

# les x sont les "features" --> données en entrée, on peut en avoir plusieurs
# les y sont les données en sortie

# 1. sélectionner un model et préciser ses hyperparametres

from sklearn.linear_model import LinearRegression

model = LinearRegression()

# 2. entrainer un model sur les données X, y via .fit
model.fit(X,y)

# 3. évaluer le model via .score
print(model.score(X,y))


# 4. utiliser le model via .predict
predictions = model.predict(X)


# graph des predictions
plt.plot(X, predictions, c='r')
