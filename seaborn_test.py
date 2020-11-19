# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 14:43:39 2020

@author: Quentin
"""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

iris = sns.load_dataset("iris")

print(iris.head())

# premier graph intéressant sur un dataset pour connaitre les relations entre les données :
sns.pairplot(iris, hue='species')


plt.figure()
sns.catplot(x='species', y='sepal_length', data=iris)

plt.figure()
sns.boxplot(x='species', y='sepal_length', data=iris)



# transformer les species en catégories si besoin
iris['species'] = iris['species'].astype('category').cat.codes
print(iris.head())


plt.figure()
sns.jointplot(x='species', y='sepal_length', data=iris)

plt.figure()
sns.heatmap(iris)

plt.figure()
sns.distplot(iris)


