# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:27:22 2020

@author: utilisateur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('titanic3.xls')

print(data.shape)
#print(data.columns)
#print(data.head())

data = data.drop(['name', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked', 'boat', 'body', 'home.dest'], axis=1)

print(data.shape)
print(data.columns)
print(data.head())

# montre les stats de base :
print(data.describe())

# on voit qu'on a pas les ages de tout le monde, on élimine ces lignes
data = data.dropna(axis=0)
print(data.shape)


# analyse classes des passagers
print(data['pclass'].value_counts())
data['pclass'].value_counts().plot.bar()
data['age'].hist()

# analyse des sexes
data.groupby(['sex']).mean()


# EXO regrouper ages en 4 catégories
#data.loc[data['age'] < 20, 'age'] = 0
#data.loc[(data['age'] >= 20) & (data['age'] < 30), 'age'] = 1
#data.loc[(data['age'] >= 30) & (data['age'] < 40), 'age'] = 2
#data.loc[data['age'] > 40] = 3

def category_ages(age):
    if age < 20:
        return 0
    elif age > 20 and age <= 30:
        return 1
    elif age > 30 and age <= 40:
        return 2
    elif age > 40:
        return 3    

data['age'] = data['age'].map(category_ages)

data.head()
