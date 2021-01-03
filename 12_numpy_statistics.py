# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:01:56 2020

@author: utilisateur
"""

import numpy as np

# =============================================================================
# A = np.array([[5,0,3],[3,7,9]])
# 
# coef_corr = np.corrcoef(A)
# 
# values, counts = np.unique(A, return_counts=True) # ressort deux tableaux
# 
# print(values[counts.argsort()]) # retri de celui qui apparait le moins a celui qui apparait le plus
# 
# =============================================================================

np.random.seed(0)

A = np.random.randint(0,100, [10,5])

print(A)

print(A.mean(axis=0))

D = (A - A.mean(axis=0)) / A.std(axis=0)
    
print(D)

