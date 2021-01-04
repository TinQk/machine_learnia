# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 13:09:15 2020

@author: Quentin
"""

import numpy as np

# =============================================================================
# A = np.array([1,2,3])
# 
# print(A.shape)
# 
# B = np.zeros((2,3))
# C = np.ones((5,4))
# 
# print(B)
# 
# np.random.seed(0)
# r = np.random.randn(3,4) # distribution normale autour de 0
# =============================================================================


def initialisation(m, n):
    matrice = np.random.randn(m, n)
    biais = np.ones((m,1))
    matrice = np.concatenate((matrice, biais), axis=1)
    return matrice


print(initialisation(4,5))