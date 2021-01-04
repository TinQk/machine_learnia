# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:05:59 2020

@author: Quentin
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data  # les objets avec des datas
y = iris.target # notre classement qu'on doit viser
names = list(iris.target_names)


# graph 2D (deux params)
plt.scatter(x[:,0], x[:,1], c=y, alpha=0.5)


# =============================================================================
# #graph 3D
# from mpl_toolkits.mplot3d import Axes3D 
# 
# ax = plt.axes(projection='3d')
# ax.scatter(x[:,0], x[:,1], x[:,2], c=y, alpha=0.5)
# 
# 
# # function
# f = lambda x, y: np.sin(x) + np.cos(x+y)
# 
# X = np.linspace(0,5,100)
# Y = np.linspace(0,5,100)
# 
# X, Y = np.meshgrid(X, Y)
# 
# Z = f(X, Y)
# ax = plt.axes(projection='3d')
# ax.plot_surface(X,Y,Z, cmap='plasma')
# 
# =============================================================================


# hist
plt.figure()
plt.hist(x[:,0])
plt.show()

# hist 2d
plt.hist2d(x[:,0],x[:,1], cmap='Blues')
plt.colorbar()


# IM SHOW ex : matrice de correlation
plt.figure()
plt.imshow(np.corrcoef(x.T), cmap='Blues')
plt.colorbar()
plt.show()