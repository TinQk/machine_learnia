# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 23:56:00 2020

@author: utilisateur
"""

from scipy import misc
import matplotlib.pyplot as plt

face = misc.face(gray=True)

plt.imshow(face, cmap=plt.cm.gray)
plt.show()


zoom = face[200:-200, 200:-200]
plt.imshow(zoom, cmap=plt.cm.gray)
plt.show()

face[face>150] = 255

plt.imshow(face, cmap=plt.cm.gray)
plt.show()
