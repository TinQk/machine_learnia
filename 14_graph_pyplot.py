# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:57:24 2020

@author: utilisateur
"""

import numpy as np

x = np.linspace(0, 2, 10)
y = x**2

print(x)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8)) # figure vide : espace de travail

plt.scatter(x, y, label='quadratique')
plt.plot(x, x**3, label='cubique')

plt.title('figure 1 XD')
plt.xlabel('axe x') 
plt.ylabel('axe y')
plt.legend()

plt.show()

plt.savefig('fig14.png')

# on peut avoir plusieurs graphiques sur une meme figure : subplot 

dataset = {f"experience {i+1}": np.random.randn(100) for i in range(4)}

def graphique(dataset):   
    for i in range(4):
        plt.subplot(4, 1, i+1)
        x = np.linspace(1,100, 100)
        y = dataset[f'experience {i+1}']
        plt.plot(x,y)
    
    plt.show
    
graphique(dataset)