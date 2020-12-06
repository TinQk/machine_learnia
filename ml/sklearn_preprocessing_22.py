# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:56:22 2020

@author: Quentin
"""

import numpy as np

X = np.array([['Chat', 'Poils'],
              ['Chien','Poils'],
              ['Chat','Poils'],
              ['Oiseau','Plumes']])

y = np.array(['Chat',
              'Chien',
              'Chat',
              'Oiseau'])

z = np.array(['Chat',
              'Chien',
              'Chat',
              'Oiseau',
              'Chien',
              'Chat',])

# Transformer : applique une chaine de prétraitement de données, pour que toutes les nouvelle données arrivent au bon format
# Module preprocessing --> contient pleins de transformers différents


### 1) les Encoder : transformer les classes en nombres

# Encodage classique :
    
    # LabelEncoder --> pour les y (une colonne)
from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()

# adapter la fonction de transformation a nos données
encoder.fit(y) 

# l'utiliser sur nos données ou n'importe quel jeu de données similaire
y_encoded = encoder.transform(y)
print("LabelEncoder y")
print(y_encoded)
# Pour aller plus vite : encoder.fit_transform(y)

# On peut aussi désencoder
print(encoder.inverse_transform(np.array([0,2,0,2,2,1])))
  
    # OrdinalEncoder --> pour les X (plusieurs colonnes)
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
# encoder.fit(X)
# encoder.transform(X)
print("OrdinalEncoder X")
print(encoder.fit_transform(X))


# Encodage "OneHot" --> crée une colonne par catégorie et enregistre en binaire
   
     # LabelBinarizer --> format "Sparse Matrix" (matrice vide)
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
print("LabelBinarizer y")
print(encoder.fit_transform(y))

    # MultiLabelBinarizer --> idem sur X

    # OneHotEncoder : dimension par catégorie (2D array) au format "CSR"
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
print("OneHotEncoder X")
print(encoder.fit_transform(X))




### 2) : Normalisation --> Mettre les datas sur la même échelle pour descente de gradient, calcul de distance etc..

# !!! Normalizer() transforme les lignes au lieu des colonnes --> piege

X = np.array([[70],
             [80],
             [120]])

# Techniques 1 et 2 : MinMaxScaler et StandardScaler, mais TRES SENSIBLES aux valeurs abérantes

# Algo MinMax (X-Xmin)/(Xmax-Xmin) --> ce que j'ai fait sur leekwars, tout entre 0 et 1
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print("MinMaxScaler X")
print(scaler.fit_transform(X))

# StandardScaler (X - Xmoy)/Xecart_type
# --> moyenne = 0 et écart type = 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
print("StandardScaler X")
print(scaler.fit_transform(X))

# Technique 3 : RobustScaler --> soustrait la médiane plutot que la moyenne et soustrait l'interquartile --> moins sensible aux valeurs abérantes
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
print("RobustScaler X")
print(scaler.fit_transform(X))




### Autres transformer :
    
# polynomial feature --> crée des polygones a partir des datas
# power transform --> donne des datas normales ou gaussiennes

# discretisation des données :
    # Binarizer
    # Kbindescretizer
    
# Transformer personnalisable : FunctionTransformer
    

# Imputation --> remplacer certaines valeurs manquantes par des stats


# Selection --> selectionne les variables les plus utiles au développement d'un modèle


# extraction --> recup de nouvelles variables "cachées" du dataset

