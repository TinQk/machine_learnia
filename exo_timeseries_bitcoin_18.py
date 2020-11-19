# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 14:33:34 2020

@author: Quentin
"""

# Turtle strategy :
"""
1. utiliser rolling() pour calculer:
    max sur les 28 derniers jours
    min sur les 28 derniers jours
2. Boolean indexing:
    Si 'Close' > max28 alors Buy = 1
    Si 'Close' < min28 alors Sell = -1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

bitcoin_data = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates=True)

#backup
bitcoin = bitcoin_data.copy() 

rolling = bitcoin.loc['2019','Close'].shift(1).rolling(window=28, center=False)


# Création d'une colonne 'Buy' qui vaut 1 quand il faut acheter, 0 sinon
bitcoin['Buy'] = np.zeros(len(bitcoin))

# Création d'une colonne 'Sell' qui vaut 1 quand il faut vendre, 0 sinon
bitcoin['Sell'] = np.zeros(len(bitcoin))

# Création d'une colonne 'Max28'
bitcoin['Max28'] = rolling.max()

# Création d'une colonne 'Min28'
bitcoin['Min28'] = rolling.min()


bitcoin.loc[bitcoin['Close'] > bitcoin['Max28'], 'Buy'] = 1
bitcoin.loc[bitcoin['Close'] < bitcoin['Min28'], 'Sell'] = -1


# Prints

print(bitcoin)

print(rolling.mean()['2019-04-10'])


# Tracés

plt.figure('bitcoin 2018')

bitcoin.loc['2019', 'Close'].plot(label='bitcoin close')

#rolling.mean().plot(label='moving average', lw=2, ls='-.')
bitcoin.loc['2019', 'Max28'].plot(label='Max28')
bitcoin.loc['2019', 'Min28'].plot(label='Min28')

plt.legend()

#plt.draw()


plt.figure('Buy and Sell')

bitcoin.loc['2019', 'Buy'].plot(label='Buy')
bitcoin.loc['2019', 'Sell'].plot(label='Sell')

plt.legend()
