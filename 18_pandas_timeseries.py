import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

bitcoin = pd.read_csv('BTC-EUR.csv', index_col='Date', parse_dates=True)

print(bitcoin.head())


bitcoin['2019']['Close'].plot()

bitcoin['2019']['Close'].resample('M').mean().plot()

plt.show()

tab = bitcoin['2019']['Close'].resample('M').agg(['mean', 'std', 'min', 'max'])
print(tab)