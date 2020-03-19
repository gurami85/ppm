import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from numpy import log
from pmdarima.arima.utils import ndiffs

# Import
df = pd.read_csv("../data/wwwusage.csv", names=['value'], header=0)

# ADF (Augmented Dickey Fuller) test: check if the series is stationary
result = adfuller(df.value.dropna())
print('ADF Statistics: %f' % result[0])
print('p-value: %f' % result[1])

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value)
axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.value.diff())
axes[1, 0].set_title("1st Order Differencing")
plot_acf(df.value.diff().dropna(), ax=axes[1,1])

# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff())
axes[2, 0].set_title("2nd Order Differencing")
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

# ADF & KPSS & PP tests #
y = df.value

# ADF test
ndiffs(y, test='adf')   # 2
# KPSS test
ndiffs(y, test='kpss')  # 0
# PP test
ndiffs(y, test='pp')    # 2

