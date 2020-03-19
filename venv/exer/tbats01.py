import pandas as pd
import matplotlib.pyplot as plt
from tbats import TBATS, BATS

# Import
df = pd.read_csv("../data/train_store_item_demand.csv")
df = df[(df['store'] == 1) & (df['item'] == 1)] # select data of item 1 in store 1
df = df.set_index('date')
y = df['sales']

y_to_train = y.iloc[:(len(y)-365)]
y_to_test = y.iloc[(len(y)-365):]

# Fit the model
estimator = TBATS(seasonal_periods=(7, 365.25))
model = estimator.fit(y_to_train)

# Forecast 365 days ahead
y_forecast = model.forecast(steps=365)

# Casting: ndarray -> Series
y_forecast = pd.Series(y_forecast, index=y_to_test.index)

# Plotting
plt.plot(y_to_train, label='Train')
plt.plot(y_to_test, label='Test')
plt.plot(y_forecast, label='TBATS')
plt.legend(loc='best')

plt.show()