import pandas as pd
import matplotlib.pyplot as plt
from tbats import TBATS, BATS

# Import
df = pd.read_csv("../data/a10.csv", parse_dates=['date'], index_col='date')

train = df[:-12]
test = df[-12:]

# Fit the model
estimator = TBATS(seasonal_periods=(12, 24))
model = estimator.fit(train)

# Forecast 12 months ahead
y_hat = model.forecast(steps=12)

# Casting: ndarray -> Series
y_hat = pd.Series(y_hat, index=test.index)

# Plotting
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(y_hat, label='TBATS')
plt.legend(loc='best')

plt.show()