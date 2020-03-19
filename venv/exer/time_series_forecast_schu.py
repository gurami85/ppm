import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.api as sm
from tbats import TBATS

# import data
df = pd.read_csv('../data/commuters_train.csv')
# print data
print(df.head())
print(df.tail())

# subsetting the dataset from (Aug. 2012 ~ Dec. 2013)
# index 11856 marks the end of year 2013
df = pd.read_csv('../data/commuters_train.csv', nrows=11856)
print(df.head())
print(df.tail())

# creating train and test data for modeling.
# training data: first 14 months (Aug. 2012 ~ Oct. 2013)
# test data: last 2 months (Nov. 2013 ~ Dec. 2013)
# index 10392 marks the end of Oct. 2013
train = df[0:10392]
test = df[10392:]

# Aggregating the dataset at daily level (downsampling)
# to_dateTime: Timestamp (%d-%m-%Y %H:%M) -> Datetime (%Y-%m-%d %H:%M:%s)
df.Timestamp = pd.to_datetime(df.Datetime, format='%d-%m-%Y %H:%M')
df.index = df.Timestamp
df = df.resample('D').mean()

train.Timestamp = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index = train.Timestamp
train = train.resample('D').mean()

test.Timestamp = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
test.index = test.Timestamp
test = test.resample('D').mean()

# plotting data
train.Count.plot(figsize=(15,8), title='Daily Ridership', fontsize=14)
test.Count.plot(figsize=(15,8), title='Daily Ridership', fontsize=14)
plt.show()

# (1) Start with a Naive Approach
# simply take the last day value as prediction, y^hat_{t+1} = y_t

dd = np.asarray(train.Count)
y_hat = test.copy()
# y_hat: predicted values
y_hat['naive'] = dd[-1]     # [-1] : last element

plt.figure(figsize=(16, 8))
plt.plot(train.index, train['Count'], label='Train')
plt.plot(test.index, test['Count'], label='Test')
plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
plt.show()

# measures RMSE values of the model
rms = sqrt(mean_squared_error(test.Count, y_hat.naive))
print(rms)

# (2) Simple Average
# predicted values = the avg. of all previously observed points
y_hat['avg_forecast'] = train['Count'].mean()   # mean() returns the average

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat['avg_forecast'], label='Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Count, y_hat.avg_forecast))
print(rms)

# (1) Moving Average
y_hat['moving_avg_forecast'] = train['Count'].rolling(60).mean().iloc[-1]

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat['moving_avg_forecast'], label='Moving Average Forecast')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Count, y_hat.moving_avg_forecast))
print(rms)

# (4) Simple Exponential Smoothing
# using weighted averages where the decrease exponentially as observations come from further in the past
# smoothing parameter (level), 0 <= a <= 1
ses_predictor = SimpleExpSmoothing(np.asarray(train['Count'])).fit(smoothing_level=0.6, optimized=False)
y_hat['simple_exp_smoothing'] = ses_predictor.forecast(len(test))

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat['simple_exp_smoothing'], label="Simple Exponential Smoothing")
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Count, y_hat.simple_exp_smoothing))
print(rms)

# (2) Holt's Linear Trend Method
# it is an extension of the simple exponential smoothing to allow forecasting of data with a trend
# Make predictions from the Holt's linear trend model
holt_linear_predictor = Holt(np.asarray(train['Count'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
y_hat['Holt_linear'] = holt_linear_predictor.forecast(len(test))

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat['Holt_linear'], label="Holt\'s Linear Trend")
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Count, y_hat.Holt_linear))
print(rms)

# (6) Holt-Winters' Seasonal Method
# adding the 'seasonality' in data
# This method = exponential smoothing to seasonal components in addition to level and trend
# length of the seasonality = 7 days (1 week)
# method type: addictive(O), multiplicative(X)
holt_winter_predictor = ExponentialSmoothing(np.asarray(train['Count']), seasonal_periods=7, trend='add', seasonal='add').fit()
y_hat['Holt_Winter'] = holt_winter_predictor.forecast(len(test))

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat['Holt_Winter'], label="Holt-Winters\' Seasonal Method (Addictive)")
plt.legend(loc='best')
plt.show()
rms = sqrt(mean_squared_error(test.Count, y_hat.Holt_Winter))
print(rms)


# (1) ARIMA (Autoregressive Integrated Moving Average) Method
# adding the correlation between 'trend' and 'seasonality'

arima_predictor = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),
                                            seasonal_order=(0, 1, 1, 7)).fit()
y_hat['ARIMA'] = arima_predictor.predict(start="2014-06-30", end="2014-07-06", dynamic=True)

plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat['ARIMA'], label='ARIMA')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(test.Count, y_hat.SARIMA))
print(rms)

# (2) TBATS Method
# modelling the multiple seasonality

tbats_predictor = TBATS(seasonal_periods=(24, 168))
model = tbats_predictor.fit(train)

# forecasting
y_hat = model.forecast(steps=len(test))

# transformation: ndarray -> Series
y_hat = pd.Series(y_hat, index=test.index)

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat['TBATS'], label='TBATS')
plt.legend(loc='best')
plt.show()



