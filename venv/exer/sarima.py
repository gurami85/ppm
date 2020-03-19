import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse

# Import
data = pd.read_csv("../data/a10.csv", parse_dates=['date'], index_col='date')

fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(data[:], label='Original Series')
axes[0].plot(data[:].diff(1), label='Usual Differencing')
axes[0].set_title("Usual Differencing")
axes[0].legend(loc="upper left", fontsize=10)

# Plotting differencing effects: regular vs seasonal
axes[1].plot(data[:], label='Original Series')
axes[1].plot(data[:].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc="upper left", fontsize=10)
plt.suptitle('a10 - Drug Sales', fontsize=16)
plt.show()

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(data, start_p=1, start_q=1, test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True,
                       d=None, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

smodel.summary()

# Forecast
n_periods = 24
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(data)
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)

plt.title("SARIMA - Final Forecast of a10 - Drug Sales")
plt.show()

# SARIMAX: SARIMA + exogeneous variables
# compute seasonal index (multiplicative), -36: recent 3 years
result_mul = seasonal_decompose(data['value'][-36:], model='multiplicative', extrapolate_trend='freq')
seasonal_index = result_mul.seasonal[-12:].to_frame()
seasonal_index['month'] = pd.to_datetime(seasonal_index.index).month

# merge with the base data
data['month'] = data.index.month
df = pd.merge(data, seasonal_index, how='left', on='month')
df.columns = ['value', 'month', 'seasonal_index']
df.index = data.index   # reassign the index.

## make a SARIMAX model
# exogenous: variable from external source
# m: the num. of periods in each season
# d: the order of first-differencing. if d=None, the runtime could be significantly longer

sxmodel = pm.auto_arima(df[['value']], exogenous=df[['seasonal_index']], start_p=1, start_q=1,
                        test='adf', max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=None,
                        D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

# Forecast
n_periods = 24
fitted, confint = sxmodel.predict(n_periods=n_periods,
                                  exogenous=np.tile(seasonal_index.value, 2).reshape(-1,1),
                                  return_conf_int=True)

index_of_fc = pd.date_range(data.index[-1], periods=n_periods, freq='MS')

# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:,0], index=index_of_fc)
upper_series = pd.Series(confint[:,1], index=index_of_fc)

# Plot
plt.plot(data['value'])
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)

plt.title("SARIMAX Forecast of a10 - Drug Sales")
plt.show()