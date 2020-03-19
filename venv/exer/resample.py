from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

def parser(x):
    return datetime.strptime('190' + x, '%Y-%m')

series = read_csv('../data/shampoo.csv', header=0, parse_dates=[0], index_col=0,
                  squeeze=True, date_parser=parser)

# original data
print(series.head())
series.plot()
pyplot.show()

# upsampling: month -> days by interpolation
upsampled = series.resample('D')    # 'D' means unit of day
interpolated = upsampled.interpolate(methods='linear') # interpolation fills out empty values between data
print(interpolated.head(32))
interpolated.plot()
pyplot.show()

# downsampling: 3 months -> 1 quarter
downsampled = series.resample('Q')  # 'Q' means unit of quarter
quarterly_mean_sales = downsampled.mean()   # make mean values of the observations by quarter
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()
pyplot.show()