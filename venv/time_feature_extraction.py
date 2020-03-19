
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# [1. Preprocess]

# reads data
df = pd.read_csv("./data/bpic2012.csv")
print(df)

# selects only 'complete' events
df = df.loc[df['lifecycle_type'] == 'complete']
df = df.reset_index()   # reset index
df_time = pd.to_datetime(df['timestamp'], utc=True)

# feature: num. of completed events by day of week
day_of_week = df_time.dt.dayofweek
day_of_week_counts = day_of_week.value_counts().sort_index()
bars = ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun')
y_pos = np.arange(len(bars))
plt.bar(y_pos, day_of_week_counts)
plt.xticks(y_pos, bars)
plt.show()

# feature: num. of completed events by hour of day
hour_of_day = df_time.dt.hour
hour_of_day_counts = hour_of_day.value_counts().sort_index()
bars = ('0am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am',
        '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm')
y_pos = np.arange(len(bars))
plt.bar(y_pos, hour_of_day_counts)
plt.xticks(y_pos, bars)
plt.show()

# feature: weights of hour of day

# feature: weights of day of week

