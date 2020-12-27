import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


"""
This creates features for the CRT prediction:
 - Weighted Execution Time
 - Time from Trace Start
 - Case Remaining Time (Target Variable)
"""

# read data
df = pd.read_csv("./data/bpic2012.csv")

# select only 'complete' events
df = df.loc[df['lifecycle_type'] == 'complete']
df = df.reset_index(drop=True)   # reset index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['REG_DATE'] = pd.to_datetime(df['REG_DATE'])

# calculate num. of completed events by hour of day
hour_of_day = df['timestamp'].apply(lambda x: x.hour)
hour_of_day_counts = hour_of_day.value_counts().sort_index()

# calculate num. of completed events by day of week
day_of_week = df['timestamp'].apply(lambda x: x.weekday())
day_of_week_counts = day_of_week.value_counts().sort_index()
num_of_events_day_of_week = day_of_week_counts[day_of_week]

# calculate trace number and lengths
num_of_traces = len(np.unique(df['case_id']))
traces_lens = df.groupby('case_id').count().seq_of_event
traces_lens = np.array(traces_lens)


"""
[Feature] Weighted Execution Time for Each Event
    Case 1: only use 'complete' events and 'REG_DATE' attributes 
            as trace start time. ex) BPIC2012
    Case 2: there is no 'REG_DATE' attributes.
"""

# [Case 1]
hdc = pd.DataFrame()     # DataFrame for weights of hour of day
hdc['weight'] = hour_of_day_counts
min_weight = float(hdc.min() / hdc.max())  # for feature range (min_weight, 1) not (0, 1)
scaler = MinMaxScaler(feature_range=(min_weight, 1))
hdc = scaler.fit_transform(hdc).reshape(-1)   # calculate hours weights through the scaling

dwc = pd.DataFrame()     # DataFrame for weights of day of week
dwc['weight'] = day_of_week_counts
min_weight = float(dwc.min() / dwc.max())
scaler = MinMaxScaler(feature_range=(min_weight, 1))
dwc = scaler.fit_transform(dwc).reshape(-1)     # calculate days weights through the scaling

# calculate weighted execution time in minutes
# ex) 2011-10-01 12:17:08.924000 ~ 2011-10-08 16:32:00.886000
event_idx = 0
exe_time_lst = []

for i in range(num_of_traces):
    num_of_events_in_trace = traces_lens[i]
    for j in range(num_of_events_in_trace):
        cur_time = df.iloc[event_idx]['timestamp']
        if j == 0:
            prev_time = df.iloc[event_idx]['REG_DATE']
        else:
            prev_time = df.iloc[event_idx-1]['timestamp']
        # (1) residual time of first hour
        # ex) 2011-10-01 12:17:08.924000 ~ 2011-10-01 13:00:00.000000
        total_period = pd.date_range(start=str(prev_time.date()), end=str(cur_time.date()))
        hour = prev_time.hour
        hour_weight = hdc[hour]
        res_hour = 0
        if len(total_period) > 1 or prev_time.hour != cur_time.hour:
            # for cases longer than a day (period >= 2days)
            if len(total_period) >= 2:
                res_hour = 24 - prev_time.hour
            # for cases within a day (same day, different hour)
            # ex) 2011-10-01 12:17:00 ~ 2011-10-01 16:00:00
            else:
                res_hour = cur_time.hour - prev_time.hour
            res_min = 60 - prev_time.minute-1  # -1 min. because of res_sec
            res_sec = 60 - prev_time.second
            if res_min != 60:
                res_hour -= 1
        # for cases within an hour (same day and hour)
        # ex) 2011-10-01 12:17:00 ~ 2011-10-01 12:48:00
        else:
            res_min = cur_time.minute - prev_time.minute
            res_sec = cur_time.second - prev_time.second
        res_time_first_hour = ((res_sec / 60) + res_min) * hour_weight
        # (2) residual time of first day
        # ex) 2011-10-01 13:00:00.000000 ~ 2011-10-02 00:00:00.000000
        res_time_first_day = 0
        if res_hour > 0:
            for hour in range(prev_time.hour+1, prev_time.hour+1 + res_hour):
                hour_weight = hdc[hour]
                res_time_first_day += 60 * hour_weight
        # (3) time of interim period
        # ex) 2011-10-02 00:00:00.000000 ~ 2011.10.08 00:00:00.000000
        interim_period_time = 0
        if len(total_period) >= 3:
            for day in range(1, len(total_period) - 1):   # except first/last days
                day_weight = dwc[total_period[day].weekday()]
                interim_period_time += 1440 * day_weight
        # (4) residual time of last day
        # ex) 2011.10.08 00:00:00.000000 ~ 2011.10.08 16:00:00.000000
        res_time_last_day = 0
        # res_time_last_day is for cases longer than a day (period >= 2days)
        if len(total_period) >= 2:
            for hour in range(cur_time.hour):
                hour_weight = hdc[hour]
                res_time_last_day += 60 * hour_weight
        # (5) residual time of last hour
        # ex) 2011.10.08 16:00:00.000000 ~ 2011.10.08 16:32:00.886000
        hour_weight = hdc[cur_time.hour]
        res_time_last_hour = 0
        # res_time_last_hour cannot applied to the cases within an hour
        if len(total_period) >= 2 or cur_time.hour != prev_time.hour:
            res_time_last_hour = (cur_time.minute + (cur_time.second / 60)) * hour_weight
        # if event_idx == 80036:  # test case: 2, 3, 10, 107, 114, 11557, 80036
        #     print('prev_time = ' + str(prev_time))
        #     print('cur_time = ' + str(cur_time))
        #     print('res_time_first_hour = ' + str(res_time_first_hour))
        #     print('res_time_first_day = ' + str(res_time_first_day))
        #     print('interim_period_time = ' + str(interim_period_time))
        #     print('res_time_last_day = ' + str(res_time_last_day))
        #     print('res_time_last_hour = ' + str(res_time_last_hour))
        exe_time = res_time_first_hour + res_time_first_day + interim_period_time + \
            res_time_last_day + res_time_last_hour
        exe_time_lst.append(exe_time)
        event_idx += 1


df['weighted_execution_time'] = exe_time_lst

# [Feature] time from trace start
# [Feature] case remaining time (target variable, days)
tfts_lst = []   # list for the set of time from trace start
crt_lst = []    # list for the set of case remaining time
case_id = None
trace_start_time = None
trace_end_time = None

for idx, val in df.iterrows():
    if case_id != val['case_id']:
        case_id = val['case_id']
        trace_start_time = val['REG_DATE']
        lst_event = df.loc[df['case_id'] == case_id].iloc[-1]
        trace_end_time = lst_event['timestamp']
    cur_event_time = val['timestamp']
    time_from_trace_start = (cur_event_time - trace_start_time).total_seconds()
    tfts_lst.append(time_from_trace_start)
    case_remaining_time = (trace_end_time - cur_event_time).total_seconds()
    # seconds -> days
    case_remaining_time /= (60*60*24)
    crt_lst.append(case_remaining_time)


# [Case 2]
# [!] not implemented

# save the updated data frame
df['time_from_trace_start'] = tfts_lst
df['case_remaining_time'] = crt_lst

df.to_csv('./data/bpic2012_refined.csv')
