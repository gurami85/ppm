# make a csv file from an xes file for the remaining time prediction
# reference: https://github.com/opyenxes/OpyenXes/tree/2018-bpm-demo/example

import pandas as pd
import numpy as np
import os, sys

from opyenxes.extension.XExtensionParser import XExtensionParser
from opyenxes.extension.XExtensionManager import XExtensionManager
from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.model.XEvent import XEvent
from opyenxes.model.XTrace import XTrace
from opyenxes.model.XAttributeBoolean import XAttributeBoolean
from opyenxes.model.XAttributeCollection import XAttributeCollection
from opyenxes.model.XAttributeContainer import XAttributeContainer
from opyenxes.model.XAttributeContinuous import XAttributeContinuous
from opyenxes.model.XAttributeDiscrete import XAttributeDiscrete
from opyenxes.model.XAttributeID import XAttributeID
from opyenxes.model.XAttributeList import XAttributeList
from opyenxes.model.XAttributeLiteral import XAttributeLiteral
from opyenxes.model.XAttributeMap import XAttributeMap
from opyenxes.model.XAttributeTimestamp import XAttributeTimestamp

# We must parse the new extension, can be the link or the xml file
print("[info] XExtensionParser starts the extension parsing")
meta_general = XExtensionParser().parse("http://www.xes-standard.org/meta_general.xesext")
meta_concept = XExtensionParser().parse("http://www.xes-standard.org/meta_concept.xesext")
meta_time = XExtensionParser().parse("http://www.xes-standard.org/meta_time.xesext")
ext_concept = XExtensionParser().parse("http://www.xes-standard.org/concept.xesext")
ext_time = XExtensionParser().parse("http://www.xes-standard.org/time.xesext")
ext_lifecycle = XExtensionParser().parse("http://www.xes-standard.org/lifecycle.xesext")
print("[info] XExtensionParser completed the extension parsing")

# Then we register the new extension
print("[info] XExtensionManager starts the registration of extensions")
XExtensionManager().register(meta_general)
XExtensionManager().register(meta_concept)
XExtensionManager().register(meta_time)
XExtensionManager().register(ext_concept)
XExtensionManager().register(ext_time)
XExtensionManager().register(ext_lifecycle)
print("[info] XExtensionManager completed the registration of extensions")

# Now we can parse
with open("./data/bpic2012_t100.xes") as file:
    logs = XUniversalParser().parse(file)

log = logs[0]
log_dict = {'case_id':[], 'concept:name':[], 'lifecycle:transition':[],
            'time:timestamp':[], 'case_remaining_time':[], 'num_of_antecedent_events':[]}

# extract basic information and make a data frame
for trace in log:
    # extract the case id from the <trace> tag
    case_id = trace.get_attributes()['concept:name'].get_value()
    # variable for counting num. of antecedent events (num. of completed activities in a case)
    num_of_antecedent_events = 0
    for event in trace:
        # except the event whose 'lifecycle:transition' is 'schedule'
        lifecycle_type = event.get_attributes()['lifecycle:transition'].get_value().lower()
        if lifecycle_type != 'schedule':
            log_dict['case_id'].append(case_id)
            # items() returns <key, value> pairs of the attributes in the event
            attrs = event.get_attributes().items()
            for key, value in attrs:
                if value.get_key() == 'concept:name':
                    log_dict['concept:name'].append(value.get_value())
                elif value.get_key() == 'time:timestamp':
                    cur_event_time = value.get_value()
                    lst_event_time = trace[len(trace)-1].get_attributes()['time:timestamp'].get_value()
                    # append the timestamp of current event
                    log_dict['time:timestamp'].append(cur_event_time)
                    # append the case remaining time (CRT)
                    # CRT is calculated as difference between current and last events' timestamps
                    log_dict['case_remaining_time'].append(lst_event_time - cur_event_time)
                elif value.get_key() == 'lifecycle:transition':
                    log_dict['lifecycle:transition'].append(value.get_value())
            log_dict['num_of_antecedent_events'].append(num_of_antecedent_events)
            if lifecycle_type == 'complete':
                num_of_antecedent_events += 1

log_df = pd.DataFrame(log_dict)
print(log_df)

log_df.loc[log_df['lifecycle:transition'] == 'complete', ['num_of_antecedent_events']]

output_file = "./data/log_info.csv"
log_df.to_csv(output_file, index_label='index')

# make a data frame for the remaining time of each activity types
# option 1: get info. from meta tags (not applicable in fragmented logs)
# <insert codes...>
# option 2: calculate and measure manually

# measure frequencies for each activity (concept:name)
# measure total sum of case remaining time for each activity
freq_dict = dict()
remaining_time_sum_dict = dict()
for idx, row in log_df.iterrows():
    concept_name = row['concept:name']
    case_remaining_time = row['case_remaining_time']
    # include only events whose 'lifecycle' is 'complete'
    if row['lifecycle:transition'].lower() != 'complete':
        continue
    if concept_name in freq_dict:
        freq_dict[concept_name] += 1
        remaining_time_sum_dict[concept_name] += case_remaining_time
    else:
        freq_dict[concept_name] = 0
        remaining_time_sum_dict[concept_name] = case_remaining_time

print(freq_dict)
print(remaining_time_sum_dict)

# calculate averages of case remaining time (sec.) for each activity
avg_crt_dict = dict()
for key in freq_dict:
    # get average of CRT
    # total_seconds() converts Timedelta to seconds
    avg_crt = (remaining_time_sum_dict[key] / freq_dict[key]).total_seconds()
    avg_crt_dict[key] = avg_crt

print(avg_crt_dict)

# make a training dataset from ingredients (log_df, avg_crt_dict)
training_dict = {'num_of_antecedent_events': [], 'avg_of_case_remaining_time': []}
for idx, row in log_df.iterrows():
    concept_name = row['concept:name']
    # 1st feature: NAE (Num. of Antecedent Events)
    nae = row['num_of_antecedent_events']
    training_dict['num_of_antecedent_events'].append(nae)
    # 2nd feature: Avg. CRT (Average of Case Remaining Time)
    avg_crt = avg_crt_dict[concept_name]
    training_dict['avg_of_case_remaining_time'].append(avg_crt)

training_df = pd.DataFrame(training_dict)
output_file = "./data/training_data.csv"
training_df.to_csv(output_file, index=False)
