# make a csv file from an xes file for the case remaining time prediction
# reference: https://github.com/opyenxes/OpyenXes/tree/2018-bpm-demo/example
# outputs: case information
#   (1) case_id
#   (2) case_execution_time
#   (3) case_execution_time_seconds
#   (4) num_of_events

import pandas as pd

from opyenxes.extension.XExtensionParser import XExtensionParser
from opyenxes.extension.XExtensionManager import XExtensionManager
from opyenxes.data_in.XUniversalParser import XUniversalParser

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
with open("./data/bpic2012.xes") as file:
    logs = XUniversalParser().parse(file)

log = logs[0]

drift_idx = 0
for trace in log:
    # the chosen case' id = 197611
    if trace.get_attributes()['concept:name'].get_value() == '197611':
        print('drift_idx = %d' % drift_idx)
        break
    drift_idx+=1

log = log[0:drift_idx]  # select the training set before the drift occurs

log_dict = {'case_id':[], 'case_execution_time':[], 'case_execution_time_seconds':[], 'num_of_events':[]}

# extract basic information and make a data frame
for trace in log:
    # extract the case id from the <trace> tag
    case_id = trace.get_attributes()['concept:name'].get_value()
    # calculates the case execution time (last event's timestamp - first event's timestamp)
    fst_event_time = trace[0].get_attributes()['time:timestamp'].get_value()
    lst_event_time = trace[-1].get_attributes()['time:timestamp'].get_value()
    case_exe_time = lst_event_time - fst_event_time
    case_exe_time_sec = case_exe_time.total_seconds()
    # appends the case id and case execution time
    log_dict['case_id'].append(case_id)
    log_dict['case_execution_time'].append(case_exe_time)
    log_dict['case_execution_time_seconds'].append(case_exe_time_sec)
    num_of_events = 0
    # counts the completed events of the case
    for event in trace:
        lifecycle_type = event.get_attributes()['lifecycle:transition'].get_value().lower()
        if lifecycle_type == 'complete':
            num_of_events += 1
    log_dict['num_of_events'].append(num_of_events)

log_df = pd.DataFrame(log_dict)
print(log_df)

log_df.to_csv('data/bpic2012_cet.csv', index_label='index')