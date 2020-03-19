# make a csv file from an xes file for storing general information of event logs
# outputs
#   (1) case_id (trace, concept:name)
#   (2) extra trace's attributes
#   (3) seq_of_event (by counting events in trace)
#   (4) activity_type (event, concept:name)
#   (5) lifecycle_type (event, lifecycle:transition)
#   (6) timestamp (event, time:timestamp)
#   (7) performer (event, org:resource)
#   (8) extra event's attributes

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
trace_dict = {'case_id':[]}
event_dict = {'case_id':[], 'seq_of_event':[], 'activity_type':[],
              'resource_id':[], 'lifecycle_type':[], 'timestamp':[]}

basic_attr_lst = ['concept:name', 'org:resource', 'lifecycle:transition', 'time:timestamp']
trace_ext_attr_lst = []     # for extra attributes in traces
event_ext_attr_lst = []     # for extra attributes in events

# find extra attributes then add to dictionary
trace = log[0]
for key, value in trace.get_attributes().items():
    if key not in basic_attr_lst:
        trace_dict.update({key: []})
        trace_ext_attr_lst.append(key)

event = trace[0]
for key, value in event.get_attributes().items():
    if key not in basic_attr_lst:
        event_dict.update({key: []})
        event_ext_attr_lst.append(key)

# extract attributes in event log
# case1: all extra attributes are mandatory (equal length)
for trace in log:
    i = 1   # counts events in a trace
    seq_of_event = 1    # counts complete events in a trace
    # extract the case id (common key between trace_dict and event_dict)
    case_id = trace.get_attributes()['concept:name'].get_value()
    trace_dict['case_id'].append(case_id)
    for key in trace_ext_attr_lst:
        trace_dict[key].append(trace.get_attributes()[key].get_value())
    for event in trace:
        # add extra attribute values from trace
        event_dict['case_id'].append(case_id)
        # extract the sequence of the event
        event_dict['seq_of_event'].append(seq_of_event)
        # extract the activity type of the event
        activity_type = event.get_attributes()['concept:name'].get_value()
        event_dict['activity_type'].append(activity_type)
        # extract the resource id of the event (can be an empty value)
        try:
            resource_id = event.get_attributes()['org:resource'].get_value()
            event_dict['resource_id'].append(resource_id)
        except KeyError:
            print("There is no 'org:resource' attribute in %dth event of trace '%s'" % (i, case_id))
            event_dict['resource_id'].append(None)
        # extract the lifecycle type of the event
        lifecycle_type = event.get_attributes()['lifecycle:transition'].get_value().lower()
        # for counting complete events
        if lifecycle_type == 'complete':
            seq_of_event += 1
        event_dict['lifecycle_type'].append(lifecycle_type)
        # extract the timestamp of the event
        timestamp = event.get_attributes()['time:timestamp'].get_value()
        event_dict['timestamp'].append(timestamp)
        # extract the event's extra attribute values
        for key in event_ext_attr_lst:
            event_dict[key].append(event.get_attributes()[key].get_value())
        i += 1

# case2: some extra attributes are optional (not equal length)
# [!] we can write codes of this part when we meet corresponding situation.

# merge trace_dict and event_dict into dataframe
trace_df = pd.DataFrame(trace_dict)
event_df = pd.DataFrame(event_dict)
df = pd.merge(trace_df, event_df, on='case_id')
df.head()

# outputs the csv file
output_file = "./data/bpic2012.csv"
df.to_csv(output_file, index=False)
