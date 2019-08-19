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
print("[info] XExtensionParser starts the extension parsing");
meta_general = XExtensionParser().parse("http://www.xes-standard.org/meta_general.xesext")
meta_concept = XExtensionParser().parse("http://www.xes-standard.org/meta_concept.xesext")
meta_time = XExtensionParser().parse("http://www.xes-standard.org/meta_time.xesext")
ext_concept = XExtensionParser().parse("http://www.xes-standard.org/concept.xesext")
ext_time = XExtensionParser().parse("http://www.xes-standard.org/time.xesext")
ext_lifecycle = XExtensionParser().parse("http://www.xes-standard.org/lifecycle.xesext")
print("[info] XExtensionParser completed the extension parsing");

# Then we register the new extension
print("[info] XExtensionManager starts the registration of extensions");
XExtensionManager().register(meta_general)
XExtensionManager().register(meta_concept)
XExtensionManager().register(meta_time)
XExtensionManager().register(ext_concept)
XExtensionManager().register(ext_time)
XExtensionManager().register(ext_lifecycle)
print("[info] XExtensionManager completed the registration of extensions");

# Now we can parse
with open("./data/bpic2012_t100.xes") as file:
    logs = XUniversalParser().parse(file)


log = logs[0]
log_dict = {'case_id':[], 'concept:name':[], 'lifecycle:transition':[],
            'time:timestamp':[], 'case remaining time':[]}

for trace in log:
    # extract the case id from the <trace> tag
    case_id = trace.get_attributes()['concept:name'].get_value()
    for event in trace:
        # except the event whose 'lifecycle:transition' is 'schedule'
        if event.get_attributes()['lifecycle:transition'].get_value().lower() != 'schedule':
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
                    log_dict['case remaining time'].append(lst_event_time - cur_event_time)
                elif value.get_key() == 'lifecycle:transition':
                    log_dict['lifecycle:transition'].append(value.get_value())

log_df = pd.DataFrame(log_dict)
log_df.get_value(self, 0)
print(log_df)

output_file = "./data/output.csv"
log_df.to_csv(output_file, index_label='index')

# make a dataframe for the remaining time of each activity types
# option 1: get info. from meta tags (not applicable in fragmented logs)
# <insert codes...>
# option 2: calculate and measure manually

