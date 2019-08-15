# make a csv file from an xes file for the remaining time prediction
# reference: https://github.com/opyenxes/OpyenXes/tree/2018-bpm-demo/example


# %load /home/jonathan/.ipython/profile_default/startup/01-setup.py
# start up settings for jupyter notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

plt.style.use('ggplot')
plt.rcParams['font.size'] = 15.0
plt.rcParams['axes.labelsize'] = 15.0
plt.rcParams['xtick.labelsize'] = 15.0
plt.rcParams['ytick.labelsize'] = 15.0
plt.rcParams['legend.fontsize'] = 15.0

# set the max column width
pd.options.display.max_colwidth = 1000

# to avoid have warnings from chained assignments
pd.options.mode.chained_assignment = None

# We must parse the new extension, can be the link or the xml file
meta_general = XExtensionParser().parse("http://www.xes-standard.org/meta_general.xesext")
meta_concept = XExtensionParser().parse("http://www.xes-standard.org/meta_concept.xesext")
meta_time = XExtensionParser().parse("http://www.xes-standard.org/meta_time.xesext")
ext_concept = XExtensionParser().parse("http://www.xes-standard.org/concept.xesext")
ext_time = XExtensionParser().parse("http://www.xes-standard.org/time.xesext")
ext_lifecycle = XExtensionParser().parse("http://www.xes-standard.org/lifecycle.xesext")

# Then we register the new extension
XExtensionManager().register(meta_general)
XExtensionManager().register(meta_concept)
XExtensionManager().register(meta_time)
XExtensionManager().register(ext_concept)
XExtensionManager().register(ext_time)
XExtensionManager().register(ext_lifecycle)

# Now we can parse
# with open("./data/review_example_large.xes") as file:
with open("./data/BPIC2012.xes.gz") as file:
    log = XUniversalParser().parse(file)

CASEID = 'caseid'


class XLog2df:
    def __init__(self):
        self.__event_ind = 0
        self.__trace_ind = 0
        self.__event_df_dict = dict()
        self.__trace_df_dict = dict()

    def parse_xattribute(self, xattrib):
        is_list = isinstance(xattrib, XAttributeList)
        is_container = isinstance(xattrib, XAttributeContainer)

        if is_list or is_container:
            return None, None, None
        else:
            return xattrib.get_key(), xattrib.get_value(), xattrib.get_extension()

    def parse_xattribute_dict(self, xattribs):
        return {key: self.parse_xattribute(val)[1] for key, val in xattribs.items()}

    def xevents2df(self, events, caseid):
        event_df_dict = dict()

        for event in events:
            # assert isinstance(event, XEvent)
            attrib_dict = self.parse_xattribute_dict(event.get_attributes())

            # add caseid
            attrib_dict[CASEID] = caseid
            event_df_dict[self.__event_ind] = attrib_dict
            self.__event_ind += 1

        return event_df_dict

    def xtraces2df(self, traces):
        trace_df_dict = dict()

        for trace in traces:
            attrib_dict = dict(trace.get_attributes())
            attrib_dict = self.parse_xattribute_dict(attrib_dict)
            trace_df_dict[self.__trace_ind] = attrib_dict
            self.__trace_ind += 1

        return trace_df_dict

    def xlog2df(self, xlog):
        trace_df_dict = self.xtraces2df(xlog)
        event_df_dict = dict()

        for trace in xlog:
            caseid = trace.get_attributes()['concept:name'].get_value()
            event_df_dict_i = self.xevents2df(trace, caseid)
            event_df_dict.update(event_df_dict_i)

        trace_df = pd.DataFrame.from_dict(trace_df_dict, 'index')
        event_df = pd.DataFrame.from_dict(event_df_dict, 'index')

        # prefix trace attributes with "trace:" and event attributes with "event:"
        trace_df.columns = ['trace:{}'.format(val) for val in trace_df.columns]
        event_df_columns = []

        for val in event_df.columns:
            renamed = 'event:{}'.format(val)
            if val != CASEID:
                event_df_columns.append(renamed)
            else:
                event_df_columns.append(val)

        event_df.columns = event_df_columns

        # merge trace_df and event_df on caseid
        trace_df[CASEID] = trace_df['trace:concept:name']

        # key column needs to be string type
        trace_df[CASEID] = trace_df[CASEID].astype(str)
        event_df[CASEID] = event_df[CASEID].astype(str)

        print(trace_df.head())
        print('---')
        print(event_df.head())

        print('Trace df columns: {}'.format(trace_df.columns))
        print('Event df columns: {}'.format(event_df.columns))

        merged_df = pd.merge(trace_df, event_df, on=CASEID)

        return merged_df


converter = XLog2df()
event_row_df = converter.xlog2df(log)

# event_row_df['event:org:resource'] = event_row_df['event:org:resource'].astype(str)
event_row_df['caseid'] = event_row_df['caseid'].astype(str)
event_row_df['trace:concept:name'] = event_row_df['trace:concept:name'].astype(str)

event_row_df.head()

output_file = "./data/output.csv"
event_row_df.to_csv(output_file, index_label=False)
