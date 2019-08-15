from opyenxes.extension.XExtensionParser import XExtensionParser
from opyenxes.extension.XExtensionManager import XExtensionManager
from opyenxes.data_in.XUniversalParser import XUniversalParser

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
with open("./data/review_example_large.xes") as file:
    logs = XUniversalParser().parse(file)

log = logs[0]
print("# traces in 1st log = %d" % (len(log)))

trace = log[0]
print("# events in 1st trace = %d" % (len(trace)))

event = trace[0]
print("# attributes in 1st event = %d" % (len(event.get_attributes())))

for key, value in event.get_attributes().items():
    print("%s=%s" %(key, value))

