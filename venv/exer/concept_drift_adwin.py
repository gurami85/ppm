# Reference: https://scikit-multiflow.github.io/scikit-multiflow/skmultiflow.drift_detection.html
import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
adwin = ADWIN()
# Simulating a data stream as a normal distribution of 1's and 0's
data_stream = np.random.randint(2, size=2000)
# Changing the data concept from index 999 to 2000
for i in range(999, 2000):
    data_stream[i] = np.random.randint(4, high=8)
    # Adding stream elements to ADWIN and verifying if drift occurred

drift_ind = []
for i in range(2000):
    adwin.add_element(data_stream[i])
    if adwin.detected_change():
        print('Change detected in data: ' + str(data_stream[i]) + ' - at index: ' + str(i))
        drift_ind.append(i)