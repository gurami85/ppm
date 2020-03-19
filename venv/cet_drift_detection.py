# Purpose: Detecting concept drifts of case execution time to decide the period of training set.

import pandas as pd
from matplotlib import pyplot as plt
from skmultiflow.drift_detection.adwin import ADWIN

df = pd.read_csv('data/bpic2012_cet.csv')

# drift detection
adwin = ADWIN()
drift_ind = []

for idx, row in df.iterrows():
    cet = row['case_execution_time_seconds']
    adwin.add_element(cet)
    if adwin.detected_change():
        print('Change detected in data: ' + str(cet) + ' - at index: ' + str(idx))
        drift_ind.append(idx)

plt.plot(df['case_execution_time_seconds'])
for i in drift_ind:
    plt.axvline(i, color='black', linestyle='--', linewidth=1)

plt.show()