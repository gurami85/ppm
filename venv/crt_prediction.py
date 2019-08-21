import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

# read data from input file
training_data = pd.read_csv("./data/training_data.csv")
print(training_data)
# do the feature scaling
scaler = StandardScaler()
training_data = scaler.fit_transform(training_data)
print(training_data)

torch.manual_seed(1)    # for random number generations

