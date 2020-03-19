# Subject: Case Remaining Time Prediction
# Method: LSTM (seq_len = 1, batch_size = each trace length, input_size = 24)
# Feature set:
#   (1) activity vector (binary, not accumulated)
#   (2) time from trace start
# Ref: https://github.com/jessicayung/blog-code-snippets/blob/master/lstm-pytorch/lstm-baseline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt

# [1. Preprocess]

# reads data
df = pd.read_csv("./data/bpic2012.csv")
print(df)

# selects only 'complete' events
df = df.loc[df['lifecycle_type'] == 'complete']
df = df.reset_index()   # reset index

# calculate the number and lengths of traces
num_of_traces = len(np.unique(df['case_id']))
traces_lens = df.groupby('case_id').nunique()['index']
traces_lens = np.array(traces_lens)

# find the parting trace which is the last trace for the train/valid separation
parting_trace_idx = int(num_of_traces * 0.8)
parting_trace_id = np.unique(df['case_id'])[parting_trace_idx]

# find the parting event's index which is the last event's index of the parting trace
# we use this index value later as a separation line between training and valid sets
parting_event_idx = df.loc[df['case_id'] == parting_trace_id].index.values.astype(int)[-1]

# make features (time from trace start, target values)
#   case 1: use 'REG_DATE' in <trace>
#   data sets fall into case 1: BPIC2012
df['REG_DATE'] = pd.to_datetime(df['REG_DATE'])
df['timestamp'] = pd.to_datetime(df['timestamp'])
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
    crt_lst.append(case_remaining_time)

# case 2: no 'REG_DATE' in <trace>
# data sets fall into case 2:
# [!] N/A

# select feature set from dataframe
df['time_from_trace_start'] = pd.DataFrame(tfts_lst)
df['case_remaining_time'] = pd.DataFrame(crt_lst)
df = df[['activity_type', 'time_from_trace_start', 'case_remaining_time']]
print(df)

# one hot encoding and feature scaling
preprocess = make_column_transformer(
    (OneHotEncoder(), ['activity_type']),
    (StandardScaler(), ['time_from_trace_start', 'case_remaining_time'])
)

# separates train/valid sets
train = preprocess.fit_transform(df[:parting_event_idx+1]).toarray()
valid = preprocess.transform(df[parting_event_idx+1:]).toarray()

# calculate the size of input vector
input_size = train.shape[1]-1    # excludes the attribute of target values
lstm_input_size = input_size


# transforming the data for the LSTM modelling
def transform_data(arr):
    x = arr[:, 0:input_size]
    x_arr = np.array(x).reshape(1, -1, input_size)
    y = arr[:, input_size]
    y_arr = np.array(y)
    print("[INFO]x_arr.shape = " + str(x_arr.shape))
    print("[INFO]y_arr.shape = " + str(y_arr.shape))
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var


x_train, y_train = transform_data(train)
x_valid, y_valid = transform_data(valid)

# dtype = torch.cuda.FloatTensor # to use cuda
# x_train = x_train.type(dtype)
# y_train = y_train.type(dtype)
# x_valid = x_valid.type(dtype)
# y_valid = y_valid.type(dtype)

# [2. Model Definition]

# setup the hyperparameters
hidden_size = 32
output_dim = 1
num_layers = 2
learning_rate = 1e-3
num_epochs = 200

# the LSTM model (baseline model in Ref)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    def forward(self, input):
        # Forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(input.view(1, self.batch_size, -1))   # [1, batch_size, 24]
        # Only take the output from the final time step
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

model = LSTM(lstm_input_size, hidden_size, batch_size=1, output_dim=output_dim, num_layers=num_layers)
# model.cuda()    # for cuda
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# [3. Train Model]
hist = np.zeros(num_epochs)

# for t in range(num_epochs):
for t in range(5):
    idx = 0                 # index of first event in current trace
    pred = torch.empty(0)   # tensor for all predictions
    for i in range(parting_trace_idx+1):
        model.batch_size = traces_lens[i]   # batch_size = current trace's length
        # initialize hidden state. Don't do this if you want your LSTM to be stateful
        model.hidden = model.init_hidden()
        # get predictions for the trace
        pred_i = model(x_train[0][idx:idx+model.batch_size])
        # concatenate current predictions to the entire set
        pred = torch.cat((pred, pred_i), 0)
        idx += model.batch_size
    # Forward pass
    loss = loss_fn(pred, y_train)
    print("[INFO] Epoch ", t, ", Loss: ", loss.item())
    hist[t] = loss.item()
    # Zero out gradient, else they will accumulate between epochs
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # Update parameters
    optimizer.step()

# [4. Visualization]

# transform data for the visualization
y_train = y_train.detach().cpu().numpy()
pred = pred.detach().cpu().numpy()

# visualize line plot
plt.plot(pred, label="Predictions")
plt.plot(y_train, label="Actual Data")
plt.legend()
plt.show()

# visualize scatter plot
fig, ax = plt.subplots()
ax.scatter(y_train, pred)
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
ax.set_xlabel('Actual Data')
ax.set_ylabel('Predictions')
plt.show()

plt.plot(hist, label="Training loss")
plt.legend(loc='best')
plt.show()