# Subject: Case Remaining Time Prediction
# Method: LSTM (seq_len = each trace length, batch_size = 100 (batch_limit_size), input_size = 24)
# Feature set:
#   (1) activity vector (binary, not accumulated)
#   (2) sequence of event
#   (3) time from trace start
#   (4) num. of events of day of week
#   (5) num. of events of hour of day
# Etc: Padding + Packing
# Ref: https://github.com/jessicayung/blog-code-snippets/blob/master/lstm-pytorch/lstm-baseline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils import rnn
from matplotlib import pyplot as plt

# [1. Preprocess]

# read data
df = pd.read_csv("./data/bpic2012.csv")
print(df)

# select only 'complete' events
df = df.loc[df['lifecycle_type'] == 'complete']
df = df.reset_index()   # reset index
df_time = pd.to_datetime(df['timestamp'], utc=True)  # convert to UTC datetime

# make features: num. of completed events by hour of day
hour_of_day = df_time.dt.hour
hour_of_day_counts = hour_of_day.value_counts().sort_index()
num_of_events_hour_of_day = hour_of_day_counts[hour_of_day]
num_of_events_hour_of_day.index = df.index
df['num_of_events_hour_of_day'] = num_of_events_hour_of_day

# make features: num. of completed events by day of week
day_of_week = df_time.dt.dayofweek
day_of_week_counts = day_of_week.value_counts().sort_index()
num_of_events_day_of_week = day_of_week_counts[day_of_week]
num_of_events_day_of_week.index = df.index
df['num_of_events_day_of_week'] = num_of_events_day_of_week

# calculate the number and lengths of traces
num_of_traces = len(np.unique(df['case_id']))
traces_lens = df.groupby('case_id').nunique()['index']
traces_lens = np.array(traces_lens)

# calculate the number of activity types
num_of_acts = len(np.unique(df['activity_type']))   # used as length of one hot encoded sequence

# find the parting trace which is the last trace for the train/valid separation
parting_trace_idx = int(num_of_traces * 0.8)
parting_trace_id = np.unique(df['case_id'])[parting_trace_idx]

# find the parting event's index which is the last event's index of the parting trace
# we use this index value later as a separation line between training and valid sets
parting_event_idx = df.loc[df['case_id'] == parting_trace_id].index.values.astype(int)[-1]

# make features: time from trace start
# make target values: case remaining time
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
# [!] not implemented

# select feature set from the data frame
df['time_from_trace_start'] = pd.DataFrame(tfts_lst)
df['case_remaining_time'] = pd.DataFrame(crt_lst)
df = df[['activity_type', 'seq_of_event', 'time_from_trace_start', 'num_of_events_hour_of_day',
         'num_of_events_day_of_week', 'case_remaining_time']]
print(df)

# one hot encoding and feature scaling
preprocess = make_column_transformer(
    (OneHotEncoder(), ['activity_type']),
    (StandardScaler(), ['seq_of_event', 'time_from_trace_start', 'num_of_events_hour_of_day',
     'num_of_events_day_of_week', 'case_remaining_time'])
)

# separate train/valid sets
train = preprocess.fit_transform(df[:parting_event_idx+1]).toarray()
valid = preprocess.transform(df[parting_event_idx+1:]).toarray()

# calculate the size of input vector
input_size = train.shape[1]-1    # excludes the attribute of target values

# transformation (ndarray -> torch)
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


# select device between gpu and cpu
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float
x_train, y_train = transform_data(train)
x_valid, y_valid = transform_data(valid)
x_train.type(dtype)
y_train.type(dtype)
x_valid.type(dtype)
y_valid.type(dtype)


# [2. Model Definition]

# setup the hyperparameters
batch_limit_size = 100
hidden_size = 32
output_dim = 1
num_layers = 2
learning_rate = 1e-3
num_epochs = 200

# the LSTM model (baseline model in Ref)
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seq_len = 0
        self.num_layers = num_layers
        # define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        # define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
    def init_hidden(self):
        # initialize hidden states
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type(dtype),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).type(dtype))
    def forward(self, input):
        input = input.type(dtype)
        # transform (batch(input) -> packed_padded_batch)
        #  (1) batch[seq_len, input_dim] -> padded_batch[batch_size, seq_len, input_dim]
        padded_batch = torch.zeros(self.batch_size, self.seq_len, self.input_dim)
        padded_batch = padded_batch.type(dtype)
        # these control the padding for each trace
        trace_idx = batch_first_trace_idx
        first_event_idx = 0
        last_event_idx = traces_lens[trace_idx] - 1
        for i in range(input.size(0)):
            # when the round of the current trace ends, prepare the variables for the next round
            if not i <= last_event_idx:
                first_event_idx += traces_lens[trace_idx]
                trace_idx += 1
                last_event_idx += traces_lens[trace_idx]
            # padding (i: the event order in the batch, j: the event order in the trace)
            for j in range(0, i - first_event_idx + 1):
                padded_batch[i][j] += input[first_event_idx + j]
        #  (2) padded_batch -> packed_padded_batch
        lengths = []
        for i in range(batch_first_trace_idx, batch_last_trace_idx+1):
            lengths += list(range(1, traces_lens[i]+1))
        packed_padded_batch = rnn.pack_padded_sequence(padded_batch, lengths, batch_first=True, enforce_sorted=False)
        # forward pass through LSTM layer
        packed_padded_out, self.hidden = self.lstm(packed_padded_batch, self.hidden)
        # undo the packing operation and take the outputs from PackedSequence[0]
        lstm_out = rnn.pad_packed_sequence(packed_padded_out, batch_first=True)[0]
        # take the outputs (including padded outputs)
        lstm_out = self.linear(lstm_out)    # seq_len * batch_size
        # masking: only take the valid outputs
        y_pred = torch.zeros(self.batch_size)
        for i in range(lstm_out.size(0)):
            y_pred[i] = lstm_out[i][lengths[i]-1]
        return y_pred

model = LSTM(input_size, hidden_size, batch_size=1, output_dim=output_dim, num_layers=num_layers)
# model.cuda()    # for cuda
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# [3. Train Model]

hist = np.zeros(num_epochs)
prev_loss = float('inf')    # for calculate loss difference
for t in range(num_epochs): # for training
# for t in range(1):         # for sanity checking
    idx = 0                 # index of first event in current trace
    batch_size = 0
    batch_first_trace_idx = 0
    batch_last_trace_idx = 0
    pred = torch.empty(0)   # tensor for all predictions
    pred = pred.type(dtype)
    for i in range(parting_trace_idx+1):
        batch_size += traces_lens[i]
        if batch_size > batch_limit_size or i == parting_trace_idx:
            batch_last_trace_idx = i
            model.batch_size = batch_size  # batch_size <= batch_limit_size
            # seq_size = the maximum of length of the selected traces
            model.seq_len = max(traces_lens[batch_first_trace_idx:batch_last_trace_idx+1])
            model.hidden = model.init_hidden()  # initialize hidden state
            # get predictions for the trace
            pred_i = model(x_train[0][idx:idx+batch_size]).type(dtype)
            # concatenate current predictions to the entire set
            pred = torch.cat((pred, pred_i), 0)
            idx += model.batch_size
            batch_size = 0
            batch_first_trace_idx = i+1  # reset the index for next batch
    # Forward pass
    loss = loss_fn(pred, y_train.type(dtype))
    print("[INFO] Epoch ", t, ", Loss: ", loss.item(), ", Difference: ", (loss.item() - prev_loss))
    prev_loss = loss.item()
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

