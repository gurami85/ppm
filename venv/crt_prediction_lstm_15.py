# Subject: Case Remaining Time Prediction
# Method: LSTM (seq_len = 1, batch_size = size of events in training set, input_size = 27)
# Feature set:
#   (1) execution time vector (weighted, accumulated), debugged
#   (2) sequence of event, debugged
#   (3) time from trace start

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils import rnn
from matplotlib import pyplot as plt

"""
1. Preprocess
"""

# read data
df = pd.read_csv("./data/bpic2012_refined.csv")

# calculate trace number and lengths
num_of_traces = len(np.unique(df['case_id']))
traces_lens = df.groupby('case_id').count().seq_of_event
traces_lens = np.array(traces_lens)

# calculate a number of activity types
num_of_acts = len(np.unique(df['activity_type']))

# find a parting trace which is the last trace for the train/valid separation
parting_trace_idx = int(num_of_traces * 0.8)
parting_trace_id = np.unique(df['case_id'])[parting_trace_idx]

# find a parting event's index which is the last event's index of the parting trace.
# used as a separation line between train/valid sets
parting_event_idx = df.loc[df['case_id'] == parting_trace_id]\
    .index.values.astype(int)[-1]

# set up the transformer (one hot encoder, feature scaler)
preprocess = make_column_transformer(
    (OneHotEncoder(), ['activity_type']),
    (MinMaxScaler(), ['seq_of_event']),
    (RobustScaler(), ['time_from_trace_start', 'case_remaining_time'])
)

# transform data and separate it into train/valid sets
train = preprocess.fit_transform(df[:parting_event_idx+1]).toarray()
valid = preprocess.transform(df[parting_event_idx+1:]).toarray()

# scale 'execution_time' values
scaler = MinMaxScaler()

# replace ont-hot-encoded values into execution time values
# for training set
event_idx = 0
for i in range(parting_trace_idx+1):
    trace_len = traces_lens[i]
    for j in range(trace_len):
        for k in range(num_of_acts):
            if train[event_idx][k] == 1:
                train[event_idx][k] = df.weighted_execution_time[event_idx]
        event_idx += 1


# for validation set
base_idx = parting_event_idx + 1
event_idx = 0
for i in range(parting_trace_idx+1, num_of_traces):
    trace_len = traces_lens[i]
    for j in range(trace_len):
        for k in range(num_of_acts):
            if valid[event_idx][k] == 1:
                valid[event_idx][k] = df.weighted_execution_time[base_idx + event_idx]
        event_idx += 1


# transform the execution time values into accumulated execution time values
event_idx = 0
# for training set
for i in range(parting_trace_idx+1):
    trace_len = traces_lens[i]
    for j in range(trace_len):
        if j != 0:
            train[event_idx][:num_of_acts] = np.add(train[event_idx][:num_of_acts],
                                                    train[event_idx-1][:num_of_acts])
        event_idx += 1


# for validation set
event_idx = 0
for i in range(parting_trace_idx+1, num_of_traces):
    trace_len = traces_lens[i]
    for j in range(trace_len):
        if j != 0:
            valid[event_idx][:num_of_acts] = np.add(valid[event_idx][:num_of_acts],
                                                    valid[event_idx-1][:num_of_acts])
        event_idx += 1


# calculate the size of input vector
input_size = train.shape[1]-1    # excludes the attribute of target values

# transformation (ndarray -> torch)
def transform_data(input_data: np.ndarray) -> (np.ndarray, np.ndarray):
    x_lst, y_lst = [], []
    size = len(input_data)
    for i in range(size - seq_len + 1):
        # input sequence
        seq = input_data[i:i+seq_len, :input_size]
        # target values of current time steps
        target = input_data[i+seq_len-1, -1]
        x_lst.append(seq)
        y_lst.append(target)
    x_arr = np.array(x_lst)
    y_arr = np.array(y_lst)
    print("[INFO]x_arr.shape = " + str(x_arr.shape))
    print("[INFO]y_arr.shape = " + str(y_arr.shape))
    return x_arr, y_arr


# select device between gpu and cpu
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.float

seq_len = 10
batch_size = 100

x_train, y_train = transform_data(train)
x_valid, y_valid = transform_data(valid)

# calculate a number of batches
num_batches = int(x_train.shape[0] / batch_size)

if x_train.shape[0] % batch_size != 0:
    num_batches += 1


"""
2. Model Definition
"""

# hyperparameters setup
hidden_size = 150        # default: 32
output_dim = 1
num_layers = 3          # default: 2
learning_rate = 1e-3    # default: 1e-3
num_epochs = 200        # default: 200

# the LSTM model
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
        # forward pass through LSTM layer
        lstm_out, self.hidden = self.lstm(input) # [1, batch_size, 24]
        # only take the output from the final time step
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred.view(-1)

model = LSTM(input_size, hidden_size, batch_size=1, output_dim=output_dim, num_layers=num_layers)
model.seq_len = seq_len
model.cuda()    # for cuda
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


"""
3. Train the Model
"""

hist = np.zeros(num_epochs)     # loss history
for t in range(num_epochs):     # for each epoch
# for t in range(1):            # [TEST]
    y_pred = np.empty(0)
    for i in range(num_batches):    # for each batch
        print("Training the model: %d/%dth epoch, %d/%dth batch..."
              % (t + 1, num_epochs, i + 1, num_batches), end='\r')
        # last batch
        if i == num_batches-1:
            x_batch_arr = x_train[i*batch_size:]
            y_batch_arr = y_train[i*batch_size:]
        # other batches
        else:
            x_batch_arr = x_train[i*batch_size:i*batch_size+batch_size]
            y_batch_arr = y_train[i*batch_size:i*batch_size+batch_size]
        # transformation (ndarray -> torch)
        x_batch = Variable(torch.from_numpy(x_batch_arr).float()).type(dtype)
        y_batch = Variable(torch.from_numpy(y_batch_arr).float()).type(dtype)
        model.batch_size = x_batch.shape[0]
        model.hidden = model.init_hidden()
        # get predictions for the batch
        pred_i = model(x_batch)
        # forward pass
        loss_train = loss_fn(pred_i, y_batch)
        # zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()
        # backward pass
        loss_train.backward()
        # update parameters
        optimizer.step()
        # store the predictions
        y_pred = np.append(y_pred, pred_i.detach().cpu().numpy(), axis=0)
    if t == 0:
        loss_prev = float('inf')
    else:
        loss_prev = hist[t-1]
    # measure a loss in the current epohch
    loss_train = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y_train)).item()
    print("[INFO] Epoch ", t, ", Loss: ", loss_train, ", Difference: ", (loss_train - loss_prev))
    hist[t] = loss_train


"""
4. Visualization
"""

# default visualization setup
plt.figure(dpi=100)     # set the resolution of plot
# set the default parameters of visualization
color_main = '#2c4b9d'
color_sub = '#00a650'
color_ssub = '#ef9c00'
color_sssub = '#e6551e'
font_family = 'Calibri'
plt.rcParams.update({'font.family': font_family, 'font.size': 23, 'lines.linewidth': 1,
                    "patch.force_edgecolor": True, 'legend.fontsize': 18})

# calculate residual errors
err_func = lambda x, y: abs(x - y)
errors = err_func(y_train, y_pred)

# line plot
# plt.plot(errors, label="Residual Errors", kind='bar')
plt.plot(y_train, label="Actual Data")
plt.plot(y_pred, label="Predictions")
plt.legend(loc='best')
plt.show()

# visualize scatter plot
fig, ax = plt.subplots()
ax.scatter(y_train, y_pred, 10)   # 10: marker size
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
ax.set_xlabel('Actual Data')
ax.set_ylabel('Predictions')
plt.show()

# visualize scatter plot of filtered data
filtered_data_index = df.loc[df['seq_of_event'] <= 1].index
# filtered_data_index = df.loc[(df['seq_of_event'] > 1) & (df['seq_of_event'] < 10)].index
# filtered_data_index = df.loc[df['seq_of_event'] > 15].index
y_train_filtered = list()
pred_filtered = list()
for i in range(filtered_data_index.size):
    y_train_filtered.append(y_train[i])
    pred_filtered.append(pred[i])

y_train_filtered = np.asarray(y_train_filtered)
pred_filtered = np.asarray(pred_filtered)

fig, ax = plt.subplots()
ax.scatter(y_train_filtered, pred_filtered, 10)   # 10: marker size
ax.plot([y_train_filtered.min(), y_train_filtered.max()], [y_train_filtered.min(), y_train_filtered.max()], 'k--', lw=2)
ax.set_xlabel('Actual Data')
ax.set_ylabel('Predictions')
plt.show()

# visualize training loss
plt.plot(hist, label="Training loss")
plt.legend(loc='best')
plt.show()

