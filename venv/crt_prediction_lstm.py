# Subject: Case Remaining Time Prediction
# Method: LSTM
# Feature set:
#   (1) activity vector (binary)
#   (2) time from trace start

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import time
import random
from matplotlib import pyplot as plt

# read data
df = pd.read_csv("./data/bpic2012.csv")
print(df)

# select events whose lifecycle type is 'complete'
df = df.loc[df['lifecycle_type'] == 'complete']
df = df.reset_index()   # reset index

# calculate the number and lengths of traces
num_of_traces = len(np.unique(df['case_id']))
lens_of_traces = df.groupby('case_id').nunique()['index']

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
train = preprocess.fit_transform(df[:parting_event_idx]).toarray()
valid = preprocess.transform(df[parting_event_idx:]).toarray()

# calculate the size of input vector
input_vector_size = train.shape[1]-1    # excludes the attribute of target values

# transforming the data for the LSTM modelling
# long sequence of data ->
# many shorter sequences (100 time bars per sequence) that are shifted by a single time bar.
# the model takes the feature of the ith time bar and predicts the target of the i+1 th time bar.
def transform_data(arr, seq_len):
    x, y =[], []
    for i in range(len(arr) - seq_len): # for ith sequence
        x_i = arr[i:i+seq_len, 0:input_vector_size]   #1st = [0:100, 0:24], 2nd = [1:101, 0:24], ...
        y_i = arr[i+1:i+1+seq_len, input_vector_size] #1st = [1:101, 24], 2nd = [2:102, 24], ...
        x.append(x_i)
        y.append(y_i)
    print("[TEMP]x.shape = " + str(np.array(x).shape))
    print("[TEMP]y.shape = " + str(np.array(y).shape))
    # x_arr = np.array(x).reshape(-1, seq_len, input_vector_size)   # num. of features: 25
    # x_arr = np.array(x).reshape(-1, input_vector_size)
    x_arr = np.array(x)
    # y_arr = np.array(y).reshape(-1, seq_len)
    # y_arr = np.array(y).reshape(-1)
    y_arr = np.array(y)
    print("[TEMP]x_arr.shape = " + str(x_arr.shape))
    print("[TEMP]y_arr.shape = " + str(y_arr.shape))
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var

seq_len = 100       # the length of sequence
x_train, y_train = transform_data(train, seq_len)
x_valid, y_valid = transform_data(valid, seq_len)

# the LSTM model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, input, future=0, y=None):
        outputs = []
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.float32)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

# A helper class to train, test and diagnose the LSTM
class Optimization:
    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []
    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i:i+batch_size]
            # x_batch = x[i:i+batch_size].reshape(-1, input_vector_size)
            y_batch = y[i:i+batch_size]
            yield x_batch, y_batch, batch
    def train(self, x_train, y_train, x_val=None, y_val=None, batch_size=100, n_epochs=15, do_teacher_forcing=None):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []
            train_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= batch
            self.train_losses.append(train_loss)
            self._validation(x_val, y_val, batch_size)
            elapsed = time.time() - start_time
            print("Epoch %d Train Loss: %.2f. Validation loss: %.2f. Avg future: %.2f. Elapsed time: %.2fs." % (epoch + 1, train_loss, self.val_losses[-1], np.average(self.futures), elapsed))
    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            print("[TEMP]x_batch.shape = " + str(x_batch.shape))    # ensure [100, 100, 24]
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred
    def _validation(self, x_val, y_val, batch_size):
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                val_loss += loss.item()
            val_loss /= batch
            self.val_losses.append(val_loss)
    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                y_pred = self.model(x_batch, future=future)
                y_pred = (y_pred[:, -len(y_batch):] if y_pred.shape[1] > y_batch.shape[1] else y_pred)
                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(y_batch[:, -1]).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            test_loss /= batch
            return actual, predicted, test_loss
    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title('Losses')

# generate future values for x_sample with the model
def generate_sequence(scaler, model, x_sample, future=1000):
    y_pred_tensor = model(x_sample, future=future)
    y_pred = y_pred_tensor.cpu().tolist()
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred

def to_dataframe(actual, predicted):
    return pd.DataFrame({'actual': actual, 'predicted': predicted})

def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df

# Model: LSTM model without teacher forcing
model = Model(input_size=24, hidden_size=21, output_size=1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
optimization = Optimization(model, loss_fn, optimizer, scheduler)
# train the LSTM
optimization.train(x_train, y_train, x_valid, y_valid, do_teacher_forcing=False)

# size match problem
# ref: https://discuss.pytorch.org/t/time-series-lstm-size-mismatch-beginner-question/4704/9
# ref: https://www.youtube.com/watch?v=6WdLgUEcJMI&feature=youtu.be
