from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import time
import random

# Subject: Forecasting taxt trips with LSTMs
# Method: LSTM
# Data: XBTUSD trading data from BitMex
# - side: buy or sell
# - tickDirection: increase/decrease in the price since the previous transaction
# - trdMatchID: the unique trade ID
# - grossValue: num. of satoshis exchanged
# - homeNotional: amount of XBT in the trade
# - foreignNotional: amount of USD in the trade

# loading the data
df_vwap = pd.read_csv('../data/bitmex/xbtusd_vwap.csv')
df_vwap.timestamp = pd.to_datetime(df_vwap.timestamp.str.replace('D', 'T'))
df_vwap.set_index('timestamp', inplace=True)
df_vwap = pd.Series(df_vwap['vwap'])

# plot the VWAP values
ax = df_vwap.plot(figsize=(14,7))
ax.axvline("2019-09-01", linestyle="--", c="black")
ax.axvline("2019-09-05", linestyle="--", c="black")
plt.show()

# separate train/validation/test sets
df_train = df_vwap[df_vwap.index < '2019-09-01'].to_frame(name='vwap')
df_valid = df_vwap[(df_vwap.index >= '2019-09-01') & (df_vwap.index < '2019-09-05')].to_frame(name='vwap')
df_test = df_vwap[df_vwap.index >= '2019-09-05'].to_frame(name='vwap')

# scaling the data
# fit on the training set
# transform the validation and test sets
# [!] if fit on all data, an overfitting may occurs
scaler = StandardScaler()
train = scaler.fit_transform(df_train)
valid = scaler.transform(df_valid)
test = scaler.transform(df_test)

# transforming the data for the LSTM modelling
# long sequence of data ->
# many shorter sequences (100 time bars per sequence) that are shifted by a single time bar.
# the model takes the feature of the ith time bar and predicts the target of the i+1 th time bar.
def transform_data(arr, seq_len):
    x, y =[], []
    for i in range(len(arr) - seq_len): # for ith sequence
        x_i = arr[i : i+seq_len]    #1st = [0:100], 2nd = [1:101], 3rd = [2:102], ...
        y_i = arr[i+1 : i+1+seq_len]#1st = [1:101], 2nd = [2:102], 3rd = [3:103]. ...
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var

seq_len = 100       # the length of sequence
x_train, y_train = transform_data(train, seq_len)
x_valid, y_valid = transform_data(valid, seq_len)
x_test, y_test = transform_data(test, seq_len)

# plotting the sequences
def plot_sequence(axes, i, x_train, y_train):
    axes[i].set_title("%d. Sequence" % (i+1))
    axes[i].set_xlabel("Time Bars")
    axes[i].set_ylabel('scaled VWAP')
    axes[i].plot(range(seq_len), x_train[i].cpu().numpy(), color='r', label='Feature')
    axes[i].plot(range(1, seq_len + 1), y_train[i].cpu().numpy(), color='b', label='Target')
    axes[i].legend()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
plot_sequence(axes, 0, x_train, y_train)
plot_sequence(axes, 1, x_train, y_train)
plt.show()

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

# Model 1: LSTM model without teacher forcing
model_1 = Model(input_size=1, hidden_size=21, output_size=1)
loss_fn_1 = nn.MSELoss()
optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-3)
scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=5, gamma=0.1)
optimization_1 = Optimization(model_1, loss_fn_1, optimizer_1, scheduler_1)
# train the LSTM
optimization_1.train(x_train, y_train, x_valid, y_valid, do_teacher_forcing=False)

# plotting the train-validation errors
optimization_1.plot_losses()
plt.show()

# model test
actual_1, predicted_1, test_loss_1 = optimization_1.evaluate(x_test, y_test, future=5, batch_size=100)
df_result_1 = to_dataframe(actual_1, predicted_1)
df_result_1 = inverse_transform(scaler, df_result_1, ['actual', 'predicted'])
df_result_1.plot(figsize=(14, 7))
print("Test loss %.4f" % test_loss_1)
plt.show()

# compare predicted, generated and actual values
x_sample = x_test[0].reshape(1, -1)
y_sample = df_test.vwap[:1100]

y_pred1 = generate_sequence(scaler, optimization_1.model, x_sample) # generated VWAP

plt.figure(figsize=(14,7))
plt.plot(range(100), y_pred1[0][:100], color='blue', lw=2, label='Predicted VWAP')
plt.plot(range(100, 1100), y_pred1[0][100:], '--', color='blue', lw=2, label='Generated VWAP')
plt.plot(range(0, 1100), y_sample, color='red', label='Actual VWAP')
plt.legend()

# Model 2: LSTM model with teacher forcing
model_2 = Model(input_size=1, hidden_size=21, output_size=1)
loss_fn_2 = nn.MSELoss()
optimizer_2 = optim.Adam(model_2.parameters(), lr=1e-3)
scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=5, gamma=0.1)
optimization_2 = Optimization(model_2, loss_fn_2, optimizer_2, scheduler_2)
# use teacher forcing
optimization_2.train(x_train, y_train, x_valid, y_valid, do_teacher_forcing=True)
# plotting the train-validation errors
optimization_2.plot_losses()
plt.show()

# model test
actual_2, predicted_2, test_loss_2 = optimization_2.evaluate(x_test, y_test, batch_size=100, future=5)
df_result_2 = to_dataframe(actual_2, predicted_2)
df_result_2 = inverse_transform(scaler, df_result_2, ["actual", "predicted"])
df_result_2.plot(figsize=(14, 7))
print("Test loss %.4f " % test_loss_2)

# compare predicted, generated and actual values
y_pred2 = generate_sequence(scaler, optimization_2.model, x_sample)
plt.figure(figsize=(14, 7))
plt.plot(range(100), y_pred2[0][:100], color="blue", lw=2, label="Predicted VWAP")
plt.plot(range(100, 1100), y_pred2[0][100:], "--", color="blue", lw=2, label="Generated VWAP")
plt.plot(range(0, 1100), y_sample, color="red", label="Actual VWAP")
plt.legend()
plt.show()

## conclusion
# 1. the teacher forcing technique helps RNNs or LSTMs to generate sequences more accurately