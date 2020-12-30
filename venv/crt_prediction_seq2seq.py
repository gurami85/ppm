"""
Subject: Case Remaining Time Prediction
Method: Seq2Seq (seq_len = 10, batch_size = 100)
Feature set:
    (1) activity_type (ont-hot encoded)
    (2) sequence of event
    (3) time from trace start
    (4) weighted execution time
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
import tensorflow as tf
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

# feature selection
fs8 = ['seq_of_event', 'activity_type',
       'time_from_trace_start', 'weighted_execution_time',
       'case_remaining_time']

# [Choice]
df = df[fs8]

# set up the transformer (one hot encoder, feature scaler)
preprocess = make_column_transformer(
    (OneHotEncoder(), ['activity_type']),
    (RobustScaler(), ['seq_of_event', 'time_from_trace_start', 'weighted_execution_time']),
    ('passthrough', ['case_remaining_time'])
)

# transform data and separate it into train/valid sets
train = preprocess.fit_transform(df[:parting_event_idx+1]).toarray()
valid = preprocess.transform(df[parting_event_idx+1:]).toarray()


"""
Transform a DataFrame to tf.data.Dataset to Work More Efficiently with Tensorflow
    - windows_size: the length of sequence
    - forecast_size: time steps that we want to forecast for each window 
    - total_size = window_size + forecast_size
"""


def create_dataset(arr, n_deterministic_features,
                   window_size, forecast_size,
                   batch_size):
    # feel free to play with shuffle buffer size
    shuffle_buffer_size = len(arr)
    # total size of windows is given by the number of steps to be considered
    # before prediction time + steps that we want to forecast
    total_size = window_size + forecast_size
    data = tf.data.Dataset.from_tensor_slices(arr)
    # selecting windows
    data = data.window(total_size, shift=1, drop_remainder=True)
    data = data.flat_map(lambda k: k.batch(total_size))
    # shuffling data
    # seed=Answer to the Ultimate Question of Life, the Universe, and Everything
    data = data.shuffle(shuffle_buffer_size, seed=42)
    # extracting past features + deterministic future features + target variable
    # k[-forecast_size+1, -1] : CRT at current time step (not forecast)
    # k[-forecast_size: -1]: CRTs at next time steps of forecast_size
    data = data.map(lambda k: ((k[:-forecast_size],
                                k[-forecast_size:, :0]),
                               k[-forecast_size:, -1]))
    return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


seq_len = 30
# forecast_len = 10
forecast_len = 1      # for on-time prediction
batch_size = 100
n_deterministic_features = 0
n_total_features = train.shape[1]

training_windowed = create_dataset(train,
                                   n_deterministic_features,
                                   seq_len,
                                   forecast_len,
                                   batch_size)

validation_windowed = create_dataset(valid,
                                     n_deterministic_features,
                                     seq_len,
                                     forecast_len,
                                     batch_size)

test_windowed = create_dataset(valid,
                               n_deterministic_features,
                               seq_len,
                               forecast_len,
                               batch_size=1)

latent_dim = 16     # num. of nodes in hidden layers

# first branch of the net is an lstm which finds an embedding for the past
past_inputs = tf.keras.Input(
    shape=(seq_len, n_total_features), name='past_inputs')

# Encoder: encoding the past
encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(past_inputs)

# Decoder: combining future inputs with recurrent branch output
future_inputs = tf.keras.Input(
    shape=(forecast_len, n_deterministic_features), name='future_inputs')

decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
x = decoder_lstm(future_inputs,
                 initial_state=[state_h, state_c])

x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
x = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='relu')(x)

model = tf.keras.models.Model(
    inputs=[past_inputs, future_inputs], outputs=output)

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.MeanAbsoluteError()
# loss = tf.keras.losses.Huber()
model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])

# training the model just by calling model.fit( ),
# since we are using tf.data.Dataset objects
history = model.fit(training_windowed, epochs=25,
                    validation_data=validation_windowed)

model.evaluate(test_windowed)


"""
Visualization
"""

fig, ax = plt.subplots(nrows=10, ncols=4, sharex='all', sharey='all')

for i, data in enumerate(test_windowed.take(40)):
    (past, future), truth = data
    truth = truth
    pred = model.predict((past,future))
    row = i//2
    col = i%2
    ax[row][col].plot(pred.flatten(), label='Prediction')
    ax[row][col].plot(truth.numpy().flatten(),label='Truth')

# Labeling axes
for i in range(2):
    ax[2][i].set_xlabel('Time Steps')

for i in range(3):
    ax[i][0].set_ylabel('CRT')


handles, labels = ax[0][0].get_legend_handles_labels()
fig.subplots_adjust(wspace=0, hspace=0.5)
fig.legend(handles, labels, loc='upper right')

fig.show()

