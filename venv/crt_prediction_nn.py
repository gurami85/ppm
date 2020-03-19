# Subject: Case Remaining Time Prediction
# Method: Neural Network (2, 7, 1)
# Feature set:
#   (1) num. of antecedent events
#   (2) avg. of case remaining time for each activity type

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# read data
df = pd.read_csv("./data/bpic2012_fs1.csv")
print(df)

# prepare train/test set
train = df[:-2000]  # train data: entire data - #2000
test = df[-2000:]   # test data: #2000

# do feature scaling
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train) # returns a numpy array as scaled data
test_scaled = scaler.fit_transform(test)

print("--- train_scaled ---")
print(train_scaled)
print("--- target scaled ---")
print(test_scaled)

# torch can only train on Variable, so convert them to Variable
# select features (0,1 th columns)
train_x = Variable(torch.Tensor(train_scaled[:,0:2]))
# select target values (2th col.)
train_y = Variable(torch.Tensor(train_scaled[:,2]))
# convert column vector to row vector
train_y = train_y.reshape(len(train_y), 1)

# NN definition
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)  # output layer
    def forward(self, x):
        x = F.relu(self.hidden(x))  # activation function for hidden layer
        x = self.predict(x)  # linear output
        return x

# create a network, n_feature=2
net = Net(n_feature=2, n_hidden=7, n_output=1)
print(net)

# optimizer setting
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
loss_lst = []

# train the network
for t in range(2000):
    y_hat = net(train_x)  # input x and predict based on x
    loss = loss_func(y_hat, train_y)  # must be (1. nn output, 2. target)
    loss_lst.append(loss)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

# Tensor -> ndarray for plotting
y_hat = y_hat.data.numpy()

# plotting training results
plt.rcParams.update({'font.size': 10})
plt.figure(dpi=100)
plt.scatter(torch.arange(0, len(train_y)), train_y, s=10, marker='.', label='Train')
plt.scatter(torch.arange(0, len(y_hat)), y_hat, s=10, marker='.', label='Prediction')
plt.title('Regression Analysis')
plt.xlabel('Events')
plt.ylabel('Case Remaining Time')
plt.legend(loc='best')
plt.show()

plt.plot(loss_lst)
plt.show()

# model test
test_x = Variable(torch.Tensor(test_scaled[:,0:2]))
test_y = Variable(torch.Tensor(test_scaled[:,2]))
test_y = test_y.reshape(len(test_y), 1)
loss_lst = []
for t in range(len(test_x)):
    y_hat = net(test_x)  # input x and predict based on x
    loss = loss_func(y_hat, test_y)  # must be (1. nn output, 2. target)
    loss_lst.append(loss)
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

# Tensor -> ndarray for plotting
y_hat = y_hat.data.numpy()

# plotting test results
plt.rcParams.update({'font.size': 10})
plt.figure(dpi=100)
plt.scatter(torch.arange(0, len(test_y)), test_y, s=10, marker='.', label='Test')
plt.scatter(torch.arange(0, len(y_hat)), y_hat, s=10, marker='.', label='Prediction')
plt.title('Regression Analysis')
plt.xlabel('Events')
plt.ylabel('Case Remaining Time')
plt.legend(loc='best')
plt.show()

plt.plot(loss_lst)
plt.show()