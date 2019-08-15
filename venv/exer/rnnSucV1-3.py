#!/usr/bin/env python
# coding: utf-8

# In[16]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt

import datetime
import pandas as pd
import pandas_datareader.data as pdr
from openpyxl import load_workbook

import torch.nn.functional as F
from torch.autograd import Variable

#import argparse
#from copy import deepcopy # Add Deepcopy for args
#from sklearn.metrics import mean_absolute_error
#import seaborn as sns 


# In[17]:


#self.input_tensor = torch.eye(self.n_words)[self.tensor, :]#(데이터 종류 갯수)[]
#https://pytorch.org/docs/stable/torch.html


# In[18]:


class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        hidden = hidden.cuda()
        return hidden


# In[19]:


#onehot encode
class Dictionary(object):
    def __init__(self):
        self.word2idx={}
        self.idx2word=[]
        self.length=0
    
    def add_word(self, word):
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = self.length+1
            self.length+=1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
    def onehot_encoded(self,word):
        vec = np.zeros(self.length)
        vec[self.word2idx[word]-1] =1
        return vec


# In[20]:


aa = [0]
aa.append(1)
aa


# In[21]:


#trace preprocess
def traceCount(traceArr):
    w_count = {}; maxlen=0; t_point = []; pre_ID = traceArr[0]
    for i in range(1,len(traceArr)):
        if pre_ID != traceArr[i]: t_point.append(i)
        pre_ID = traceArr[i]
        try: 
            w_count[traceArr[i]] += 1 
            if maxlen < w_count[traceArr[i]]: maxlen=w_count[traceArr[i]]
        except: w_count[traceArr[i]]=1

    t_point.append(len(traceID))
        # trace개수, trace 최대 길이, trace 시작 포인트
    return len(w_count), maxlen, t_point 


# In[22]:


#<excel 파일을 읽어오기>
data = pd.read_excel('C:/dev/dataset/dataSet2.xlsx')
#data = pd.read_excel('C:/Users/user/aiHun/mini.xlsx')
traceID = data['traceID'].values
eventID = data['eventID'].values
remainTime = data['remainTime'].values
data.head()


# In[23]:


# eventID를 onehot encode하기
dic = Dictionary()
dic.add_word('end') # trace별 길이 고정을 위해 빈값 end로 매꿈 (추후 개선 요망)
for tok in eventID:
    dic.add_word(tok)
#print(dic.word2idx)


# In[24]:


# event id 모두 onehot encode로 변경
eventID_oh = []
for x in range(len(eventID)):
    eventID_oh.append(dic.onehot_encoded(eventID[x]))


# In[25]:


#### 정규화 선택 란 ####
# Robust:0 ## avr:1 ## midle:2
normalizeKey = 0


# In[26]:


# event별 정규화를 위한 evnet별 remainTime 분류 및 정렬
eachArr = {}
for i in range(len(eventID)):
    try: 
        eachArr[eventID[i]].append(remainTime[i])
    except: eachArr[eventID[i]]=[]
eachArr['end']=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,]

normalizeArr = {}
for p in dic.word2idx:
    if p != 'end':        
        #robust 정규화
        if normalizeKey==0:
            tmpArr = np.sort(eachArr[p])
            normalizeArr[p]=[tmpArr[int(len(eachArr[p])/4)], tmpArr[int(len(eachArr[p])/2)], tmpArr[int(len(eachArr[p])/4)+int(len(eachArr[p])/2)], tmpArr[-1]]
            print(p,"/", len(tmpArr)," : ", normalizeArr[p])
        elif normalizeKey==1:
            print("notyet")
        elif normalizeKey==2:
            print("notyet")


# In[27]:


# remainTime 정규화
nor_remainT = [] # 정규화된 값이 저장될 배열
for i in range(len(remainTime)):
    if normalizeKey==0:
        if(eventID[i] != 'end'):
            nor_remainT.append((remainTime[i]-normalizeArr[eventID[i]][1])/(normalizeArr[eventID[i]][2]-normalizeArr[eventID[i]][0]))
        else:
            nor_remainT.append(0)
    elif normalizeKey==1:
        print("notyet")
    elif normalizeKey==2:
        print("notyet")


# In[28]:


# train & test 셋으로 데이터 분할
trace_len, maxlen, t_point = traceCount(traceID)
train_num = int(trace_len*0.8)
maxlen = maxlen+1 # end 이벤트를 추가했기 때문에
train_x, train_y, test_x, test_y = [], [], [], []
tmp_x = dumy_x = [dic.onehot_encoded('end')]*maxlen
tmp_y = dumy_y = [0]*maxlen
traceStartPoint = 0

for i in range(trace_len):
    tmp_x[0:t_point[i]-traceStartPoint] = eventID_oh[traceStartPoint:t_point[i]]
    tmp_y[0:t_point[i]-traceStartPoint] = nor_remainT[traceStartPoint:t_point[i]] 
    
    if (i%5)!=0 : train_x.append(tmp_x); train_y.append(tmp_y)
    #if i>= (trace_len-train_num):train_x.append(tmp_x); train_y.append(tmp_y)
    #if i< train_num : train_x.append(tmp_x); train_y.append(tmp_y)
    else : test_x.append(tmp_x); test_y.append(tmp_y)
    traceStartPoint = t_point[i]
    tmp_x = dumy_x = [dic.onehot_encoded('end')]*maxlen
    tmp_y = dumy_y = [0]*maxlen


# In[29]:


is_cuda = torch.cuda.is_available()
dict_size = len(dic.word2idx)
model = Model(input_size=dict_size, output_size=1, hidden_dim=12, n_layers=1)
if is_cuda : model = model.cuda()
    
# Define hyperparameters
n_epochs = 500
lr=0.01
batch_size=125

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# In[15]:


# Training Run & Test Run
for epoch in range(1, n_epochs + 1):
    model.train()
    for i in range(0,len(train_x),batch_size):
        if i+batch_size > len(train_x): break
        else:
            x = torch.tensor(np.array(train_x[i:i+batch_size]))
            y = torch.tensor(np.array(train_y[i:i+batch_size]))
            
            if is_cuda: x,y = x.cuda(), y.cuda()
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output, hidden = model(x.float())
            output = output.squeeze(1)
            loss = criterion(output, y.view(-1).float())
            
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly    
            if epoch == 10 or epoch%50 == 0:
                print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
                print("Loss:\t{:.8f}".format(loss.item()))
                
    for i in range(0,len(test_x),batch_size):
        if i+batch_size > len(test_x): break
        else:
            x = torch.tensor(np.array(test_x[i:i+batch_size]))
            y = torch.tensor(np.array(test_y[i:i+batch_size]))
            
            if is_cuda: x,y = x.cuda(), y.cuda()
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            
            model.eval(); volatile=True
            
            output, hidden = model(x.float())  
            output = output.squeeze(1)
            loss = criterion(output, y.view(-1).float())  
            if epoch == 10 or epoch%50 == 0:
                print('Epoch: {}/{}...<eval>....'.format(epoch, n_epochs), end=' ')
                print("Loss:\t{:.8f}".format(loss.item()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




