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

import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_sequence

# LSTM definition
class EncoderRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        # huny modify
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    def init_hidden(self):
        hidden_h = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        hidden_c = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        hidden_h = hidden_h.cuda();
        hidden_c = hidden_c.cuda()
        return hidden_h, hidden_c
    def forward(self, input_seqs, input_lengths, hidden=None):
        self.hidden = self.init_hidden()
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seqs, input_lengths, batch_first=True)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        out = outputs.contiguous()
        out = out.view(-1, out.shape[2])
        out = self.fc(out)
        return out, hidden

# onehot encoding class
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.length = 0
    def add_word(self, word):
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = self.length + 1
            self.length += 1
        return self.word2idx[word]
    def __len__(self):
        return len(self.idx2word)
    def onehot_encoded(self, word, addVec):
        vec = np.zeros(self.length + len(addVec))
        vec[:len(addVec)] = addVec
        vec[self.word2idx[word] - 1 + len(addVec)] = 1
        return vec

# trace preprocess
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

# read excel file
data = pd.read_excel('../data/LB2000.xlsx')
traceID = data['traceID'].values
eventID = data['eventID'].values
remainTime = data['remainTime'].values
eventNum = data['eventNum'].values
runningH = data['runningH'].values
executedT = data['executedTime'].values
data.head()

# onehot enconding for eventID
dic = Dictionary()
for tok in eventID:
    dic.add_word(tok)

eventID_oh = []
for x in range(len(eventID)):
    eventID_oh.append(dic.onehot_encoded(eventID[x], [eventNum[x], executedT[x]]))

# regularization
# Robust:0 ## avr:1 ## midle:2
normalizeKey = 0

# event별 정규화를 위한 evnet별 remainTime 분류 및 정렬
eachArr = {}
for i in range(len(eventID)):
    try:
        eachArr[eventID[i]].append(remainTime[i])
    except:
        eachArr[eventID[i]] = []

normalizeArr = {}
listForSpecial = []
for p in dic.word2idx:
    # robust 정규화
    if normalizeKey == 0:
        tmpArr = np.sort(eachArr[p])
        a = tmpArr[int(len(eachArr[p]) / 4)]
        b = tmpArr[int(len(eachArr[p]) / 2)]
        c = tmpArr[int(len(eachArr[p]) / 4) + int(len(eachArr[p]) / 2)]
        d = tmpArr[0]
        e = tmpArr[-1]
        if a != c:  # 일반경우
            normalizeArr[p] = [a, b, c, 0]  # 일반적 로버스트 가능 상황은 네번째 0을 넣어 놓음 (0이 아니면 특별처리 필요)
        elif c == 0 and e != 0:
            normalizeArr[p] = [np.mean(tmpArr), np.std(tmpArr), 0, 1]  # 로버스트 세 값이 0인데 끝쪽은 0보다 큰경우
            listForSpecial.append(p)
        elif a == c and c != e:
            normalizeArr[p] = [np.mean(tmpArr), np.std(tmpArr), 0, 1]  # 로버스트 세 값이 0인데 끝쪽은 0보다 큰경우
            listForSpecial.append(p)
        elif d == e:  # 모든 값이 동일 할 경우
            normalizeArr[p] = [int(2 / 3 * b), b, int(4 / 3 * b), 0]
        else:
            normalizeArr[p] = [int(2 / 3 * b), b, int(4 / 3 * b), 0]  # 해당 파트 수정 필요
        print(p, "/", len(tmpArr), " : ", normalizeArr[p], 0)

# remainTime 정규화
nor_remainT = [] # 정규화된 값이 저장될 배열
for i in range(len(remainTime)):
    if normalizeKey==0:
        if normalizeArr[eventID[i]][3] == 1:
            nor_remainT.append( (remainTime[i]-normalizeArr[eventID[i]][0])/normalizeArr[eventID[i]][1] )
        elif normalizeArr[eventID[i]][2]!=0:
            nor_remainT.append((remainTime[i]-normalizeArr[eventID[i]][1])/(normalizeArr[eventID[i]][2]-normalizeArr[eventID[i]][0]))
        else:
            nor_remainT.append(0)


trace_len, maxlen, t_point = traceCount(traceID)
sortedX,sortedY = [eventID_oh[0:t_point[0]]],[nor_remainT[0:t_point[0]]]
sortedID = [eventID[0:t_point[0]]]

for i in range(1,trace_len):
    for k in range(len(sortedX)):
        if len(sortedX[k])<(t_point[i]-t_point[i-1]):
            sortedX.insert(k,eventID_oh[t_point[i-1]:t_point[i]])
            sortedY.insert(k,nor_remainT[t_point[i-1]:t_point[i]])
            sortedID.insert(k,eventID[t_point[i-1]:t_point[i]])
            break
        if((k+1)==len(sortedX)):
                sortedX.append(eventID_oh[t_point[i-1]:t_point[i]])
                sortedY.append(nor_remainT[t_point[i-1]:t_point[i]])
                sortedID.append(eventID[t_point[i-1]:t_point[i]])

train_x, train_y, test_x, test_y = [], [], [], []
id_x = []  # loss를 분으로 계산하기 위한 리스트
lenVec = len(dic.word2idx) + 2

for i in range(trace_len):
    tmp_x = torch.tensor(np.array(sortedX[i]))
    tmp_x = tmp_x.view(len(tmp_x), lenVec)
    tmp_y = torch.tensor(np.array(sortedY[i]))
    tmp_y = tmp_y.view(len(tmp_y), 1)
    if (i % 5) != 0:
        train_x.append(tmp_x); train_y.append(tmp_y)
    else:
        test_x.append(tmp_x);
        test_y.append(tmp_y)
        id_x.append(sortedID[i])

is_cuda = torch.cuda.is_available()
dict_size = lenVec
hidden_dim = dict_size*2
print("dic",dict_size)
# Define hyperparameters
n_epochs = 150

batch_size=50

model = EncoderRNN(input_size=dict_size, output_size=1, hidden_size=hidden_dim, n_layers=15, batch_size=batch_size)
if is_cuda:
    model = model.cuda()

lr=0.005

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

devKey = 2  # 프린트문 찍는 간격
lossMin = 9999999.0
# Training Run & Test Run
for epoch in range(1, n_epochs + 1):
    model.train()
    lossSum = 0
    lossCnt = 0
    for i in range(0, len(train_x), batch_size):
        if i + batch_size > len(train_x):
            break
        else:
            x = train_x[i:i + batch_size]
            y = train_y[i:i + batch_size]
            lenArr = []
            for x_len in x:
                lenArr.append(len(x_len))
            x = rnn_utils.pad_sequence(x, batch_first=True)
            x = Variable(x)
            y = rnn_utils.pad_sequence(y, batch_first=True)
            y = Variable(y)
            input_lengths = torch.LongTensor(lenArr)
            # print("x : ",x.size())
            # print("y : ",y.size())
            # print(input_lengths)
            if is_cuda: x = x.cuda(); y = y.cuda()
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            h, c = model.init_hidden()
            output, (h, c) = model(x.float(), input_lengths, (h, c))
            output = output.squeeze(1)
            # train
            loss = criterion(output, y.view(-1).float())
            loss.backward()  # Does backpropagation and calculates gradients
            optimizer.step()  # Updates the weights accordingly
            if epoch <= 10 or epoch % devKey == 0:
                lossSum += loss.item()
                lossCnt += 1
    if epoch <= 10 or epoch % devKey == 0:
        model.eval()
        testLossSum = 0
        testLossCnt = 0
        lossSumM = 0
        lossCntM = 0
        for i in range(0, len(test_x), batch_size):
            if i + batch_size > len(test_x):
                break
            else:
                x = test_x[i:i + batch_size]
                y = test_y[i:i + batch_size]
                checkID = id_x[i:i + batch_size]
                lenArr = []
                for x_len in x:
                    lenArr.append(len(x_len))
                x = rnn_utils.pad_sequence(x, batch_first=True)
                x = Variable(x)
                y = rnn_utils.pad_sequence(y, batch_first=True)
                y = Variable(y)
                input_lengths = torch.LongTensor(lenArr)
                if is_cuda: x = x.cuda(); y = y.cuda()
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                h, c = model.init_hidden()
                output, (h, c) = model(x.float(), input_lengths, (h, c))
                output = output.squeeze(1)
                for h in range(50):
                    for lenNum in range(lenArr[h]):
                        if normalizeArr[checkID[h][lenNum]][3] == 1:
                            a = output[50 * h + lenNum] * normalizeArr[checkID[h][lenNum]][1] + \
                                normalizeArr[checkID[h][lenNum]][0]
                            b = y[h][lenNum] * normalizeArr[checkID[h][lenNum]][1] + normalizeArr[checkID[h][lenNum]][0]
                            lossSumM += abs(a - b)
                            lossCntM += 1
                        else:
                            outM = output[50 * h + lenNum] * (
                                        normalizeArr[checkID[h][lenNum]][2] - normalizeArr[checkID[h][lenNum]][0]) + \
                                   normalizeArr[checkID[h][lenNum]][1]
                            outY = y[h][lenNum] * (
                                        normalizeArr[checkID[h][lenNum]][2] - normalizeArr[checkID[h][lenNum]][0]) + \
                                   normalizeArr[checkID[h][lenNum]][1]
                            lossSumM += abs(outM - outY)
                            lossCntM += 1
                # test
                loss = criterion(output, y.float())
                testLossSum += loss.item()
                testLossCnt += 1
        lossSumM = lossSumM.tolist()
        print('E{}/{}..'.format(epoch, n_epochs), end=' ')
        print("Train L:\t{:.8f}".format(lossSum / lossCnt), end=' ')
        print("  //Test L:\t{:.8f}".format(testLossSum / testLossCnt), end=' ')
        # print(lossSumM,"////",lossCntM)
        print("  //  Minute L:\t{:.8f}".format(lossSumM[0] / lossCntM))
        if lossMin > (lossSumM[0] / lossCntM):
            modelName = "RoEXModelLSTM.pth"
            torch.save(model.state_dict(), modelName)
            lossMin = lossSumM[0] / lossCntM
            print(epoch, "번째 loss :", lossMin)