#!/usr/bin/env python
# coding: utf-8

# 참고문헌 https://www.youtube.com/watch?v=sG_WeGbZ9A4&t=200s

import datetime
from openpyxl import Workbook

readFile = "C:/dev/dataset/review_example_large.xes"
inputFile= open(readFile, 'r')
flagKey = 10

traceStartTime = datetime.datetime(1994, 1, 1, 0, 0, 0, 0)
traceEndTime = datetime.datetime(1995, 1, 1, 0, 0, 0, 0) # year, month, day, hour, minute, second, microsecond


y=1994; m=10; d=25; h=7; mi=20; se=15
startEventFlag=0
endTimeArr=[]

cnt = 0
write_wb = Workbook()
write_ws = write_wb.active
write_ws['A1']="traceID"
write_ws['B1']="eventID"
write_ws['C1']="runTimeYear"
write_ws['D1']="runTimeMonth"
write_ws['E1']="runTimeDay"
write_ws['F1']="runTimeHour"
write_ws['G1']="runTimeMinute"
write_ws['H1']="endTime"
write_ws['I1']="remainDay"
write_ws['J1']="remainMinute"
write_ws['K1']="executedDay"
write_ws['L1']="executedMinute"
write_ws['M1']="remainTime"
while True:
    line = inputFile.readline()
    if not line: break


    
    # mata data
    if flagKey == 10:
        #print("flag 10")
        if "<trace>" in line: 
            flagKey=20 # start trace
            print("start Trace")
        elif "<global scope=\"event\">" in line: flagKey = 12 # get global event attribute          
    elif flagKey == 12:
        if "</global>" in line: flagKey=10
        
    # trace
    elif flagKey == 20:
        if "<event>" in line: 
            flagKey=30 # start event
        elif "concept:name" in line:
            traceId = line.split('\"')[3]
        elif "</trace>" in line:
            print("end trace")
            flageKey=10
            

            cnt = 0      #초기화
            
    # event
    elif flagKey == 30:
        if "concept:name" in line:
            eventId = line.split('\"')[3]
        
        elif "time:timestamp" in line:
            arr = line.split('\"')
            arrD = arr[3].split('T')

            arrDate = arrD[0].split('-')
            y=int(arrDate[0]);m=int(arrDate[1]);d=int(arrDate[2])

            arrTime = arrD[1].split('+')
            arrTime = arrTime[0].split(':')
            h=int(arrTime[0]);mi=int(arrTime[1])

            eventTime = datetime.datetime(y, m, d, h, mi, 0, 0)
            
            if startEventFlag==0 :
                startEventFlag = 1
                startEventTime = eventTime
                
            excutedTime = eventTime-startEventTime
            
        elif "</trace>" in line: 
            flagKey=20  # end trace
            startEventFlag = 0
            endTimeArr.append(eventTime)
            #write_ws.append([eventTime])            
            
        elif "</event>" in line:
            cnt += 1
            write_ws.append([traceId, eventId, y, m, d, h, mi,0,0,0,excutedTime.days,int(excutedTime.seconds/60)])
    else :
        print("Error")
    
write_wb.save("dataSet.xlsx")
write_wb.close()

print("Save File")


# In[4]:


from openpyxl.reader.excel import load_workbook
from openpyxl import load_workbook

load_wb = load_workbook("C:/Users/hahn/Google 드라이브/Colab Notebooks/ppm/remainingTime_shham_190805/dataSet.xlsx", data_only=True)
load_ws = load_wb['Sheet']

endCnt = 0
load_year = load_ws.cell(2,3).value
load_month = load_ws.cell(2,4).value
load_day = load_ws.cell(2,5).value
load_hour = load_ws.cell(2,6).value
load_minute = load_ws.cell(2,7).value

eventTime = datetime.datetime(load_year,load_month,load_day,load_hour,load_minute, 0, 0)
remainTime = endTimeArr[endCnt] - eventTime
load_ws.cell(2,8,endTimeArr[endCnt])
load_ws.cell(2,9,remainTime.days)
load_ws.cell(2,10,int(remainTime.seconds/60))
load_ws.cell(2,13,int(remainTime.days*24*60+remainTime.seconds/60))

for i in range(1,cnt):
    if load_ws.cell(i+2,1).value != load_ws.cell(i+1,1).value:
        endCnt += 1
    load_year = load_ws.cell(i+2,3).value
    load_month = load_ws.cell(i+2,4).value
    load_day = load_ws.cell(i+2,5).value
    load_hour = load_ws.cell(i+2,6).value
    load_minute = load_ws.cell(i+2,7).value
    eventTime = datetime.datetime(load_year,load_month,load_day,load_hour,load_minute, 0, 0)
    remainTime = endTimeArr[endCnt] - eventTime
    load_ws.cell(i+2,8,endTimeArr[endCnt])
    load_ws.cell(i+2,9,remainTime.days)
    load_ws.cell(i+2,10,int(remainTime.seconds/60))
    load_ws.cell(i+2,13,int(remainTime.days*24*60+remainTime.seconds/60))

load_wb.save('dataSet2.xlsx')
load_wb.close()


# In[ ]:


import pandas_datareader.data as pdr
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import argparse
from copy import deepcopy # Add Deepcopy for args
from sklearn.metrics import mean_absolute_error
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook


# In[ ]:


data = pd.read_excel('C:/dev/dataset/dataSet2.xlsx')
traceID = data['traceID'].values
eventID = data['eventID'].values
remainTime = data['remainTime'].values


# In[ ]:


data.head()


# In[6]:


w_count = {}
max=0
for lst in traceID:
    try: 
        w_count[lst] += 1 
        if max < w_count[lst]: 
            max=w_count[lst]
    except: w_count[lst]=1

print(max)
print(len(w_count))
train_num = int(len(w_count)*0.8)
print(train_num)


# In[7]:


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
        vec[self.word2idx[word]] =1
        return vec


# In[8]:


dic = Dictionary()
for tok in eventID:
    dic.add_word(tok)
print(dic.word2idx)


# In[9]:


wb = Workbook()
ws = wb.active
#ws.append([])


# In[ ]:




