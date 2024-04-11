from copyreg import pickle
from hashlib import new
from math import floor, isclose
import math
from operator import ge, truediv
import os
from re import T
#from turtle import position
from xmlrpc.client import DateTime
#import neat
import random
import multiprocessing as mp
import datetime
from datetime import timedelta
from datetime import datetime
import time
#from polygon import RESTClient
from typing import cast
import numpy as np
import statistics
import pickle
#import graphviz
from itertools import count
import torch
import torch.nn as nn
import pygad
import pygad.torchga as tga
import numpy as np
import random
#from neat.config import ConfigParameter, DefaultClassConfig
#from neat.math_util import mean
import threading
from collections import deque

selection = []
tickers = ["AAPL","MSFT","AMZN","TSLA","META","JNJ","NVDA","V","XOM","WMT","JPM","CVX","PFE","KO","BAC","ABBV","MRK","ORCL","DIS","MCD","CRM","VZ","CSCO","NKE","NEE","CMCSA","TXN","QCOM","WFC","AMD","BMY","INTC","RTX","CVS","SCHW","T","MDT","BX","COP","IBM","PYPL","NFLX","C","SBUX","AMAT","MDLZ","GE","MO","GILD","KR"]
bots = []
data = []
#data [x][0] is the price data of the xth stock in the ticker list [x][1] is the volume array
errors =[]
now = datetime.today()
currentDay = now - timedelta(days=1)
marketOpen = datetime(currentDay.year,currentDay.month,currentDay.day,8,30,0,0)
stamp = int(marketOpen.timestamp() * 1000)
totaldays = 0
dayiterator = 0
fillerdays = 0
sequence = []
para_earnings = []
para_population = []
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

#Business Logic is the same.
def reload():
    global fillerdays
    global currentDay
    global data
    global totaldays
    dbfile = open('pricedatafile.text', 'rb')
    self =  pickle.load(dbfile)
    delta = currentDay - datetime.fromtimestamp(self[1])
    data = self[0]
    fillerdays = delta.days
    totaldays = floor(len(data[0][0])/390)
    dbfile.close()


#step 1.A:Create numpy dataset of sample minutes and labels.
class Dataset(torch.utils.data.Dataset):

    def __init__(self, samples, labels, device):
        super(Dataset, self).__init__()

        self.create_dataset(samples, labels, device)

    def create_dataset(self, all_samples, all_labels, device):
        self.dataset = []

        for sample, label in zip(all_samples, all_labels):
            self.dataset.append((torch.tensor(sample).float().to(device), torch.tensor(label).long().to(device)))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

#Simple backprop MLP
class MLP(torch.nn.Module):

    def __init__(self, learning_rate):
        super(MLP, self).__init__()

        self.alpha = learning_rate

        self.network = torch.nn.Sequential(torch.nn.Linear(69, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(128, 64),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(64, 3))

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.alpha)

    def objective(self, preds, labels):
        obj = torch.nn.CrossEntropyLoss()

        return obj(preds, labels)

    def forward(self, x):
        return self.network(x)


def calculate_accuracy(all_preds, all_labels):
    accuracy = 0
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):

        pred = np.argmax(pred)

        if (pred == label):
            accuracy = accuracy + 1

    return accuracy / (i + 1)

#This was the beating heart of the algorithm. Using future prices, it was supposed to return a buy or a sell at every step so that profit would be maximized.
def find_best_actions(prices):
    actions = [2] * 390
    max_profit = 0
    min_price = float('inf')
    buy_minute = 0

    for i in range(len(prices)):
        if prices[i] < min_price:
            min_price = prices[i]
            buy_minute = i

        potential_profit = prices[i] - min_price

        if potential_profit > max_profit:
            max_profit = potential_profit
            actions[buy_minute] = 0
            actions[i] = 1
            # Reset min_price and buy_minute for the next buy opportunity
            min_price = float('inf')
            buy_minute = 0
            max_profit = 0

    return actions

#Useful class that handles changes in stock price.
class botFrame:
    def __init__(self, ticker):
        self.prices = []
        self.volumes = []
        self.averages = []
        self.cash = 100.0
        self.pos  = 0.0
        #indexes 0-2 are the 10 min aves,3-4 are the 15 min, and 5 is the 30 min
        self.currmin = 0
        for x in range(30):
            self.prices.append(-1.0)
            self.volumes.append(-1.0)
        for x in range(6):
            self.averages.append(-1.0)
        self.name = ticker
        a = self.prices + self.volumes + self.averages + [self.cash] + [self.pos] + [float(self.currmin)]
        self.n = np.array([a])
    def update(self,newPrice,newVol):
        self.n = np.delete(self.n,0,axis=1)
        self.n = np.insert(self.n,29,float(newPrice),axis=1)
        #if(self.n[0][68] > 29):
        self.n[0][67] = (self.n[0][67] * (self.n[0][29] / self.n[0][28]))
        #elif(self.n[0][68] > 0):
        #    self.n[0][67] = (self.n[0][67] * (self.n[0][int(self.n[0][68])] / self.n[0][int(self.n[0][68]) - 1]))
        self.n = np.delete(self.n,30,axis=1)
        self.n = np.insert(self.n,59,float(newVol),axis=1)
        tensplit = np.array_split(self.n[0][:30],3)
        fifteensplit = np.array_split(self.n[0][:30],2)
        if self.n[0][68] > 9:
            self.n[0][60] = np.mean(tensplit[0])
        if self.n[0][68] > 14:
            self.n[0][63] = np.mean(fifteensplit[0])
        if self.n[0][68] > 19:
            self.n[0][61] = np.mean(tensplit[1])
        if self.n[0][68] > 29:
            self.n[0][62] = np.mean(tensplit[2])
            self.n[0][64] = np.mean(fifteensplit[1])
            self.n[0][65] = np.mean(self.prices)
        self.n[0][68] += 1
        self.input = torch.tensor(self.n,dtype=torch.float32)

#Think the dataset object from earlier, but for reinforcement learning.
def npFormatting():
    global fillerdays
    global currentDay
    global data
    global totaldays
    train_sample, train_label, valid_sample, valid_label = [], [], [], []
    day = stock_day_full(0,0)
    for stock in range(50):
        for daynum in range(totaldays-1):
            day.reset(stock,daynum)
            label = find_best_actions(data[stock][0][(daynum*390):(daynum*390 + 390)])
            i = 0
            for choice in label:
                #print(str(data[stock][0][(daynum*390) + i]) + ":",end="")
                i += 1
                train_label.append(choice)
                #print(choice)
                train_sample.append(np.copy(day.get_state()))
                day.step(choice)
            #print(day.value)
        day.reset(stock, totaldays - 1)
        #future actions??? idk something seems off with this.
        label = find_best_actions(data[stock][0][(daynum * 390):(daynum * 390 + 390)])
        i = 0
        for choice in label:
            # print(str(data[stock][0][(daynum*390) + i]) + ":",end="")
            i += 1
            valid_label.append(choice)
            valid_sample.append(np.copy(day.get_state()))
            day.step(choice)
    #print(train_label)
    return np.vstack(train_sample),np.hstack(train_label),np.vstack(valid_sample),np.hstack(valid_label)

#Class that allows one to run an entire stock day minute by minute, choosing to buy, sell, or hold at each minute.
#This is a helper function for the reinforment optimization algorithm. Reinforcement learning requires some of these
#helpers in order to peek into the future. Not entirely sure this is wired up correctly. Second set of eyes may be nice.
class stock_day_full:
    def __init__(self,index,day):
        global data
        global totaldays
        global tickers
        global sequence
        self.botindex = index#np.random.randint(0,50)
        self.frame = botFrame(tickers[self.botindex])
        self.dayweight = day * 390#np.random.randint(0,totaldays) * 390
        self.frame.update(data[self.botindex][0][0 + self.dayweight],data[self.botindex][1][0 + self.dayweight])
        self.min = 0
        self.bought = 0
        self.value = 100

    def reset(self,index,day):
        global data
        global totaldays
        global tickers
        global sequence
        self.botindex = index#np.random.randint(0, 50)
        self.frame = botFrame(tickers[self.botindex])
        self.dayweight = day * 390#np.random.randint(0, totaldays) * 390
        self.frame.update(data[self.botindex][0][0 + self.dayweight], data[self.botindex][1][0 + self.dayweight])
        self.min = 0
        self.bought = 0
        self.value = 100

    def step(self, action):
        reward = 0.0
        if(action == 0):
            buy = self.frame.n[0][66]
            sell = 0.0
        elif action == 1:
            sell = self.frame.n[0][67]
            buy = 0.0
        else:
            sell, buy = 0.0, 0.0
        if (buy > 1.0):
            # print("BOUGHT! ",end=" ")
            #acted += 1
            self.frame.n[0][66] -= buy
            self.frame.n[0][67] += buy
            #reward += 0.01
            self.bought += buy
        elif (sell > 1.0):
            # print("SOLD! ",end=" ")
            #acted += 1
            self.frame.n[0][66] += sell
            self.frame.n[0][67] -= sell
        #print(self.min)
        if self.min < 389:
            self.min += 1
            #print(data[self.botindex][0][self.min + self.dayweight - 1])
            self.frame.update(data[self.botindex][0][self.min + self.dayweight], data[self.botindex][1][self.min + self.dayweight])
        self.value = self.frame.n[0][66] + self.frame.n[0][67]

    def get_state(self):
        return self.frame.n

    def get_ticker_day(self):
        return (self.botindex,int(self.dayweight/390))

#If neither of these work, it's because of the dreaded "cuda v pytorch versions problem. This is why we use docker."
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
reload()
ts, tl, vs, vl = npFormatting()
batch_size = 16
train_dataset = Dataset(ts,tl,device)
valid_dataset = Dataset(vs,vl,device)
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
valid_dataset = torch.utils.data.DataLoader(valid_dataset, batch_size = 1, shuffle = False)
model = MLP(1e-2)
model.to(device)

model.init_optimizer()

# Train and validate network

num_epochs = 10000

training_loss, training_acc = [], []

for epoch in range(num_epochs):

    # Train network

    model.train()
    print(f"Epoch:{epoch}")
    start = time.time()
    epoch_loss = 0
    for i, (sample, label) in enumerate(train_dataset):
        preds = model(sample)
        #print(f"Pred:{torch.argmax(preds, dim=1)}")
        #print(f"Labl:{label}")

        loss = model.objective(preds, label)
        #print(loss)
        epoch_loss = epoch_loss + loss.item()

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    epoch_loss = epoch_loss / (i + 1)


    training_loss.append(epoch_loss)

    # Validate network

    model.eval()

    all_preds = []
    for i, (sample, label) in enumerate(valid_dataset):
        preds = model(sample)

        all_preds.append(preds.cpu().detach().numpy())

    # print(all_preds)

    epoch_accuracy = calculate_accuracy(np.asarray(all_preds), vl)
    print(f"Epoch Loss:{epoch_loss}, Epoch Accuracy:{epoch_accuracy}, Time:{time.time() - start}s")
    training_acc.append(epoch_accuracy)

