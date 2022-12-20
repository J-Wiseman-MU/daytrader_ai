from copyreg import pickle
from hashlib import new
from math import floor, isclose
import math
from operator import ge, truediv
import os
from re import T
from turtle import position
from xmlrpc.client import DateTime
import neat
import random
import multiprocessing
import datetime
from datetime import timedelta
from datetime import datetime
import time
from polygon import RESTClient
from typing import cast
from urllib3 import HTTPResponse
import numpy as np
import statistics
import pickle
import graphviz
from itertools import count
import torch
import torch.nn as nn
import pygad
import pygad.torchga as tga
import numpy as np
import random
from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean

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





class MLP(torch.nn.Module):
    
    def __init__(self):
    
        super(MLP, self).__init__()
    
        self.alpha = 1e-3
        
        nonlin = torch.nn.LeakyReLU()
        self.network = torch.nn.Sequential(torch.nn.Linear(69,136 ),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136,136),
                                           nonlin,
                                           torch.nn.Linear(136, 2))
     
        
    def init_optimizer(self):

        #self.optimizer = torch.optim.SGD(self.parameters(), lr = self.alpha)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = self.alpha)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr = self.alpha)

    def objective(self, preds, labels):

        obj = torch.nn.CrossEntropyLoss()
        #obj = torch.nn.L1Loss()

        return obj(preds, labels)

    def forward(self, x):

        return self.network(x)   

class botFrame:
    def __init__(self, ticker):
        self.prices = []
        self.volumes = []
        self.averages = []
        self.cash = 100.0
        self.position  = 0.0
        #indexes 0-2 are the 10 min aves,3-4 are the 15 min, and 5 is the 30 min
        self.currmin = 0
        for x in range(30):
            self.prices.append(-1.0)
            self.volumes.append(-1.0)
        for x in range(6):
            self.averages.append(-1.0)
        self.name = ticker
    def update(self,newPrice,newVol):
        self.prices.pop(0)
        self.prices.append(float(newPrice))
        if(self.currmin > 29):
            self.position = (self.position * (self.prices[29] / self.prices[28]))
        elif(self.currmin > 0):
            self.position = (self.position * (self.prices[self.currmin] / self.prices[self.currmin - 1]))
        self.volumes.pop(0)
        self.volumes.append(float(newVol))
        tensplit = np.array_split(self.prices,3)
        fifteensplit = np.array_split(self.prices,2)
        #print(type(tensplit[0][0]))
        if self.currmin > 9:
            self.averages[0] = statistics.mean(tensplit[0])
        if self.currmin > 14:
            self.averages[3] = statistics.mean(fifteensplit[0])
        if self.currmin > 19:
            self.averages[1] = statistics.mean(tensplit[1])
        if self.currmin > 29:
            self.averages[2] = statistics.mean(tensplit[2])
            self.averages[4] = statistics.mean(fifteensplit[1])
            self.averages[5] = statistics.mean(self.prices)
        self.currmin += 1
        a = self.prices + self.volumes + self.averages + [self.cash] + [self.position] + [float(self.currmin)]
        n = np.array([a])
        #print(n)
        self.input = torch.from_numpy(n).to(dtype=torch.float32)
        #print(self.input)
                
def get_data():
    global data
    global errors
    global totaldays
    client = RESTClient("qOzFjU4nysnLGV5mK9xVZQibcbknHwaW")
    for x in range(50):
        data.append([])
        data[x].append([])
        data[x].append([])
    global stamp
    for num in reversed(range(28)):
        day = (stamp - (num*86400000))
        close = day + 23400000
        temp = datetime.fromtimestamp(day/1000)
        if num > 1 and temp.weekday() < 4:
            for x in range(50):
                aggs = client.stocks_equities_aggregates(tickers[x], 1, "minute", str(int(day)), str(int(close)))
                if(len(aggs.results) <390):
                    if tickers[x] not in errors:
                        errors.append(tickers[x])
                for i in range(390):
                    data[x][0].append(aggs.results[i]["o"])
                    data[x][1].append(aggs.results[i]["v"])
                time.sleep(12)
    totaldays = (len(data[0][0])/390) - 1
    condense()


def historic_eval_genomes(genomes,config):
    #impliment non-live evaluation and training of nns
    #Step 1, assign each genome to a random stock at the start of each generation
    global selection
    global tickers
    global totaldays
    global dayiterator
    global bots
    selection.clear()
    bots.clear()
    placements = []
    for x in range(len(generation)):
        placements.append(random.randint(0,49))
    for x in placements:
        selection.append(tickers[x])
    #threads = multiprocessing.Pool()
    for x in selection:
        bots.append(botFrame(x))
    #Step 2, call simulate in a paralellized fasion to get the final monetary balance of each net
    #threads.map(simulate,genomes)
    for g in genomes:
        simulate(g)
    #Step 3, incriment the day
    if dayiterator == totaldays:
        dayiterator = 0
    else:
        dayiterator += 1
    
def gen_alg(maxgen,mute_rate,pop_size,elitism,survivors,threshold):
    global selection
    global tickers
    global totaldays
    global dayiterator
    global bots
    selection.clear()
    bots.clear()
    placements = []
    model = MLP()
    init_chromosome = tga.model_weights_as_vector(model)
    population = []
    for i in range(pop_size):
        #print(np.shape(init_chromosome))
        population.append([np.random.uniform(low=-1,high=1,size=np.shape(init_chromosome)[0]),0.0])
    model.eval()
    for i in range(maxgen):
        for x in range(pop_size):
            placements.append(random.randint(0,49))
        for x in placements:
            selection.append(tickers[x])
            #thr    eads = multiprocessing.Pool()
        for x in selection:
            bots.append(botFrame(x))
        for j in range(pop_size):
            model.load_state_dict(tga.model_weights_as_dict(model=model,weights_vector=population[j][0]))
            population[j][1] = simulate(model,j)
            print(str(j) + ":" + str(population[j][1]) + ",", end=" ")
        population.sort(reverse=True, key = lambda x : x[1])
        print("\n" + str(i) + ":" + str(population[0][1]) + "\n")
        chain = population[:elitism] + random.sample(population[elitism:],(survivors-elitism))
        chain = chain[:threshold]
  #print("CHAIN\n")
  #print(chain)
        population.clear()
        for j in range(elitism):
            population.append(chain[j])
        pool = 0
        for j in range(pop_size-elitism):
            if pool == threshold: pool = 0
            parents = random.randint(0,threshold-1)
            population.append([np.random.uniform(chain[pool][0],chain[parents][0]),0.0])
            for k in range(np.shape(init_chromosome)[0]):
                if(random.uniform(0,1) < mute_rate): 
                    population[j][0][k] = random.uniform(-4,4)
        pool += 1


def simulate(model,index):
    #Step 1:get stock data for current day and the ticker in selection that matches the index of the current genome and genarate the genome's net
    global bots
    global data
    global dayiterator
    global totaldays
    global tickers
    earnings = []
    if dayiterator == totaldays:
        dayweights = [dayiterator - 2,dayiterator - 1,dayiterator]
    elif dayiterator == 0:
        dayweights = [dayiterator,dayiterator + 1,dayiterator + 2]
    else:
        dayweights = [dayiterator - 1,dayiterator,dayiterator + 1]
    #data = yf.download(bots[index].name,start = (((currentDay.strftime("%Y %m %d")).replace(" ","-")).replace("-0","-")),end = ((((currentDay + timedelta(days=1)).strftime("%Y %m %d")).replace(" ","-")).replace("-0","-")),period="1d",interval="1m")
    #Step 2:iterate through the day, adding the appropriate data to the bots percep variables, -1 could mean no data
    botindex = tickers.index(bots[index].name) 
    for y in range(totaldays):
        dayweight = y * 390
        acted = 0
        slow = False
        for x in range(390):
    #Step 2:Data includes last 30 min stock prices for particular stock and 1 30min,2 15min,and 3 10min rolling averages
    #Step 2:Formula for rolling ave for 10min would be (p1+p2+p3+...+p10)/10
    #Step 2:Update the value of the bot's holdings using the formula (Holdings)*(currentprice/priceatprevmin)
            bots[index].update(data[botindex][0][x + dayweight],data[botindex][1][x + dayweight])
    #Step 3:Start a timer and have the network generate a buy/sell with the given data
            start = time.time()
            #print(bots[index].input)
            output = (torch.nn.functional.normalize(model(bots[index].input),p=2.0,dim = 1)).detach().numpy()[0]
    #Step 4:If the network took less than 2sec and a buy/sell value is a dollar or more and conforms with how much money/holdings it has, then buy/sell
            if((time.time() - start) < 1.2):
            #output[0] is buy and output [1] is sell
                buy = output[0] * bots[index].cash
                sell = output[1] * bots[index].position
                #print("buy:" + str(output[0]) + ", " + "sell:" + str(output[1]) + ";",end=" ")
                if(buy > 1.0 and buy < bots[index].cash):
                    #print("BOUGHT! ",end=" ")
                    if acted == 0:
                        acted += 1
                    bots[index].cash -= buy
                    bots[index].position += buy
                elif(sell > 1.0 and sell < bots[index].position):
                    #print("SOLD! ",end=" ")
                    if acted == 1:
                        acted += 1
                    bots[index].cash += sell
                    bots[index].position -= sell
            else:
                print("Tooslow")
                slow = True
        #print(acted)
        if(acted >= 2 and not slow):
            #print("success")
            if bots[index].cash > 100.0:
                #print(str(index) + ": " + str(bots[index].cash))
                earnings.append(bots[index].cash)
            else:
                earnings.append(bots[index].cash)
        else:
            earnings.append(-150.0)
            #break
    #Step 5:after iteration, update the fitness val of the genome to its balance unless it performed no actions, in such case, set fitness to 0
    if -1.0 in earnings:
        return -150.0
    else:
        meanearnings = statistics.mean(earnings)
        return meanearnings - 150.0


#run(config_path)
#print(errors)
#for x in range(50):
def condense():
    global data
    global currentDay
    #dbfile = open('pricedatafile.text', 'wb',pickle.HIGHEST_PROTOCOL)
    constamp = datetime.timestamp(currentDay)
    with open('pricedatafile.text','wb') as dbfile:
        bundle = [data , constamp]
        pickle.dump(bundle,dbfile)
    
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


#a = [1,1,1,1,1,1,1,1,1,1,4]
#n = np.array([a])
#t= torch.from_numpy(n)
#print(t)
#get_data()
reload()
#local_dir = os.path.dirname(__file__)
#config_path =os.path.join(local_dir, 'CONFIG.txt')
#learn_history(config_path)
#print(torch.cuda.is_available())
gen_alg(400,0.000025,400,10,200,35)