#I tried a lot of stuff here.
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

#Data management for stocks.
#This is very ugly, most gets thrown in an object next. If I had been doing this enterprize, this should all be on a 
#db. But I wasn't and I was mostly just curious.
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

#This is that object I talked about. Should've used setters and getters, may have taken out that stuff to enable
#Trying parallel stuff further down the doc and was using globals to run threads.
class botFrame:
    def __init__(self, ticker):
        self.prices = []
        self.volumes = []
        self.averages = []
        self.cash = 100.0
        self.position  = 0.0
        #These are all stock terms.
        #indexes 0-2 are the 10 min aves,3-4 are the 15 min, and 5 is the 30 min
        #There are 69(An accident because I wanted a clean 30 min of prices chunk in
        #the ) NEURAL NETWORK INPUT NEURONS.
        #Basicially, every min, the price of a stock will flow into the 30 min chunk.
        #From that, some relevant stats are calculated. delete partition overflow.
        #Send to NN outputs a prediction of buy/sell proportion.
        #Entry points on nn are stored as a vector of floats.
        
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
        #print(self.input





#Some neurons. Trying to brute force the problem with deep. Not the problem in a ever-changing 69 float enviornment.
#Basically, the gist with this was twofold. 1: Primitive Backprop classifier, 2:Geometric Genetic Algorithm.
#Those will be covered when we get to the training functions for them. For now, this is what is on the tin.
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
     
     #More experimenting, didn't feel like doing an if statement would do in enterprise.
    #I think adam panned out the best here, but there's no attention, mayble a different optimizer is better with attention)
    def init_optimizer(self):

        #self.optimizer = torch.optim.SGD(self.parameters(), lr = self.alpha)
        #self.optimizer = torch.optim.Adam(self.parameters(), lr = self.alpha)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr = self.alpha)

    #Doctor I studied under had a soft spot for CrossEntropy
    #Think it performed better too, the extra stability in
    #all the noise really helped.
    def objective(self, preds, labels):

        obj = torch.nn.CrossEntropyLoss()
        #obj = torch.nn.L1Loss()

        return obj(preds, labels)

    def forward(self, x):

        return self.network(x)   

#Ok, so did you know there was a library that lets you extract the vectors out of pytorch?
#We can treat those as chromosomes.
#I SENSE RESONABLE SKEPTICISM for a couple reasons. #1 Genetic Algorithms don't have
#to converge, and finding the right spot is really hard. #2 Genetic Algorithms don't traditionally
#perform well on a large, noisy search space. If only one could reduce noise. FOCUS or something... idk.
#Anyway. I'm not going to go over the theroy of how this works in the doc, it would be too long...
#SO here's a paper I wrote. https://docs.google.com/document/d/1P8zF9v1qJe_AXndhVCdiIZGvrZd0hFNdlXATOSjJqnc/edit?usp=sharing (on anyone can view with link.)
#Spoiler alert, It uses squares, uniform probability, and was much faster on large search spaces.
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

#Ok, this is a few years old, but it was basically the first time I messed with threading to
#get extra performance. If you want spoilers to see where this goes, check the bottom of the doc.
#What this started as was my first GA. Then, because it was slow, I tried to parallelize evaluation.
#This resulted in moderate speedup, but eventually, I outgrew arrays and went to np arrays and 
#there was a better soulution that didn't use python's basic multithreading.
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
    #Early multithreading, right?
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
    

#Parallelised Genetic Algorithm Evaluation
#Multiple offspring at once. Produced resonable speedup using arrays(check old commits.)
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


#Data loaders.
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
#CUDA V11.7.64 may work for this device, otherwise try 12.1. I got it working, but local CUDA insalls are a nightmare.
reload()
local_dir = os.path.dirname(__file__)
#config_path =os.path.join(local_dir, 'CONFIG.txt')
#learn_history(config_path)
print(torch.cuda.is_available())
gen_alg(400,0.000025,400,10,200,35)
