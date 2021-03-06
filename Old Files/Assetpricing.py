from __future__ import division
import numpy as np
from numpy.linalg import inv
import time
from scipy import linalg as la
from scipy import optimize
import scipy
from matplotlib import pyplot as plt 
import math
import csv
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
 
dates=[]
Tbill=[]
inflation=[]
Will5000=[]
RealC=[]
 
 
 
with open('revised.csv','r') as csv_file: #Takes a csv file with two columns and creates two lists of male and female tributes.
    csv_reader=csv.reader(csv_file)
    for date , WILL5000 ,inflations , TBILL , Realcons in csv_reader:
        for line in csv_reader:
            dates.append(line[0])
            Will5000.append(line[1])
            inflation.append(line[2])
            Tbill.append(line[3])
            RealC.append(line[4])
 
def destring(arrayin):
    a=len(arrayin)
    for i in xrange(a):
        arrayin[i]=float(arrayin[i])
 
    return arrayin
 
destring(Tbill)
destring(inflation)
destring(Will5000)
destring(RealC)
 
datearray=np.asarray(dates)
Tbillarray=np.asarray(Tbill)
Cpiarray=np.asarray(inflation)
Will5000array=np.asarray(Will5000)
RealCarray=np.asarray(RealC)
 
 

m = RealCarray[1:]/RealCarray[:-1]
realrisky=((Will5000array[1:]/Will5000array[:-1]) - (Cpiarray[1:]/Cpiarray[:-1]) + 1)
realriskless=((1+(Tbillarray[1:]/100.))**(1/4) - (Cpiarray[1:]/Cpiarray[:-1]) + 1)

#print Tbillarray
#print Inflationarray
print realriskless
print realrisky
print m
print len(realriskless)
print len(realrisky)
print len(m)
 
identity=np.identity(2)
print identity






def f(input):
    n = len(m)
    n1 = (np.sum((input[0]*m**(-input[1])*realrisky)/(n-1))-1)
    n2 = (np.sum((input[0]*m**(-input[1])*realriskless)/(n-1))-1)
    vec = np.array([n1,n2])
    vect = np.transpose(vec)
    one = np.dot(vec, identity)
    two = np.dot(one, vect)
    return two
bnds = ([(0,1),(0,10)])
print "unbounded optimization GAMMAS CHANGE"
print scipy.optimize.minimize(f, [.5,1],  method = "BFGS")
print "unbounded optimization GAMMAS CHANGE"
print scipy.optimize.minimize(f, [.5,5],  method = "BFGS")
print "unbounded optimization GAMMAS CHANGE"
print scipy.optimize.minimize(f, [.5,10],  method = "BFGS")
print "unbounded optimization BETAS CHANGE"
print scipy.optimize.minimize(f, [.1,1],  method = "BFGS")
print "unbounded optimization BETAS CHANGE"
print scipy.optimize.minimize(f, [.5,1],  method = "BFGS")
print "unbounded optimization BETAS CHANGE"
print scipy.optimize.minimize(f, [.9,1],  method = "BFGS")

print "bounded optimization"
print scipy.optimize.minimize(f, [.5,1],  method = "L-BFGS-B", bounds = bnds, tol = 10**-15) 
print "bounded optimization"
print scipy.optimize.minimize(f, [.5,5],  method = "L-BFGS-B", bounds = bnds, tol = 10**-15) 
print "bounded optimization"
print scipy.optimize.minimize(f, [.5,10],  method = "L-BFGS-B", bounds = bnds, tol = 10**-15) 
print "bounded optimization BETAS CHANGE"
print scipy.optimize.minimize(f, [.1,5],  method = "L-BFGS-B", bounds = bnds, tol = 10**-15) 
print "bounded optimization"
print scipy.optimize.minimize(f, [.5,5],  method = "L-BFGS-B", bounds = bnds, tol = 10**-15) 
print "bounded optimization"
print scipy.optimize.minimize(f, [.9,5],  method = "L-BFGS-B", bounds = bnds, tol = 10**-15) 

solution = scipy.optimize.minimize(f, [.5,5],  method = "BFGS")
values =  solution['x']

beta = np.linspace(-1,2,100)
gamma = np.linspace(-10,100,100)
X,Y = np.meshgrid(beta, gamma)

def f2(x,y):
    temp = np.array([m**x[i] for i in xrange(len(x))])
    n1 = np.array([x*np.dot(m**-y[i], realrisky)/(len(m)*1.0-1.0) - 1 for i in xrange(len(y))])
    n2 = np.array([x*np.dot(m**-y[i], realriskless)/(len(m)*1.0-1.0) - 1 for i in xrange(len(y))])
    return n1**2 + n2**2
 
Z = f2(beta,gamma)
Z = np.array(Z)

fig1 = plt.figure(1)
ax = fig1.gca(projection = '3d')
ax.set_xlabel("Beta")
ax.set_ylabel("Gamma")
plt.title('Problem 1')
ax.plot_surface(X, Y, Z)
 
Gx, Gy = np.gradient(Z) # gradients with respect to x and y G = (Gx**2+Gy**2)**.5  # gradient magnitude ќ N = G/G.max()  # normalizefig.png 0..1 surf = ax.plot_surface( X, Y, Z, rstride=1, cstride=1, facecolors=cm.jet(N), linewidth=0, antialiased=False, shade=False) plt.show() orderbook.png#2

dates=[]
Tbill=[]
inflation=[]
Will5000=[]
RealC=[]
Nasdaq=[] 
Gold=[]
Housing=[]
 
 
with open('revisedNGH.csv','r') as csv_file:
    csv_reader=csv.reader(csv_file)
    for date , WILL5000 ,inflations , TBILL , Realcons, NASDAQ, GOLD, HOUSING in csv_reader:
        for line in csv_reader:
            dates.append(line[0])
            Will5000.append(line[1])
            inflation.append(line[2])
            Tbill.append(line[3])
            RealC.append(line[4])
            Nasdaq.append(line[5]) 
            Gold.append(line[6])
            Housing.append(line[7])

def destring(arrayin):
    a=len(arrayin)
    for i in xrange(a):
        arrayin[i]=float(arrayin[i])
 
    return arrayin
 
destring(Tbill)
destring(inflation)
destring(Will5000)
destring(RealC)
destring(Nasdaq)
destring(Gold)
destring(Housing)

datearray=np.asarray(dates)
Cpiarray=np.asarray(inflation)
RealCarray=np.asarray(RealC)
Tbillarray=np.asarray(Tbill)
Will5000array=np.asarray(Will5000)
Nasdaqarray=np.asarray(Nasdaq)
Goldarray=np.asarray(Gold)
Housingarray=np.asarray(Housing)

inflation = Cpiarray[1:]/Cpiarray[:-1]
willgrowth = Will5000array[1:]/Will5000array[:-1]
nasdaqgrowth = Nasdaqarray[1:]/Nasdaqarray[:-1]
goldgrowth = Goldarray[1:]/Goldarray[:-1]
housinggrowth = Housingarray[1:]/Housingarray[:-1]

realwill5000=(willgrowth - inflation + 1)
realtbill=((1+(Tbillarray[1:]/100.))**(1/4) - inflation + 1)
realnasdaq=(nasdaqgrowth-inflation+1)
realgold=(goldgrowth-inflation+1)
realhousing=(housinggrowth-inflation+1)



def g(input):
    beta = input[0]
    alpha = input[1]
    phi = input[2]
    gamma = input[3]
    m = ((alpha*RealCarray[1:] + phi*gamma)/(alpha*RealCarray[:-1]+phi*gamma))**-gamma 
    n = len(m)


    n1 = beta*(np.dot(m,realwill5000)/(n-1))-1
    n2 = beta*(np.dot(m,realtbill)/(n-1))-1
    n3 = beta*(np.dot(m,realnasdaq)/(n-1))-1
    n4 = beta*(np.dot(m,realgold)/(n-1))-1
    n5 = beta*(np.dot(m,realhousing)/(n-1))-1


    N = [n1, n2, n3, n4, n5]  
    I = np.identity(5)
    M = np.dot(N,np.dot(I,N))
    return M

print "unbounded"
print optimize.minimize(g, [.01,.01,.01,.01])

import numpy as np
from matplotlib import pyplot as plt
import pandas
import csv
 
print "Loading data..."
orderBook = {} #This is a dictionary that will hold the orderbook
reader=csv.reader(open("GOOG_062314.csv","rb"),delimiter=',') #Read in the messages. Notice that each field is loaded as a string
x=list(reader)
messages = np.array(x)
print "Complete."
 
#We will hold the spread high and low here in these variables
#We will update them as we process each line
spreadHigh = np.inf
spreadLow = -np.inf
 
u_means_remove = True #This variable keeps track of if the next "U" message is a removal or addition
 
def process(r):
    r[4] = r[4].strip()
    #Addition
    if r[1] == "A": #Add
        addition(r[6], r[5], r[4])
 
    elif r[1] == "C": #Execute with price
        #TODO: Implement?
        #The "C" message is only 35 messages of the >300,000 messages
        #It is complicated since it means an order was executed at not the original price
        #Since this is hard to implement and is insignificant, I think we should skip it.
        return
 
    #Deletion
    elif r[1] == "D": #Delete
        code = remove(r[6], r[5], r[4])
        if code == -1:
            print "D"
 
    elif r[1] == "E": #Execute
        code = remove(r[6], r[5], r[4])
        if code == -1:
            print "E"
     
    elif r[1] == "F": #Add with MPID
        addition(r[6], r[5], r[4])
 
    elif r[1] == "P": #Execute non-displayable trade
        remove(r[6], r[5], r[4])
        return
     
    elif r[1] == "U":
        global u_means_remove
        if u_means_remove == True:
            code = remove(r[6], r[5], r[4])
            if code == -1:
               print "U"
            u_means_remove = False
        else:
            addition(r[6], r[5], r[4])
            u_means_remove = True
 
    #Execute
    elif r[1] == "X": #Cancel
        code = remove(r[6], r[5], r[4])
        if code == -1:
            print "X"
 
#Process an addition
def addition(price, numShares, b_s):
    numShares = int(numShares) #numShares is a string, so we change it to an int
    '''
    if float(price) > 5000: #there are a few orders at $200,000 which are not relevant and mess up the graph, so we simply do not add these.
        return -1
    '''
    if price in orderBook:
        if b_s == "B" or b_s == " B":
            orderBook[price] = (orderBook[price][0] + numShares, "B")
        else:
            orderBook[price] = (orderBook[price][0] + numShares, "S")
    else:
        if b_s == "B" or b_s == " B":
            orderBook[price] = (numShares, "B")
        else:
            orderBook[price] = (numShares, "S")
    return 0
    #TODO
    #Code to update the spreadHigh and spreadLow
    '''
    if r[4] == "B":
        if r[6] > spreadLow:
            spreadLow == r[6]
    else:
        if r[6] < spreadHigh:
            spreadHigh == r[6]
    '''
 
def remove(price, numShares, b_s):
    #TODO: update spread variables
    numShares = int(numShares) #numShares is a string, so we change it to an int
    if price in orderBook:
        if orderBook[price][0] >= numShares:
            orderBook[price] = (orderBook[price][0] - numShares, b_s)
        else:
            #print str(numShares) + " shares are trying to be removed at price " + str(price) + " but only " + str(orderBook[price]) + " are on the book."
            return 3
        if orderBook[price][0] == 0: #If there are no shares under that price, delete the entry
            orderBook.pop(price, None)
            return 1
        return 0
    else: 
        #Sometimes the price is not in orderBook despite a request to remove
        #This is probably due to the fact that I'm not processing updates and other things
        #This code captures when that happens and displays a message so it's clear what happened
        #print "Tried to remove " + str(price) + " from orderBook, but the key does not exist"
        return -1
 
def plotOrderBook(oBook):
    BAR_SIZE = .01
    #print "Converting dictionary to array..."
    dictlist = []
    #Just iterate through all the keys in the dictionary and put them into a list
    for key, value in oBook.iteritems():
        temp = [float(key), float(value[0]), value[1]]
        dictlist.append(temp)
    dictlist = np.array(dictlist)
    print "Plotting..."
 
    buys = np.array([dictlist[i] for i in xrange(len(dictlist)) if (dictlist[i, 2] == "B") or (dictlist[i, 2] == " B")])
    sells = np.array([dictlist[i] for i in xrange(len(dictlist)) if dictlist[i, 2] == "S"])
 
    #What we might want to do here is use np.histogram() to clean this up
    #Right now the bars overlap and there is some ugliness
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.bar(buys[:, 0].astype(float), buys[:, 1].astype(float), width=BAR_SIZE, color="blue") #On the x axis we have the prices, y axis is number of shares
    ax.bar(sells[:, 0].astype(float), sells[:, 1].astype(float), width=BAR_SIZE, color="red") #On the x axis we have the prices, y axis is number of shares
    plt.xlim((557, 559))
    plt.ylim((0,110))
    plt.show()
 
def getSpread(oBook):
    buys = []
    sells = []
    #Just iterate through all the keys in the dictionary and put them into a list
    for key, value in oBook.iteritems():
        if value[1] == "B":
            buys.append(float(key))
        else:
            sells.append(float(key))
 
    maxBuy = np.max(buys)
    minSell = np.min(sells)
    return minSell-maxBuy
 
def volumeProcess(r):
    if r[1] == "C": #Execute with price
        return int(r[5])
 
    elif r[1] == "E": #Execute
        return int(r[5])
 
    elif r[1] == "P": #Execute non-displayable trade
        return int(r[5])
    else:
        return 0
#Plot the order book...
print "Building the order book..."
for row in messages[1:]:
    if float(row[2]) >= 37800: #37800 is 10:30
        print "It's 10:30!!!"
        break
    else:
        process(row)
 
print "Order book built."
plotOrderBook(orderBook)
print "Finished plotting the order book."
 
 
#Code to calculate the average spread
orderBook = {}
stop = 34200 #First stop is at 9:30AM, opening time
CLOSING_TIME = 57600 #4PM
AVERAGE_SPREAD_CALCULATION_INTERVAL = 10
print "Calculating average spread using steps of " + str(AVERAGE_SPREAD_CALCULATION_INTERVAL) + " seconds..."
spreads = []
for row in messages[1:]:
    process(row)
    if float(row[2]) >= stop:
        spread = getSpread(orderBook)
        spreads.append(spread)
        stop += AVERAGE_SPREAD_CALCULATION_INTERVAL
    if float(row[2]) >= CLOSING_TIME:
        break
 
spreads = np.array(spreads)
average_spread = np.average(spreads)
print "AVERAGE SPREAD =" + str(average_spread*100) + " cents"
 
#Code to calculate total volume traded
print "Calculating the volume of executed trades..."
orderBook = {}
totVolume = 0
for row in messages[1:]:
    totVolume += volumeProcess(row)
 
print "TOTAL VOLUME OF EXECUTED TRADES: " + str(totVolume)
print "TOTAL PROFITS BY MARKET MAKERS: " + str(totVolume * average_spread)








