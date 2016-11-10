from __future__ import division  
import numpy as np
import matplotlib.pyplot as plt
from quantecon import mc_compute_stationary, mc_sample_path
from scipy.stats import norm
 
 
#1
print "Exercise 1"
 
alpha = .1
 
beta = .1
p =beta/(alpha+beta)
 
P = np.array([[1-alpha,alpha],[beta, 1-beta]])
 
n=10000
x1=0
x2=1
X1= mc_sample_path(P,x1,n)
X2=mc_sample_path(P,x2,n)
 
indicator1 = X1 == 0
indicator2 = X2 == 0
 
X_bar1 = indicator1.cumsum()/(1+np.arange(n))
X_bar2 = indicator2.cumsum()/(1+np.arange(n))
 
zero = np.zeros(n)
 
plt.plot(X_bar1-p, color='green',label = r'$X_0 = \, 0 $'.format(x1))
plt.plot(X_bar2-p, color='black',label = r'$X_0 = \, 1 $'.format(x2))
plt.plot(zero,color='black')
plt.legend(loc="upper right")
plt.ylim(-.3,.3)
plt.xlabel("N")
plt.ylabel("XBARNP")
plt.title("XBAR on N graph")
plt.grid(True)
plt.show()
 
 
 
#2
print "Exercise 2"
 
infile = 'web_graph_data.txt'
pgs = 'abcdefghijklmn'
indices = np.arange(0,14,1)
print indices
 
 
n = 14 # Total number of web pgs (nodes)
 
# == Create a matrix Q indicating existence of links == #
#  * Q[i, j] = 1 if there is a link from i to j
#  * Q[i, j] = 0 otherwise
pathmat = np.zeros((n, n), dtype=int)
f = open(infile, 'r')
diffpaths = f.readlines()
f.close()
 
for i in diffpaths:
    pathst = i[0]
    pathend = i[5]
    k,j = pgs.index(pathst),pgs.index(pathend)
    pathmat[k,j]=1
 
P=np.empty((n,n))
 
for i in xrange(n):
    P[i,:]=pathmat[i,:]/pathmat[i,:].sum()
 
r=mc_compute_stationary(P)[0]
 
ranks = tuple((pgs[i],r[i]) for i in xrange(n))
 
sortedranks=sorted(ranks,reverse=True,key=lambda x: x[1])
 
print "Ranks after sorting: "
for i in xrange(n):
    sortedrank = sortedranks[i]
    print "Web page:", sortedrank[0] ,"Prob. = ", sortedrank[1]
 
#3
print "Exercise 3"
 
def markovapprox(rho,sigma_u,m=3,n=7):
    F = norm(loc=0, scale=sigma_u).cdf
    sigma_y = np.sqrt((sigma_u**2)/(1-rho**2))
    x0=-m*sigma_y
    xn1=m*sigma_y
    s=(xn1-x0)/(n-1)
    x=np.zeros(n)
    x[0]=x0
    x[n-1]=xn1
    P=np.zeros((n,n))
    for i in xrange(n-1):
        x[i+1]=x[i]+s 
    for i in xrange(n):
        for j in xrange(n):
            if j==0:
                P[i,j]=F(x0-rho*x[i]+s/2)
            elif j==n-1:
                P[i,j]=1-F(xn1-rho*x[i]-s/2)
            else:
                P[i,j]=F(x[j]-rho*x[i]+s/2)-F(x[j]-rho*x[i]-s/2)
    return x,P
 
x,P = markovapprox(.9,1)
print "State space vector: ", x
print "Transition matrix: ", P
