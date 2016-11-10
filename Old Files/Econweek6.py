from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp 
import scipy.optimize as opt
import pandas as pd
from numpy.matlib import repmat
import sys
import math
import csv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import tabulate as tab 
import scipy.stats as stats
 
data=np.loadtxt("clms.txt")
mean=np.mean(data)
variance=np.var(data)
median=np.median(data)
minimum=np.min(data)
maximum=np.max(data)
stdev=variance**(1/2)
N=len(data)

print "Mean: ", mean
print "Median: ", median
print "Max: ", maximum
print "Min: ", minimum
print "Standard Deviation: ", stdev

data2=data[data<800]
data1=data

plt.hist(data1, 1000, weights=np.zeros_like(data1)+1/N)

plt.xlabel('Health Expenditures')
plt.ylabel('Probability')
plt.title('Histogram of Health Expenditures')
plt.xlim([0,9000])
plt.grid(True)
plt.show()

plt.hist(data2, 100, weights=np.zeros_like(data2)+1/N)

plt.xlabel('Health Expenditures')
plt.ylabel('Probability')
plt.title('Histogram of Health Expenditures')
plt.xlim([0,800])
plt.grid(True)
plt.show()

#Problem 14.2

def gamma_dist(x,alpha,beta):
    return 1/(beta**alpha*math.gamma(alpha))*x**(alpha-1)*np.exp(-x/beta)

def gen_gamma_dist(x,alpha,beta,m):
    return (m/((beta**alpha)*math.gamma(alpha/m)))*(x**(alpha-1))*np.exp(-(x/beta)**m)

def gamma_mle(guess,data):
    alpha=guess[0]
    beta=guess[1]
    mle=np.log(gamma_dist(data,alpha,beta))
    loglikelihood=np.sum(mle)
    return -loglikelihood

def gen_gamma_mle(guess,data):
    alpha = guess[0]
    beta = guess[1]
    m = guess[2]
    mle = np.log(gen_gamma_dist(data,alpha,beta,m))
    loglikelihood=np.sum(mle)
    return -loglikelihood
n,bins,patches = plt.hist(data2, 100, weights=np.zeros_like(data2)+1/N)
beta0 = variance/mean
alpha0 = mean/beta0
guess = np.array([alpha0,beta0])
bnds = tuple([(.0000001,.9999999),(0,None)])
solution=opt.minimize(gamma_mle,guess,method = "L-BFGS-B", bounds=bnds, args=data1, tol=1e-25)
result = solution.x
alpha,beta = result[0],result[1]
loglikelihood1=-gamma_mle(result,data1)
print "Log likelihood: ", loglikelihood1
print "Alpha value: ", alpha
print "Beta value: ", beta
print "Graph of fitted gamma pdf over actual histogram"
 
x_values1 = np.linspace(0,800,1000)
x_values1 = np.reshape(x_values1,(1000,1))
fitted_gamma = np.sum(n)*8*np.array(gamma_dist(x_values1,alpha,beta))
plt.plot(x_values1,fitted_gamma)
n,bins,patches = plt.hist(data2, 100, weights=np.zeros_like(data2)+1/N)
plt.xlabel('Health Expenditures ')
plt.ylabel('Probability')
plt.title('Histogram of Health Expenditures Overlayed With Gamma Fit')
plt.xlim([0,800])
plt.ylim([0,.07])
plt.grid(True)
plt.show()
 
#Problem 14.3

beta0 = beta
alpha0 = alpha
m0 = 1.0
guess = np.array([alpha0,beta0,m0])
bnds = tuple([(.000001,None),(.0000001,None),(.0000001,None)])
solution=opt.minimize(gen_gamma_mle,guess,method = "Nelder-Mead", \
bounds=bnds,args=data1, tol=1e-25)
print solution
result = solution.x
alpha,beta,m = result[0],result[1],result[2]
loglikelihood2=-gen_gamma_mle(result,data1)
print "Log likelihood: ", loglikelihood2
print "Alpha value: ", alpha
print "Beta value: ", beta
print "M value: ", m
print "Graph of fitted generalized gamma pdf over actual histogram"
 
x_values2 = np.linspace(0,800,1000)
fitted_gen_gamma = np.sum(n)*8*np.array(gen_gamma_dist(x_values2,alpha,beta,m))
print "Fitted gen gamma", fitted_gen_gamma
print len(x_values2)
print len(fitted_gen_gamma)
plt.plot(x_values2,fitted_gen_gamma)
n,bins,patches = plt.hist(data2, 100, weights=np.zeros_like(data2)+1/N)
plt.xlabel('Health Expenditures ')
plt.ylabel('Probability')
plt.title('Health Expenditures with \
GeneralizedGamma Fit')
plt.xlim([0,800])
plt.ylim([0,.07])
plt.grid(True)
plt.show()

#Problem 4

print 4

def beta_func(a,b):
    '''uses log-gamma function or inbuilt math.lgamma() to compute values of beta function'''
    beta = math.exp(math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b))
    return beta
 
def gb2(x,a,b,p,q):
    return (a*x**(a*p-1))/((b**(a*p))*(beta_func(p,q))*(1+(x/b)**a)**(p+q))
 
def gb2_mle(guess,data):
    a = guess[0]
    b = guess[1]
    p = guess[2]
    q = guess[3]
    mle = np.log(gb2(data,a,b,p,q))
    loglikelihood=np.sum(mle)
    return -loglikelihood
 
a0 = m
b0 = 100**(1/m)
p0 = alpha/m
q0 = 100
guess = np.array([a0,b0,p0,q0])
bnds = tuple([(.0000001,None),(0.0000001,None),(.0001,None),(0.000000001,None)])
solution=opt.minimize(gb2_mle,guess,method = "L-BFGS-B", \
bounds=bnds,args=data1, tol=1e-20)
print solution
result = solution.x
a,b,p,q = result[0],result[1],result[2],result[3]
loglikelihood3=-gb2_mle(result,data1)
print "Log likelihood: ", loglikelihood3
print "a value: ", a
print "b value: ", b
print "p value: ", p
print "q value: ", q
print "Graph of fitted generalized gamma pdf over actual histogram"
 
x_values3 = np.linspace(0,800,1000)
fitted_gb2 = np.sum(n)*8*np.array(gb2(x_values3,a,b,p,q))
plt.plot(x_values3,fitted_gb2)
n,bins,patches = plt.hist(data2, 100, weights=np.zeros_like(data2)+1/N)
 
print np.sum(fitted_gb2)
plt.xlabel('Health Expenditures')
plt.ylabel('Probability')
plt.title('Health Expenditures Histogram Overlayed With GB2 Fit')
plt.xlim([0,800])
plt.grid(True)
plt.show()

#Problem 5
print "problem 5"
plt.plot(x_values1,fitted_gamma,label="Gamma Distribution")
plt.plot(x_values2,fitted_gen_gamma,label="Generalized Gamma Distribution")
plt.plot(x_values3,fitted_gb2,label="GB2 Distribution")
plt.legend(loc="upper right")
n,bins,patches = plt.hist(data2, 100, weights=np.zeros_like(data2)+1/N)
plt.xlabel('Health Expenditures')
plt.ylabel('Probability')
plt.title('Health Expenditures Histogram Overlayed With All Distribution Fits')
plt.xlim([0,800])
plt.ylim([0,.07])
plt.grid(True)
plt.show()
 
print "The best way to compare goodness of fit is the"
print "likelihood ratio test, although the eyeball test pretty"
print "well demonstrates that the GB2 is the best fit"
print "This is confirmed by the likelihood ratio test: "
print "Gamma likelihood: ", loglikelihood1
print "Generalized Gamma likelihood: ", loglikelihood2
print "GB2 likelihood: ", loglikelihood3
