from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import math
import cmath
import scipy

def f(x):
    return np.cos(x)
def g(x):
    return (x**2)*np.sin(1/x)
def h(x):
    return ((np.sin(x)/x)-x)    
def i(x):
    return (x**2-1)
def j(x):
    return (x**3 - x)
def k(x):
    return (x**(1/3))

def fp(x):
    return -np.sin(x)
def gp(x):
    return 2*(x)*np.sin(1/x)-(x**2)*np.cos(1/x)*x**-2
def hp(x):
    return ((np.sin(x) - np.cos(x)*x)/(x**2))-1
def ip(x):
    return (2*x)
def jp(x):
    return (3*x**2 - 1)
def kp(x):
    return (1/3)*(x**-(2/3))

#def Newtonfirst(function, fprime=None, error, maxiterations, guess):
def Newton(function, fprime, error, maxiterations, guess):
    fx1 = function(guess)
    
    converge = False
    if fprime is None:
        functionprime = ((function(guess+.000001) - function(guess))/.000001)
    else:
        functionprime = fprime(guess)
    x2 = guess - (function(guess)/functionprime)
    difference = np.abs(guess - x2)
    xtoday = x2
    counter = 0
    while np.any(difference -error > 0):
        if fprime is None:
            functionprime = ((function(xtoday+.000001) - function(xtoday))/.000001)
        else:
            functionprime = fprime(xtoday)
        xtomorrow = xtoday - (function(xtoday)/functionprime)
        difference = np.abs(xtomorrow-xtoday)
        xtoday = xtomorrow
        counter += 1
        if counter>= maxiterations:
            print converge
            print 'counter ', counter
            return xtoday#, converge
    if counter <maxiterations:
        converge = True
        print 'counter ', counter
        print converge
        return xtoday#, converge

# trying to create a function that does the rest of what is below
'''
functions = [f,g,h,i,j]
functionsp = [fp,gp,hp,ip,jp]
for z in range(len(functions)):
    print 'function ', functions[i]
    print Newton(functions[i], None, 10**-9, 100, 1)
    print Newton(functions[i],functionsp[i], 10**-9, 100, 1)
'''


print "function f"
print Newton(f, None, 10**-9, 100, 1)
print Newton(f,fp, 10**-9, 100, 1)

print "function g"
print Newton(g, None, 10**-9, 100, 1)
print Newton(g,gp, 10**-9, 100, 1)

print "function h"
print Newton(h, None, 10**-9, 100, 1)
print Newton(h,hp, 10**-9, 100, 1)

print "function i"
print Newton(i, None, 10**-9, 100, 2)
print Newton(i,ip, 10**-9, 100, 2)

print "function j"
print Newton(j, None, 10**-9, 100, 2)
print Newton(j,jp, 10**-9, 100, 2)


print "function i between 2 and -2, increment by 1"
print Newton(i,None, 10**-9, 100, -2)
print Newton(i,None, 10**-9, 100, -1)
print Newton(i,None, 10**-9, 100, 1)
print Newton(i,None, 10**-9, 100, 2)


print "function j between 2 and -2, increment by 1"
print Newton(j,None, 10**-9, 100, -2)
print Newton(j,None, 10**-9, 100, -1)
print Newton(j,None, 10**-9, 100, 0)
print Newton(j,None, 10**-9, 100, 1)
print Newton(j,None, 10**-9, 100, 2)
'''
print "function k" 
print Newton(k, None, 10**-9, 100, 2)
print Newton(k, kp, 10**-9, 100, 2)
'''
#Doesn't work for x**1/3 because you cannot raise a negative to a fractional power.

#2
def plot(z, xmin, xmax, ymin, ymax, res):
    zprime = z.deriv(1)
    zroots = z.roots
    zroots = np.around(zroots, 3)
    x = np.linspace(xmin,xmax,res)
    y = np.linspace(ymin,ymax,res)
    plt.xlim([xmin,xmax])
    plt.ylim([ymin,ymax])
    X,Y = np.meshgrid(x,y)
    C1 = Newton(z,zprime,10**-5, 100, X+ 1j*Y)
    C = np.around(C1,3) 
    
    for i, root in enumerate(zroots):
        C[C == root] = i + 1


    plt.pcolormesh(X,Y,C)
    plt.show()


a = np.poly1d([1,-2,-2,2])
plot(a,-.5,0,-.25,.25,500)
 
b = np.poly1d([3,-2,-2,2])
plot(b,-1,1,-1,1,500)
 
c = np.poly1d([1,3,-2,-2,2])
plot(c,-1,1,-1,1,500)
 
d = np.poly1d([1,0,0,-1])
plot(d,-1,1,-1,1,500)


#3
x = np.linspace(-1.5,.5,500)
y = np.linspace(-1,1,500)
plt.xlim([-1.5,.5])
plt.ylim([-1,1])
X,Y = np.meshgrid(x,y)

C = X +1j*Y

x1 = C.copy()
x2 = np.zeros_like(C, dtype = int)

counter = 0
while counter < 30:
    x1 = x1**2 + C
    counter += 1
    x2[x1<10000000000] +=1

plt.pcolormesh(X,Y,x2)
plt.show()
