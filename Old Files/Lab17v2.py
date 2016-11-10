from __future__ import division
import scipy as sp
import numpy as np

#1

def simpson(f, a, b):
    return ((b-a)/6.0)*sum(sp.array([1,4,1])*f(sp.array([a,(a+b)/2.0,b])))

def simpson38(f, a, b):
    return ((b-a)/8)*(f(a) + 3*f(a + (b-a)/3.) + 3*f((2*(a + (b-a))/3.)+ f(((a + (b-a))))))

def boole(f,a,b):
    return ((b-a)/90)*(7*f(a) + 32*f(a + (b-a)/4) + 12*f(a + (b-a)/2) + 32*f(a + 3*(b-a)/4) + 7*f(b))

#2

def simpsoncomp(f,a,b,n):
    intlen = (b-a)/n 
    x = np.zeros(n)
    integral = 0
    for i in range(n):
        x[i] = a + i*intlen 
    for j in range(n-1): 
        integral += simpson(f,x[j],x[j+1])
    return integral

def simpson38comp(f,a,b,n):
    intlen = (b-a)/n 
    x = np.zeros(n)
    integral = 0
    for i in range(n):
        x[i] = a + i*intlen 
    for j in range(n-1): 
        integral += simpson38(f,x[j],x[j+1])
    return integral

def boolecomp(f,a,b,n):
    intlen = (b-a)/n 
    x = np.zeros(n)
    integral = 0
    for i in range(n):
        x[i] = a + i*intlen 
    for j in range(n-1): 
        integral += boole(f,x[j],x[j+1])
    return integral

f = lambda x: x**(1/3)

print simpsoncomp(f,0,1,1000)
print simpson38comp(f,0,1,1000)
print boolecomp(f,0,1,1000)

#3

def adaptSimpson(func,a,b,maxerr):
    error=1000000
    vector=np.array([a,b])
    ans=1
    while ans==1:
        x=a
        tot=0
        n=len(vector)-1
        ans=0
        currenterr=maxerr/n
        for i in xrange(n):
            x=vector[i]
            y=vector[i+1]
            c=(y+x)/2.
            err = abs(simpson(func,x,c)+simpson(func,c,y)-simpson(func,x,y))
            if err >= currenterr:
                vector=np.insert(vector,i+1,c)
                ans=1
            else:
                tot=tot+simpson(func,x,y)
    return tot
 
print adaptSimpson(f,0.,1.,.001)


















