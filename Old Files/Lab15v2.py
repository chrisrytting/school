import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import timeit
import time
 
#Exercise 1
 
#The reason why there is such a large error because of how large the floating point is. If it's large enough then adding 1 is like adding 0.
#The best way to conteract this is by using double precision.
 
#Exercise 2
 
h=np.linspace(.000001,.0001,1000)
x=np.linspace(-5,5,1000)
y=abs((np.sin(1+h)-np.sin(1))/h-np.cos(1))
plt.plot(h,y)
plt.show()
#as h gets smaller, the error gets smaller as well
 
#Exercise 3
 
 
def lnseriesapproximation(x,n=3):
    tot=0
     
    for m in xrange(n+1):
        equa=((-1)**m*x**m)/(m+1)
        tot=tot+equa
 
    return tot
 
x1=np.linspace(-5,30,1000)
x2=np.linspace(1,30,1000)
x2new=10**(-x2)
y1=np.log(10**(-x1)+1)*10**(x1)
y2=lnseriesapproximation(x2new)
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.ylim(0,2)
plt.show()
 
#Exercise 4
 
#Squaring and adding the floats together will make them become very large.
#So therefore the numbers will be incorrect.
 
#Exercise 5
 
error=.1-(209715./2097152.)
print error*100.*1676.*60.*60.
