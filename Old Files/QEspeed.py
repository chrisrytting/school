import matplotlib.pyplot as plt
import numpy as np
from numba import jit
import time
import cymodule
from scipy import sparse
 
 
 
 
p,q=.1,.2
 
def computeseries(n):
    x=np.empty(n,dtype=int)
    x[0]=1
    U=np.random.uniform(0,1,size=n)
    for t in xrange (1,n):
        current_x=x[t-1]
        if current_x==0:
            x[t]=U[t]<p
        else:
            x[t]=U[t]>q
    return x
 
n=100000
start=time.time()
x=computeseries(n)
Timer3=time.time()-start
print(np.mean(x==0))
print "Pure Python time:", Timer3
 
start=time.time()
compute_series_numba=jit(computeseries)
Timer4=time.time()-start
print "Numba Time:",Timer4
 
x2=compute_series_numba(n)
print(np.mean(x2==0))
 
 
 
start=time.time()
compute_cython=cymodule.computeseriesCY(n)
Timer5=time.time()-start
 
print "Cython Time:",Timer5
 
#Numba is fastest option
