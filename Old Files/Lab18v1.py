from __future__ import division
import scipy.stats as stats
from matplotlib import pyplot as plt
import numpy as np

def mc_int(f, mins, maxs, numPoints=500, numIters=100):
    integral = 0
    for i in xrange(numIters):
        domdim = len(maxs)
        points = np.random.rand(numPoints, domdim)
        points = mins +(maxs - mins)*points
        fval = np.apply_along_axis(f, 1, points)
        integral += np.prod(maxs-mins)*sum(fval)/numPoints
    integralmean = integral / numIters
    return integralmean

f = lambda x: np.hypot(x[0], x[1]) <= 1
print mc_int(f, np.array([-1,-1]), np.array([1,1]))

#1

mins = np.array([-.5, 0 , 0 , 0])
maxs = np.array([.75, 1 , 0.5 , 1])
avs = np.identity(4)
p, i = stats.mvn.mvnun(mins, maxs, avs, covs)
integral = p

#2

def f2(x_vec):
    N= len(x_vec)
    return 1/np.sqrt( (2*np.pi)**N ) * np.exp(- x_vec.dot(x_vec) / 2.)
 
N= [10,100,1000,10000]
error=[]
 
for i in N:
    estimate = mc_int(f2, mins, maxs, numPoints=i)
    error.append(abs(estimate-value))
    print 'With '+ str(i) + ' sample points, the integral is ' +  str(estimate)
 
plt.plot(N, error)
plt.show()
