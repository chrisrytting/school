import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys

#Problem 38
def Problem1(a,b):
    mean1, var1 = stats.beta.stats(a,b, moments='mv')
    x = np.linspace(stats.beta.ppf(.01,a,b), stats.beta.ppf(.99, a, b), 100)
    plt.plot(x,stats.beta.pdf(x,a,b))
    print "Mean: ", mean1,"\nVariance: ", var1
    vals = np.array([1,2,4,8,16,32])
    for i in vals:
        plt.plot(x,mlab.normpdf(x,mean1,np.sqrt(var1/i)))
        xbar = np.average(stats.beta.rvs(a,b,size=(i,1000)),axis = 0)
        plt.hist(xbar,normed=True)

Problem1(1,4)
plt.show()
Problem1(1,1)
plt.show()
