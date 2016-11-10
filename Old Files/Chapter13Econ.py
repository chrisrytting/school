from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.optimize as opt
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import scipy.optimize as opt
import scipy
import re

print 'Problem 2'

def f(mu, sigma, N, k):
    leftnode = mu - k*sigma
    rightnode = mu + k*sigma
    x = np.linspace(leftnode, rightnode, N)
    w = np.zeros(len(x))
    w[0] = norm.cdf((x[0] + x[1])/2, mu, sigma)
    w[N-1] = 1 - norm.cdf((x[N-2] + x[N-1])/2, mu, sigma)
    for i in range(1,N-1):
        w[i] = norm.cdf((x[i] + x[i+1])/2, mu, sigma) - norm.cdf((x[i-1] + x[i])/2, mu, sigma)
    print x
    print w

f(0,1,11,3)

print 'Problem 3'

def g(mu, sigma, N, k):
    leftnode = mu - k*sigma
    rightnode = mu + k*sigma
    x = np.linspace(leftnode, rightnode, N)
    w = np.zeros(len(x))
    w[0] = norm.cdf((x[0] + x[1])/2, mu, sigma)
    w[N-1] = 1 - norm.cdf((x[N-2] + x[N-1])/2, mu, sigma)
    for i in range(1,N-1):
        w[i] = norm.cdf((x[i] + x[i+1])/2, mu, sigma) - norm.cdf((x[i-1] + x[i])/2, mu, sigma)
    for i in range(N):
        x[i] = math.exp(x[i])
    print x
    print w
    return sum(x*w)
print g(0,1,11,3)

print 'Problem 4'

print g(10.5, .8, 20, 3)
print np.exp(10.5 + (.8**2/2))
print "The values are very close to each other considering the magnitude of the guesses."

print 'Problem 5'

def system(guess, bounds):
    w1 = guess[0]
    w2 = guess[1]
    w3 = guess[2]
    x1 = guess[3]
    x2 = guess[4]
    x3 = guess[5]
    a = bounds[0]
    b = bounds[1]
    eq1 = w1 + w2 + w3 - (b-a)
    eq2 = w1*x1 + w2*x2 + w3*x3 - (b**2./2. - a**2./2.)
    eq3 = w1*x1**2.+ w2*x2**2.+ w3*x3**2.- (b**3./3. - a**3./3.)
    eq4 = w1*x1**3.+ w2*x2**3.+ w3*x3**3.- (b**4./4. - a**4./4.)
    eq5 = w1*x1**4.+ w2*x2**4.+ w3*x3**4.- (b**5./5. - a**5./5.)
    eq6 = w1*x1**5.+ w2*x2**5.+ w3*x3**5.- (b**6./6. - a**6./6.)
    return np.array([eq1,eq2,eq3,eq4,eq5,eq6])

g = lambda x: .1*x**4. - 1.5*x**3. + .53*x**2. + 2.*x+1.

bounds = np.array([-10.,10.])
guess = np.array([10.,1.,1.,.3,.4,.3])

results = opt.fsolve(system, guess, args = bounds, xtol=1e-50)
print results[0]*g(results[3])+results[1]*g(results[4])+results[2]*g(results[5])

print 'This is problem 6'

print scipy.integrate.quad(g,-10.,10.)

print 'They are almost identical'

print 'This is problem 7'
def mc_int(f, mins, maxs, N, numIters=100):
    numPoints = N
    cum_integral = 0
    for i in xrange(numIters):
        Ddim = len(maxs) #dimension of the domain
        points= np.random.rand(numPoints, Ddim)
        points = mins + (maxs - mins) * points
 
        f_value= np.apply_along_axis(f, 1, points) # numPoints by 1
 
        cum_integral += np.prod(maxs - mins) * sum(f_value) / numPoints
 
    avg_integral = cum_integral / numIters
    return avg_integral

f = lambda x: np.hypot(x[0], x[1]) <= 1
print mc_int(f, np.array([-1,-1]), np.array([1,1]), 500)

print 'This is problem 8'

def re(sequence, n, d):
    newsequence = sequence(n,d)
    return newsequence[n-1]
print 'This is problem 9'

def isPrime(n):
    # see http://www.noulakaz.net/weblog/2007/03/18/a-regular-expression-to-check-for-prime-numbers/
    return re.match(r'^1?$|^(11+?)\1+$', '1' * n) == None
 
 
N = 10 # number of primes wanted (from command-line)
M = 100              # upper-bound of search space
l = list()           # result list
 
while len(l) < N:
    l += filter(isPrime, range(M - 100, M)) # append prime element of [M - 100, M] to l
    M += 100                                # increment upper-bound
 
print l[:N] # print result list limited to N elements
def weyl(n,s):
    seq=np.zeros((s,n))
    primes=l[:s]
    for i in xrange(1,n+1):
        for j in xrange(1,s+1):
            seq[(j-1),(i-1)]=(i*primes[j-1]**.5-math.floor(i*primes[j-1]**.5))
    return seq


def mc_int(f, g, mins, maxs, N, numIters=100):
    numPoints = N
    cum_integral = 0
    for i in xrange(numIters):
        Ddim = len(maxs) #dimension of the domain
        points= (g(numPoints, Ddim)).T
        points = mins + (maxs - mins) * points
        f_value= np.apply_along_axis(f, 1, points) # numPoints by 1
        cum_integral += np.prod(maxs - mins) * sum(f_value) / numPoints
    avg_integral = cum_integral / numIters
    return avg_integral

def Haber(n,s):
    seq=np.zeros((s,n))
    primes=l[:s]
    for i in xrange(1,n+1):
        for j in xrange(1,s+1):
            seq[j-1,i-1]=(i*(i+1.)/2.*primes[j-1]**.5-math.floor(i*(i+1.)/2.*primes[j-1]**.5))
    return seq

def Niederreiter(n,s):
    seq=np.zeros((s,n))
    for i in xrange(1,n+1):
        for j in xrange(1,s+1): 
            seq[j-1,i-1]=(i)*2.**((j)/(j+1.)) - math.floor((i)*2.**((j)/(j+1.)))
    return seq

def Baker(n,s):
    seq=np.zeros((s,n))
    primes=l[:s]
    for i in xrange(n):
        for j in xrange(s):
            seq[j,i]=((i+1)*np.exp(j+1))-math.floor((i+1)*np.exp(j+1))
    return seq

f = lambda x: np.hypot(x[0], x[1]) <= 1
print mc_int(f, weyl, np.array([-1,-1]), np.array([1,1]), 500)
print mc_int(f, Haber, np.array([-1,-1]), np.array([1,1]), 500)
print mc_int(f, Niederreiter, np.array([-1,-1]), np.array([1,1]), 500)
print mc_int(f, Baker, np.array([-1,-1]), np.array([1,1]), 500)

def errors(f, g, h, mins, maxs, N):
    error = np.zeros(4) 
    for i in range(len(error)):
        error[i] = np.abs(f(g,h,mins,maxs,N[i]) - np.pi)
    return error
N = np.array([10,100,1000,10000])
Nex = np.log(N)**2/N


errorweyl = errors(mc_int, f, weyl, np.array([-1,-1]), np.array([1,1]), N)
errorhaber = errors(mc_int, f, Haber, np.array([-1,-1]), np.array([1,1]), N)
errorniederreiter = errors(mc_int, f, Niederreiter, np.array([-1,-1]), np.array([1,1]), N)
errorbaker = errors(mc_int, f, Baker, np.array([-1, -1]), np.array([1,1]), N)

plt.subplot(2,2,1)
plt.suptitle("QCM errors")
plt.plot(N,errorweyl)
plt.plot(N, Nex)
plt.title("Weyl")

plt.subplot(2,2,2)
plt.plot(N,errorhaber)
plt.plot(N, Nex)
plt.title("Haber")

plt.subplot(2,2,3)
plt.plot(N,errorniederreiter)
plt.plot(N, Nex)
plt.title("Niederreiter")

plt.subplot(2,2,4)
plt.plot(N,errorbaker)
plt.plot(N, Nex)
plt.title("Baker")

plt.show()








