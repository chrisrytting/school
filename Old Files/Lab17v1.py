from __future__ import division
import math
import scipy
from scipy import linalg
import numpy as np
from scipy import sparse
from math import sqrt
import matplotlib.pyplot as plt
from scipy.stats import norm
def f(x):
    return x**2

def g(x):
    return(9./4.)*x**2 + (15./2.)*x + (25./4.)

def G(x):
    return (9./8.)*x**3 + (45./8.)*x**2 + (75./8.)*x


print G(1) - G(-1) 


def func(f, a, b):
    def g(x):
        return f((((b-a)/2.)*x) + ((b+a)/2.))
    return g

gtest = func(f,1,4)

x1 = np.linspace(1,4,100)
x2 = np.linspace(-1,1,100)

plt.subplot(1,2,1)
plt.title('f(x)')
plt.plot(x1, f(x1))
plt.subplot(1,2,2)
plt.title('f(x)')
plt.plot(x2, (((4.-1.)/2.)*gtest(x2)))
plt.show()

#2

def func2(f, points, weights, intlim1, intlim2):
    g = lambda x: f((b-a)/ 2 * x + (a + b) / 2)
    return (b-a)/2 * np.inner(weights, g(points))

a, b = -np.pi, np.pi
f = np.sin
points = np.array([- sqrt(5 + 2 * sqrt(10. / 7)) / 3,
                   - sqrt(5 - 2 * sqrt(10. / 7)) / 3,
                   0,
                   sqrt(5 - 2 * sqrt(10. / 7)) / 3,
                   sqrt(5 + 2 * sqrt(10. / 7)) / 3])
weights = np.array([(322 - 13 * sqrt(70)) / 900,
                    (322 + 13 * sqrt(70)) / 900,
                    128. / 225,
                    (322 + 13 * sqrt(70)) / 900,
                    (322 - 13 * sqrt(70)) / 900])


integral = (b - a)/2 * np.inner(weights, g(points))

integraltest = func2(f, points, weights, a, b)
print integral
print integraltest

#3

def jacob(alpha, beta, gamma):
    a = -beta/alpha
    b = (gamma[1:]/(alpha[:-1]*alpha[1:]))**0.5
    jacobian = np.diag(b,-1) + np.diag(a,0) + np.diag(b,1) 
    return jacobian

#4

def Legendre(n):
    alpha = np.zeros(n) 
    beta = np.zeros(n)
    gamma = np.zeros(n)
    for i in range( 1,n+1):
        alpha[i-1] = (2*i - 1)/i
        beta[i-1] = 0
        gamma[i-1] = (i - 1)/i
    print alpha
    print beta
    print gamma
    return jacob(alpha, beta, gamma)

x, evecs =  linalg.eig(Legendre(5))

w = 2*evecs[0]**2
print 'x vector = ', x
print 'w vector = ', w

#5

def n(t):
    return ((1/np.sqrt(2*math.pi))*math.e**(-t**2 / 2))

N = norm()
print N.cdf(1)

def func3(f,a,b):
    return scipy.integrate.quad(f,a,b)

print func3(n, -5, 1)






