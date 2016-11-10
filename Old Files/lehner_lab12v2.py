# name this file solutions.py
"""Volume 2 Lab 12: Gaussian Quadrature
Lehner White
Math 321
Dec. 8 2015

Do not chage the names of the functions defined below.
You may name additional functions whatever you want to.
"""

import numpy as np
import scipy as sp
from scipy.integrate import quad
from matplotlib import pyplot as plt
import math
from math import sqrt
from scipy import linalg
from scipy import sparse
from scipy.stats import norm

def exercise1():
    f = lambda x : x**2
    g = lambda x : (9/4.)*x**2 + (15/2.)*x + (25/4.)
    G = lambda x : (9/8.)*x**3 + (45/8.)*x**2 + (75/8.)*x

    print G(1) - G(-1)
    print quad(f, 1, 4)[0]
    print "They are equal...."

#exercise1()

def gaussian(f, a, b):
    def g(x):
            return f(((b-a)/2.)*x+(b+a)/2.)
    return g

def shift_example():
    """Plot f(x) = x**2 on [1, 4] and the corresponding function
    ((b-a)/2)*g(x) on [-1, 1].
    """
    f = lambda x : x**2
    
    x = np.linspace(1,4,200)
    y = np.linspace(-1,1,200)
    z = gaussian(f,1,4)
    f = f(x)

    plt.subplot(1,2,1)
    plt.title('f(x)')
    plt.plot(x,f)

    plt.subplot(1,2,2)
    plt.title('z(x)')
    plt.plot(y,1.5*z(y))

    plt.show()

shift_example()

def estimate_integral(f, a, b, points, weights):
    """Estimate the value of the integral of the function 'f' over the
    domain [a, b], given the 'points' to use for sampling, and their
    corresponding 'weights'.

    Return the value of the integral.
    """
    g = gaussian(f, a, b)
    return np.sum((b-a)/2. * np.inner(weights,g(points)))    

def Jacobian(gamma, alpha, beta):
    g_n=len(gamma)
    a = np.zeros((g_n))
    b = np.zeros((g_n-1))

    for i in xrange(g_n):
            a[i] = (-1*beta[i]/alpha[i])
    
    for i in xrange(g_n-1):
            b[i] = (gamma[i+1]/(alpha[i]*alpha[i+1]))**(1/2.)
    
    jacobian = np.diag(a, 0) + np.diag(b,1) + np.diag(b,-1)
    return jacobian

def Legendre(n):
    alpha = np.zeros((n))
    beta = np.zeros((n))
    gamma = np.zeros((n))
    
    for i in xrange(n):
        alpha[i] = (2.*(i+1.)-1)/(i+1.)
        gamma[i] = ((i+1.)-1)/(i+1.)
    
    jacobian = Jacobian(gamma,alpha,beta)
    X, eig_vecs = np.linalg.eig(jacobian)
    w = 2*eig_vecs[0]**2
    return X, w

def gaussian_quadrature(f, a, b, n):
    """Using the functions from the previous problems, integrate the function
    'f' over the domain [a,b] using 'n' points in the quadrature.
    """
    pts , w = Legendre(n)
    integral = estimate_integral(f, a, b, pts, w)
    return integral


def normal_cdf(x):
    """Compute the CDF of the standard normal distribution at the point 'x'.
    That is, compute P(X <= x), where X is a normally distributed random
    variable.
    """
    pdf = lambda x : (1/((2.*np.pi)**(1/2.)))*np.exp((-x**2.)/2.)    
    return sp.integrate.quad(pdf, -5., x)

