from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy as sp 
import scipy.optimize as opt
from numpy import random as rand
from numpy.matlib import repmat
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import tabulate as tab 
import scipy.stats as stats
from numpy import linalg as la 
from numpy.lib import scimath
import time
 
print "Problem 1"
 
def powermethod(A,tol=1e-10):
    m,n=np.shape(A)
    x0= np.random.rand(n)
    norm_x0 = la.norm(x0)
    x0 = x0/norm_x0
    error = 1000
    xk=x0
    i=0
    while error > tol:
        xk1=np.dot(A,xk)/la.norm(np.dot(A,xk))
        error = la.norm(xk-xk1)
        xk=xk1 
        i=i+1
    eigenvalue = np.inner(np.dot(A,xk1),xk1)/np.dot(xk1,xk1)
    eigenvector = xk1
    return eigenvalue, eigenvector
 
 
A=np.random.randint(2,10,(5,5))
print "A =  \n ", A
eigenvalue, eigenvector = powermethod(A)
print "Lambda = ", eigenvalue
print "Eigenvector = ", eigenvector

#2
print "Problem 2"
 
def QR(A,nIters,tol=1e-10):
    i=0
    H=sp.linalg.hessenberg(A)
    Ak=H
    for i in xrange(nIters):
        Qk,Rk = la.qr(Ak)
        Ak1=np.dot(Rk,Qk)
        Ak=Ak1
    S=Ak1 
    m,n=np.shape(Ak1)
    eigenvalues=[]
    i=0
    while i < m:
        if i == m-1:
            eigenvalues.append(S[i,i])
            i=i+1
        elif S[i+1,i] <= tol: 
            eigenvalues.append(S[i,i])
            i=i+1
        elif S[i+1,i] > tol:
            a=S[i,i]
            b=S[i+1,i]
            c=S[i,i+1]
            d=S[i+1,i+1]
            lambda1=((a+d)+scimath.sqrt(((a+d))**2-4*(a*d-b*c)))/2
            lambda2=((a+d)-scimath.sqrt(((a+d))**2-4*(a*d-b*c)))/2
            eigenvalues.append(lambda1)
            eigenvalues.append(lambda2)
            i=i+2
    return eigenvalues
 
A=np.array([[4,12,17,-2],[-5.5,-30.5,-45.5,9.5],[3,20,30,-6],[1.5,1.5,1.5,1.5]])

np.set_printoptions(suppress=True,precision=5)
print "A =", A 
print "Eigenvalues: ", QR(A,nIters=70,tol=1e-15)

#3
print "Problem 3"
 
def markov(A,x0,nIters):
    eigenvalues, eigenvectors = sp.linalg.eig(A)
    index = 0
    for i in xrange(len(eigenvalues)):
        if eigenvalues[i]==1:
            index = i 
    x=eigenvectors[:,index]
    x=x/np.sum(x)
    j = 0
    An = A
    for i in xrange(nIters-1):
        An=np.dot(A,An)
    return x,np.dot(An,x0)
 
x0=np.array([1,0,0])
A=np.array([[.5,.3,.4],[.2,.2,.3],[.3,.5,.3]])
print "Original matrix: ", A
print "Fixed point: ", markov(A,x0,2)[0]
print "State after 2 iterations: ", markov(A,x0,2)[1]
 
 
#4
print "Problem 4"
 
A=np.array([[.75,.5],[.25,.5]])
x0 = np.array([1,0])
fixed, state = markov(A,x0,3)
print "Third free throw likelihood: ", state[0]
print "Average free throw percentage: ", fixed[0]
