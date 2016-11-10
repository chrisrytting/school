import sys
import numpy.random as random
import numpy as np
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import random as random

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


def extraproblem15():
    n = int(sys.argv[1]) 
    A=np.random.rand((n**2))
    A=np.reshape(A,(n,n))
    print A
    originalmatrix=np.copy(A)
    #print "Matrix A",A
     
     
    def invfunc(a11,a22,a21,a12, a22inv,a11inv):
        A11=np.linalg.inv(a11-np.dot(np.dot(a12,a22inv),a21))#
        A12=-np.dot(np.dot(a11inv,a12),(np.linalg.inv(a22-np.dot(np.dot(a21,a11inv),a12))))#
        A21=-np.dot(np.dot(a22inv,a21),np.linalg.inv(a11-np.dot(np.dot(a12,a22inv),a21)))#
        A22=np.linalg.inv(a22-np.dot(np.dot(a21,a11inv),a12))#
        A1=np.concatenate((A11,A12), axis=1)
        A2=np.concatenate((A21,A22), axis=1)
        A=np.concatenate((A1,A2),axis=0)
        #print "A.shape:", A.shape
        return A
     
    i=0
    #This is if even
    if n%2==0:
        print "Even"
        A22=A[n-2:,n-2:]
        A11=A[n-4:n-2,n-4:n-2]
        A12=A[n-4:n-2,n-2:]
        A21=A[n-2:,n-4:n-2]
        A11inv=np.linalg.inv(A11)
        A22inv=np.linalg.inv(A22)
        Ainverse=invfunc(A11,A22,A21,A12,A22inv,A11inv)
        i+=4
    #This is if odd
    else:
        print "Odd"
        A22=A[n-1:,n-1:]
        A11=A[n-3:n-1,n-3:n-1]
        A12=A[n-3:n-1,n-1:]
        A21=A[n-1:,n-3:n-1]
        A11inv=np.linalg.inv(A11)
        A22inv=np.linalg.inv(A22)
        Ainverse=invfunc(A11,A22,A21,A12,A22inv,A11inv)
        i+=3
     
    #Loop
    while i<n:
        A11=A[n-i-2:n-i,n-i-2:n-i]
        A12=A[n-i-2:n-i,n-i:]
        A21=A[n-i:,n-i-2:n-i]
        A22=A[n-i:,n-i:]
        A11inv=np.linalg.inv(A11)
        Ainverse=invfunc(A11,A22,A21,A12,Ainverse,A11inv)
        i+=2
     
    print "A inverse: ", Ainverse

extraproblem15()
