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

def extraproblem10():
     
     
    n = int(sys.argv[1]) 
    tol = .0001
    A= np.ones((n,n))
    A[:,0]=100
    A[:,-1]=100
    A[0]=0
    A[-1]=0
     
    comm.Bcast([A, MPI.DOUBLE], root=0)
    def LaPlaceTransform(matrix,errortol):
        matrixcopy=np.copy(matrix)
        difference=1
        while difference >=errortol:
            matrixcopy[1:-1,1:-1]=(matrix[:-2,1:-1]+matrix[2:,1:-1]+matrix[1:-1,2:]+matrix[1:-1,:-2])/4
            differencematrix=abs(matrixcopy-matrix)
            difference=np.max(differencematrix)
            matrix=np.copy(matrixcopy)
        return matrix
    newA=LaPlaceTransform(A, tol)
    comm.Reduce(newA, A)
    A=A/size
    if (rank==0):
        print A


extraproblem10()
