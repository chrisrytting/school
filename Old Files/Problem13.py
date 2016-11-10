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


def problem13():
    n = int(sys.argv[1])
    maximumiterations = int(sys.argv[2])
     
    matrix1 = np.zeros((n,n), dtype='i')
     
    if rank == 0:
        matrix1 = np.array([random.getrandbits(1) for i in xrange(n) for j in xrange(n)], dtype='i').reshape((n,n))
        print "Initial matrix:\n", matrix1
         
    comm.Bcast(matrix1, root = 0)
     
    iterations = 0
     
    counts1 = [n/size + 2 for i in xrange(size)]
    counts2 = [n/size for i in xrange(size)]
    counts1[0] -= 1
    counts1[-1] -= 1
    remainder = n % size
    sums = [1 if i < remainder else 0 for i in xrange(size)]
    counts1 = np.add(counts1, sums)
    differences = [-1 for i in xrange(size)]
    differences[0] = 0
    counts2 = np.add(counts2, sums)
     
    displacements = [np.sum(counts2[:i]) for i in xrange(size)]
    displacements = np.add(displacements, differences)
     
    def updateGrid(A):
        summationmatrix = A[:-2, :-2] + A[:-2, 1:-1] + A[:-2, 2:] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, :-2] + A[2:, 1:-1] + A[2:, 2:]
        summationmatrix = summationmatrix.flatten()
        A = A[1:-1, 1:-1]
        A2 = (summationmatrix == 3).astype(np.int32).reshape((counts1[rank],n)) + ((summationmatrix == 2).astype(np.int32).reshape((counts1[rank],n)) * A)
     
        A = A2.astype(np.int32)
     
        return A 
         
    def buffer_with_zeros(B):
        B = B.reshape((counts1[rank], n))
         
        A = np.zeros((B.shape[0]+2, B.shape[1]+2), dtype='i')
        A[1:-1, 1:-1] = B
        return A
         
    localmatrix1 = np.zeros(counts1[rank]*n, dtype='i')
     
    while iterations <= maximumiterations:
        iterations += 1
             
        comm.Scatterv([matrix1, tuple(counts1*n), tuple(displacements*n), MPI.INT], localmatrix1, root=0)
        new_matrix1 = updateGrid(buffer_with_zeros(localmatrix1))
        if rank == 0:
            new_matrix1 = new_matrix1[:-1]
        elif rank != size-1:
            new_matrix1 = new_matrix1[1:-1]
        else:
            new_matrix1 = new_matrix1[1:]
             
             
        comm.Gatherv(new_matrix1, [matrix1, tuple(counts1*n), tuple(displacements*n), MPI.INT], root=0)
        if rank == 0:
            print "After ", iterations, " iterations, we have: \n",matrix1

problem13()
