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

def problem11():
     
    import numpy.random as random
    processes = 4
    n = int(sys.argv[1])
    comm  = MPI.COMM_SELF.Spawn(sys.executable, args=['worker_pi.py'], maxprocs=processes)
    A = random.random((n,n))
    B = random.random((n,n))
     
    n = np.array([n])
    comm.Bcast([n, MPI.INT], root=MPI.ROOT)
    comm.Bcast([A, MPI.DOUBLE], root = MPI.ROOT)
    comm.Bcast([B, MPI.DOUBLE], root = MPI.ROOT)
    finalresult = np.zeros((n,n))
    comm.Gather(None, finalresult, root = MPI.ROOT)
    print finalresult
    print np.allclose(finalresult, np.dot(A, B))
    comm.Disconnect()

problem11()
