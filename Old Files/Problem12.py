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


def problem12():
    processes = 20
    d = np.array([int(sys.argv[1])])
    l = np.array([int(sys.argv[2])])
    simulations = np.array([int(sys.argv[3])])
    comm  = MPI.COMM_SELF.Spawn(sys.executable, args=['worker_pi2.py'], maxprocs=processes)
     
    comm.Bcast([d, MPI.INT], root=MPI.ROOT)
    comm.Bcast([l, MPI.INT], root=MPI.ROOT)
    comm.Bcast([simulations, MPI.INT], root=MPI.ROOT)
     
    pi = np.array([0.0])
    comm.Reduce(None, [pi, MPI.DOUBLE], op=MPI.SUM, root = MPI.ROOT)
    print pi/processes
     
    comm.Disconnect()

problem12()

