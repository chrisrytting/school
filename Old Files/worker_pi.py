from mpi4py import MPI
import numpy as np
 
comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
 
n = np.array([0])
comm.Bcast([n, MPI.INT], root=0)
A = np.zeros((n,n))
B = np.zeros((n,n))
comm.Bcast([A, MPI.DOUBLE], root=0)
comm.Bcast([B, MPI.DOUBLE], root=0)
columns_to_do = n/size
result = np.dot(A[columns_to_do*rank:columns_to_do*(rank+1)], B)
comm.Gather([result, MPI.DOUBLE], None, root=0)
comm.Disconnect()

