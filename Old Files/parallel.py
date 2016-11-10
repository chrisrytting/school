from mpi4py import MPI
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

print "Process ", rank, "running out of ", size
import numpy as np

rand = np.zeros(1)

if rank==1:
    rand = nprandom.random_sample(1)
if rank == 0:
    print "process", rank, "has the number", rand[0], "before receiving."
