from mpi4py import MPI
import numpy as np
import numpy.random as random
 
 
comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
 
d = np.array([0])
l = np.array([0])
sims = np.array([0])
comm.Bcast([d, MPI.INT], root=0)
comm.Bcast([l, MPI.INT], root=0)
comm.Bcast([sims, MPI.INT], root=0)
 
local_sims = sims/size
rem = sims % size
if rank < rem:
    local_sims += 1
     
counter = 0
hits = 0
while counter < local_sims:
    counter += 1
    land_point = random.uniform(low=0.0, high=d/2.)
    angle = random.uniform(low=0.0, high=np.pi/2)
    upper_y = land_point + (l/2.0)*np.sin(angle)
    lower_y = land_point - (l/2.0)*np.sin(angle)
    if upper_y > d/2.:
        hits += 1
         
pi = ((2.*d)/l) * (local_sims/(hits*1.0))
 
 
comm.Reduce([pi, MPI.DOUBLE], None, op=MPI.SUM, root=0)
comm.Disconnect()
