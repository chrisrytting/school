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

if rank == 0:
    print "REMEMBER TO TEST THESE PROBLEMS ONE AT A TIME, AS SOME TAKE DIFFERENT ARGUMENTS. TO TEST PROBLEMS, SIMPLY CALL THE FUNCTIONS problem1(), problem2(), ... , etc."

comm.Barrier()

def problem1():
    if rank == 0:
        print "!!! PROBLEM 1 !!!" 
    comm.Barrier()
    if rank%2 == 0:
        print "Hello from process ", rank, " of ", size
    if rank%2 == 1:
        print "Goodbye from process ", rank, " of ", size

#problem1()

#2
def problem2():
    if rank == 0:
        print "!!! PROBLEM 2 !!!"
    comm.Barrier()
    if rank == 0:
        if size!=5:
            print "Error: This program must run with five processes."
            comm.Abort()
        else: 
            print "Success!"

#problem2()

#3
def problem3():
    if rank == 0:
        print "!!! PROBLEM 3 !!!"
    comm.Barrier()
    rand = np.zeros(1)
    rand = np.random.random_sample(1)
    randrec = np.zeros(1)
    print "Process ", rank, "drew random number ", rand[0]

    if rank == size -1:
        comm.Send(rand[0], dest = 0)
    else:
        comm.Send(rand[0], dest = rank + 1)

    comm.Barrier()

    if rank == 0:
        comm.Recv(randrec, source = size - 1)
    else: 
        comm.Recv(randrec, source = rank - 1)

    comm.Barrier()

    print "Process ", rank, "received random number ", randrec[0]


#problem3()

#4
def problem4():
    if rank == 0:
        print "!!! PROBLEM 4 !!!"
    comm.Barrier()
    a=float(sys.argv[1])
    b=float(sys.argv[2])
    n=int(sys.argv[3])
    h=(b-a)/n
    local_n=n/size
    remainder=n%size
    if rank<remainder:
            local_n+=1
    if rank<remainder:
            local_a=a+rank*local_n*h
    else: local_a=(remainder+local_n*rank)*h

    local_b=local_a+local_n*h

    integral=np.zeros(1)
    recv_buffer=np.zeros(1)

    def f(x):
            return x*x

    def integrateRange(a,b,n):
            integral=-(f(a)+f(b))/2.0
            for x in np.linspace(a,b,n+1):
                    integral=integral+f(x)
            integral=integral*(b-a)/n
            return integral

    integral[0]=integrateRange(local_a,local_b,local_n)

    if rank==0:
            total=integral[0]
            for i in range(1,size):
                    comm.Recv(recv_buffer, ANY_SOURCE)
                    total+=recv_buffer[0]
    else:
            comm.Send(integral)

    if comm.rank==0:
            print "With n ={} trapezoids, our estimate of the integral from\
            {} to {} is {}".format(n, a, b, total)

#problem4()



def problem5():
    if rank == 0:
        print "!!! PROBLEM 5 !!!"
    N = 500
    n = int(sys.argv[1])
    local_n = n/size
    remainder = n%size
    if rank <remainder:
        local_n += 1

    local_int = np.zeros(1)
    recv_int = np.zeros(1)

    def mc_int(f, mins, maxs, N, numIters):
        numPoints = N
        cum_integral = 0
        numItersf = np.float(numIters)
        for i in xrange(numIters):
            Ddim = len(maxs) #dimension of the domain
            points= np.random.rand(numPoints, Ddim)
            points = mins + (maxs - mins) * points
     
            f_value= np.apply_along_axis(f, 1, points) # numPoints by 1

            numPointsf = np.float(numPoints)
     
            cum_integral += np.prod(maxs - mins) * sum(f_value) / numPointsf
     
        avg_integral = cum_integral / numItersf
        return avg_integral

    f = lambda x: np.hypot(x[0], x[1]) <= 1
    local_int[0] = mc_int(f, np.array([-1,-1]), np.array([1,1]),N, local_n)

    comm.Reduce(local_int, recv_int, op = MPI.SUM, root = 0)
    sizef = np.float(size)
    recv_int = recv_int / sizef

    if comm.rank == 0:
            print "With n = {} iterations, our estimate of the integral is{}".format(n, recv_int)


#problem5()


def problem6():
    if rank == 0:
        print "!!! PROBLEM 6 !!!"
    #length of vectors
    n = int(sys.argv[1]) 
    


    #example vectors
    x = np.linspace(0,100,n) if comm.rank==0 else None
    y = np.linspace(20,300,n) if comm.rank==0 else None

    #initialize results as numpy arrays
    dot = np.array([0.])
    local_n = np.array([0])

    local_n = n / size
    remainder = n % size
    if rank < remainder:
        local_n += 1
    vec = np.ones(size)*(n/size)
    vec[:remainder] += 1
    dis = np.cumsum(vec) - vec

    # test for conformability
    if rank == 0:
        if (n!= y.size):
            print "Vector-length mismatch"
            comm.Abort()
        


    #local result initialized as np arrays
    local_x = np.zeros(local_n)
    local_y = np.zeros(local_n)

    #divide up the vectors
    comm.Scatterv([x, tuple(vec), tuple(dis), MPI.DOUBLE], local_x, root = 0)
    comm.Scatterv([y, tuple(vec), tuple(dis), MPI.DOUBLE], local_y, root = 0)

    # local computation of dot product
    local_dot = np.array([np.dot(local_x, local_y)])

    #sum the results of each and send back to the master
    comm.Reduce(local_dot, dot, op=MPI.SUM)

    if (rank==0):
        print "the dot product is: ", dot[0], "computed in parallel"
        print "and ",np.dot(x,y), "computed in serial" 

#problem6()

def problem7():
    if rank == 0:
        print "!!! PROBLEM 7 !!!"
    #example vector
    n = int(sys.argv[1]) 
    v = np.linspace(-2,-12,n) if comm.rank==0 else None
    print "v ", v
    #print "n ", n
    #How many elements are devoted to each process.
    val = np.array([0.])
    local_n = np.zeros(1)
    local_n = n / size
    remainder = n % size
    if rank < remainder:
        local_n += 1
    #print "process ", rank, " has local_n = ", local_n
    vec = np.ones(size)*(n/size)
    vec[:remainder] += 1
    #print "process ", rank, " has vec ", vec
    dis = np.cumsum(vec) - vec
    
    #print "process ", rank, " has dis ", dis

    # test for conformability
    if rank == 0:
        if (n!= v.size):
            print "Vector-length mismatch"
            comm.Abort()
        

    local_val = np.zeros(local_n)
    comm.Scatterv([v, tuple(vec), tuple(dis), MPI.DOUBLE], local_val, root = 0)
    local_val = np.array([np.max(np.abs(local_val))])
    print "process ", rank, " has local_val ", local_val
    comm.Reduce(local_val, val, op = MPI.MAX)
    print "process ", rank, " has val ", val
    if rank == 0:
        print "The supremum norm of this #problem is ", val

#problem7()

def problem8():
    if rank == 0:
        print "!!! PROBLEM 8 !!!"
    #example vector
    n = int(sys.argv[1]) 
    p = int(sys.argv[2])
    v = np.linspace(1,3,n) if comm.rank==0 else None
    print "v ", v
    #print "n ", n
    #How many elements are devoted to each process.
    val = np.array([0.])
    local_n = np.zeros(1)
    local_n = n / size
    remainder = n % size
    if rank < remainder:
        local_n += 1
    #print "process ", rank, " has local_n = ", local_n
    vec = np.ones(size)*(n/size)
    vec[:remainder] += 1
    #print "process ", rank, " has vec ", vec
    dis = np.cumsum(vec) - vec
    
    #print "process ", rank, " has dis ", dis

    # test for conformability
    if rank == 0:
        if (n!= v.size):
            print "Vector-length mismatch"
            comm.Abort()
        

    local_val = np.zeros(local_n)
    comm.Scatterv([v, tuple(vec), tuple(dis), MPI.DOUBLE], local_val, root = 0)

    #calculates absolute values and raises them to the p

    local_val = np.abs(local_val)
    local_val = local_val**p

    #sums up local absolute values

    local_val = np.sum(local_val)
    print "process ", rank, " has local_val ", local_val

    #sums up all absolute values

    comm.Reduce(local_val, val, op = MPI.SUM)

    #raises them to the p TDODODODO
    val = val**(1./p)

    if rank == 0:
        print "process ", rank, " has val ", val
    if rank == 0:
        print "The p norm of this #problem is ", val


#problem8()



    


    

'''
This is potential code for passing in a matrix A and a vector v.
#A matrix
A = sys.argv[2]
#v vector
v = sys.argv[3] 
'''

def problem9():
    if rank == 0:
        print "!!! PROBLEM 9 !!!"
    #length of v
    n = int(sys.argv[1]) 
    v = np.ones(n)*2
    A = np.ones((n,n))*2
    finalmat = np.zeros((n,1))
    local_n = n / size
    local_mat = np.zeros((local_n, n))
    comm.Scatter(A, local_mat)
    local_mat = (np.dot(local_mat, v))
    comm.Gather(local_mat, finalmat, root = 0)

    if rank == 0:
        print "Process ", rank, " has ", finalmat 
    

#problem9()

def problem10():
    start = 0
    if rank == 0:
        start = MPI.Wtime()
     
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    n = int(sys.argv[3])
     
    def f(x):
        return np.exp(np.sin(x)*np.cos(x))*2.0**x
         
    def integrateRange(a, b, n):
        integral = -(f(a) + f(b))/2.0
        for x in np.linspace(a, b, n+1):
            integral = integral + f(x)
        integral = integral *(b-a)/n
        return integral
         
    h = (b-a)/n
     
    normalSteps = n/size
    remainder = n % size
    if rank < remainder:
        local_n = n/size + 1
    else:
        local_n = n/size
     
    local_a = a + rank*local_n*h
    local_b = local_a + local_n*h
     
    integral = np.zeros(1)
    recv_buffer = np.zeros(1)
     
    integral[0] = integrateRange(local_a, local_b, local_n)
     
    if rank==0:
        total = integral[0]
        for i in range(1, size):
            comm.Recv(recv_buffer, ANY_SOURCE)
            total += recv_buffer[0]
    else:
        comm.Send(integral)
         
         
    if comm.rank == 0:
        print "With n =", n, "trapezoids, our estimate of the integral from", a, "to", b, "is", total
        print "Elapsed time for", size, "processes:", MPI.Wtime()-start
        

#problem10()

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

#problem11()


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

#problem12()

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

#problem13()
 
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


#extraproblem10()

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

#extraproblem15()









