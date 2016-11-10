import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sl

'''Functions for use in problem 1.'''
# Run through a single for loop.
def func1(n):
    n = 500*n
    sum(xrange(n))

# Run through a double for loop.
def func2(n):
    n = 3*n
    t = 0
    for i in xrange(n):
        for j in xrange(i):
            t += j

# Square a matrix.
def func3(n):
    n = int(1.2*n)
    A = np.random.rand(n, n)
    np.power(A, 2)

# Invert a matrix.
from scipy import linalg as la
def func4(n):
    A = np.random.rand(n, n)
    la.inv(A)

# Find the determinant of a matrix.
from scipy import linalg as la
def func5(n):
    n = int(1.25*n)
    A = np.random.rand(n, n)
    la.det(A)


def Problem1():
    """Create a plot comparing the times of func1, func2, func3, func4, 
    and func5. Time each function 4 times and take the average of each.
    """
    print 'NOTE, WE WERE TOLD THAT WE COULD HARD-CODE IN THE AVERAGES OF THE TIMES, THAT IS WHY MY ANSWERS ARE IN THIS FORMAT'
    x = [100, 200, 400, 800]

    fun1 = [439, 890, 1800, 3700]

    fun2 = [1970, 7900, 31100, 125000]

    fun3 = [521, 2080, 8310, 36000]

    fun4 = [632, 1910, 7250, 38800]

    fun5 = [530, 1700, 6130, 33100]

    ax1 = plt.subplot(111)
    ax1.plot( x, fun1, c='r')
    ax1.plot( x, fun2, c='b')
    ax1.plot( x, fun3, c='m')
    ax1.plot( x, fun4, c='y')
    ax1.plot( x, fun5, c='k')
    plt.xlabel('N')
    plt.ylabel('Time in microseconds')
    plt.legend(["Function 1", "Function 2", "Function 3", "Function 4", "Function 5"], loc = 'upper left')
    plt.show()

def Problem2(n):
    maindiag = np.linspace(2,2,n)
    otherdiag = np.linspace(-1,-1,n-1)
    otherdiag1 = np.linspace(-1,-1,n)
    tridiag = sparse.spdiags(maindiag, 0, n, n, format = 'csr') + sparse.spdiags(otherdiag, -1, n, n, format = 'csr') + sparse.spdiags(otherdiag1, 1, n, n, format = 'csr')
    tridiag = tridiag.todense()
    return tridiag

def Problem3(n):
    """Generate an nx1 random array b and solve the linear system Ax=b
    where A is the tri-diagonal array in Problem 2 of size nxn
    """
    b = np.random.randint(100, size=(n,1))
    a = Problem2(n)
    x = np.linalg.solve(a,b)
    return x

def Problem4(n, sparse=False):
    """Write a function that accepts an integer argument n and returns
    (lamba)*n^2 where (lamba) is the smallest eigenvalue of the sparse 
    tri-diagonal array you built in Problem 2.
    """
    A = Problem2(n)
    egvals, egvecs = sl.eigs(A,k=n-2, which = 'SM') 
    minegval = np.min(egvals)
    minegval = minegval*n**2
    return minegval

