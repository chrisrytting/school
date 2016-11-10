
import numpy as np
import scipy
import scipy.sparse.linalg as sl
import scipy.sparse as sparse
import matplotlib.pyplot as plt
'''
Problem 1

a.
timeit func1(100)
1000 loops, best of 3: 439 mus per loop
timeit func1(200)
1000 loops, best of 3: 890 mus per loop
timeit func1(400)
1000 loops, best of 3: 1.8 ms per loop
timeit func1(800)
100 loops, best of 3: 3.77 ms per loop

fun1 = [439, 890, 1800, 3700]

b.
timeit func2(100)
10000 loops, best of 3: 1970 mus per loop
timeit func2(200)
10000 loops, best of 3: 7900 mus per loop
timeit func2(400)
1000 loops, best of 3: 31100 mus per loop
timeit func2(800)
1000 loops, best of 3: 125000 mus per loop

fun2 = [1970, 7900, 31100, 125000]

c.
timeit func3(100)
1000 loops, best of 3: 521 mus per loop
timeit func3(200)
100 loops, best of 3: 2.08 ms per loop
timeit func3(400)
100 loops, best of 3: 8.31 ms per loop
timeit func3(800)
10 loops, best of 3: 36 ms per loop

fun3 = [521, 2080, 8310, 36000]

d.
timeit func4(100)
1000 loops, best of 3: 632 mus per loop
timeit func4(200)
1000 loops, best of 3: 1.91 ms per loop
timeit func4(400)
100 loops, best of 3: 7.25 ms per loop
timeit func4(800)
10 loops, best of 3: 38.8 ms per loop

fun4 = [632, 1910, 7250, 38800]

e.
timeit func5(100)
1000 loops, best of 3: 530 mus per loop
timeit func5(200)
1000 loops, best of 3: 1.7 ms per loop
timeit func5(400)
100 loops, best of 3: 6.13 ms per loop
timeit func5(800)
10 loops, best of 3: 33.1 ms per loop
'''
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
plt.legend(["Function 1", "Function 2", "Function 3", "Function 4", "Function 5"])
#plt.show()

'''
d. 
function 1 takes the least time of all the functions because it has a significantly less amount of temporal complexity than does function 2 (which iterates through a for loop n^2 times instead of just n times and also happens to be the most time consuming function.

the last three functions have a significant amount of spatial complexity, but since there is so much space on the computer, this is less significant than the temporal complexity of functions 1 and 2.


'''

#Problem 2
def maketridiag(n):
    maindiag = np.linspace(2,2,n)
    otherdiag = np.linspace(-1,-1,n-1)
    otherdiag1 = np.linspace(-1,-1,n)
    tridiag = sparse.spdiags(maindiag, 0, n, n, format = 'csr') + sparse.spdiags(otherdiag, -1, n, n, format = 'csr') + sparse.spdiags(otherdiag1, 1, n, n, format = 'csr')
    tridiag = tridiag.todense()
    return tridiag
print maketridiag(12)

#Problem 3

def solvlin(n):
    b = np.random.randint(100, size=(n,1))
    a = maketridiag(n)
    x = np.linalg.solve(a,b)
    return x
print solvlin(5)

#Problem 4

def findmineg(n):
    A = maketridiag(n)
    egvals, egvecs = sl.eigs(A,k=n-2, which = 'SM') 
    minegval = np.min(egvals)
    minegval = minegval*n**2
    return minegval

print np.sqrt(findmineg(500))
#approaches pi as n approaches 500
