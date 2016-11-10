from __future__ import division
import scipy.optimize as opt 
from tabulate import tabulate
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import lines as lines

class simplex(object):
    def __init__(self, c, A, b):
        self.unbounded = False
        sizeA = np.shape(A)
        c = -c
        m = sizeA[0]
        n = sizeA[1]
        I = np.eye(m)
        self.shape = (m + 1, m + n + 1)
        self.table = np.zeros(self.shape)
        self.table[:1,:n] = c
        self.table[1:,:n] = A
        self.table[1:,n:n+m] = I
        self.table[1:,n+m:] = b

    def pivotcolumn(self):
        toprow = self.table[:1]
        negindices = []
        toprow = toprow.flatten()
        for i,x in enumerate(toprow):
            if x < 0:
                negindices.append(i)
        return negindices[0]

    def pivotrow(self):
        col = self.pivotcolumn()
        ratios =  self.table[1:,-1:]/ self.table[1:, col:col+1]

        if ((ratios<0).all()):
            self.unbounded = True
            return
        else:
            ratios = ratios.flatten()
            smallest = 10**9
            smallestind = np.inf
            smallestifzero = np.inf
            smallestindifzero = np.inf

            for i,x in enumerate(ratios):
                if x < smallest:
                    if x > 0:
                        smallest = x
                        smallestind = i
                    if x == 0:
                        smallestifzero = x
                        smallestindifzero = i
            if smallest == 10**9:
                return smallestifzeroind
            else:
                return smallestind




    def pivot(self):

        if (self.table[1:,-1:] < 0).any():
            print "The origin is infeasible. Negative entries of B"
            return
        if (self.table[0,:] >= 0).all():
            print "There is no column to pivot on. All entries of top row are non-negative. We have here the final result."
            return self.table
        m,n = np.shape(self.table) 
        r,c = self.pivotrow(), self.pivotcolumn()
        if self.unbounded == True:
            return
        if r == m - 2 :
            for i in xrange(m-1):
                ratiotomult = -self.table[i,c]/self.table[r+1,c]
                selftable = self.table[r+1,:]*ratiotomult + self.table[i,:]
                self.table[i] = selftable
            return self.table
        else:
            for i in xrange(r + 1):
                ratiotomult = -self.table[i,c]/self.table[r+1,c]
                selftable = self.table[r+1,:]*ratiotomult + self.table[i,:]
                self.table[i] = selftable
            for i in xrange(r+2,m):
                ratiotomult = -self.table[i,c]/self.table[r+1,c]
                selftable = self.table[r+1,:]*ratiotomult + self.table[i,:]
                self.table[i] = selftable
            return self.table

    def PIVOT(self):
        print self.table
        if (self.table[0,:] >= 0).all():
            print "There are no negative values in the first row."
        while (self.table[0,:] < 0).any():
            a = self.table.copy()
            self.pivot()
            if self.unbounded == True:
                print "Sorry. Unbounded."
                return
            b = self.table.copy()
            if (a==b).all():
                print "There is a problem with this matrix."
                return self.table 
            print self.table
            
        return self.table





        

#test
print 'Test case 1'
b = np.array([(0,0,1)])
b = b.T
c = np.array([(10,-57,-9,-24)])
A = np.array([(.5,-5.5,-2.5,9), (.5,-1.5,-.5,1), (1, 0, 0, 0)])
test = simplex(c,A,b)
print test.table
print test.pivot()
print test.table
print test.pivot()
'''
print test.table
print test.pivot()
print test.table
print test.pivot()
print test.table
print test.pivot()
test.PIVOT()

'''
'''
print 'Test case 2'    

b = np.array([(0, 18, 4)])
b = b.T
c = np.array([(-3,1)])
A = np.array([(1, 3), (2,3), (1,-1)])
test = simplex(c,A,b)

print test.pivot()


print test.PIVOT()
'''

#9.5
def f1(x):
    return 5-(1./3.)*x

def f2(x):
    return 6-(2./3.)*x

def f3(x):
    return -4+x

def f4(x):
	return 11+x
def f5(x):
	return 27-x
def f6(x):
	return 18-(2./5.)*x
x2 = np.linspace(-1., 7., 100)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(x2, f1(x2))
plt.plot(x2, f2(x2))
plt.plot(x2, f3(x2))
A =  6, 4, 0, 0, 3
B =  2, 0, 0, 5, 4
plt.plot(A, B)
for xy in zip(A,B):
    ax.annotate('(%s, %s)' % xy, xy = xy, textcoords = 'offset points')
ax.axhline(y=0, color = 'k')
ax.axvline(x=0, color = 'k')
'''
plt.grid()
plt.axes()
plt.show()
'''

y = np.linspace(-5., 30., 35)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(y, f4(y))
plt.plot(y, f5(y))
plt.plot(y, f6(y))
A =  15, 27,0,0,5
B =  12,0, 0,11,16
plt.plot(A, B)
for xy in zip(A,B):
    ax.annotate('(%s, %s)' % xy, xy = xy, textcoords = 'offset points')
ax.axhline(y=0, color = 'k')
ax.axvline(x=0, color = 'k')
plt.show()problem922.pnoproblem922.png
'''
plt.grid()
plt.axes()
plt.show()
'''
#9.6
print "EXERCISE 9.6"

b = np.array([(15,18,4)])
b = b.T
c = np.array([(3,1)])
A = np.array([(1,3), (2,3), (1,-1)])
test = simplex(c,A,b)
print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")

print test.table
print test.pivot()
print test.table
print test.pivot()

print "EXERCISE 9.6 part 2"

b = np.array([(11,27,90)])
b = b.T
c = np.array([(4,6)])
A = np.array([(-1,1), (1,1), (2,5)])
test = simplex(c,A,b)


print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")

#9.7
print "EXERCISE 9.7"

b = np.array([(1800, 150)])
b = b.T
c = np.array([(4,3)])
A = np.array([(15, 10), (1,1)])
test = simplex(c,A,b)


print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")


'''
#9.8(i)
print "EXERCISE 9.8 part 1"

b = np.array([(-8,6,3)])
b = b.T
c = np.array([(1,2)])
A = np.array([(-4,-2), (-2,3), (1,0)])
test = simplex(c,A,b)


print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")

'''
#9.8
print "EXERCISE 9.8 part 2"

b = np.array([(5,2)])
b = b.T
c = np.array([(15,15,-12)])
A = np.array([(5,3,-4), (3,5,-3)])
test = simplex(c,A,b)

print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print test.pivot()
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")

#9.8 (iii)
print "EXERCISE 9.8 part 3"

b = np.array([(4,6)])
b = b.T
c = np.array([(-3,1)])
A = np.array([(0,1), (-2,3)])
test = simplex(c,A,b)


print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")

#9.19
print "EXERCISE 19"

b = np.array([(3,5,4)])
b = b.T
c = np.array([(1,1)])
A = np.array([(2,1), (1,3), (2,3)])
test = simplex(c,A,b)


print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")

print "EXERCISE 19 part 2"

b = np.array([(3,5,4)])
b = b.T
c = np.array([(1,1)])
A = np.array([(2,1), (1,3), (2,3)])
test = simplex(c,A,b)


print tabulate(test.table, tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")
print tabulate(test.pivot(), tablefmt="latex")

#9.22
print "EXERCISE 22"

b = .95

r = .05 

a = 1  

g1 = (3*r - 2*b)/(2) + (1)/(2) * np.sqrt( (2*b - 3*r)**2 - 8*(r-b) )

g2 = (3*r - 2*b)/(2) - (1)/(2) * np.sqrt( (2*b - 3*r)**2 - 8*(r-b) )

def function(guess):
	c1 = guess[0]
	c2 = guess[1]

	equation1 = c1 * np.exp(g1*0) + c2 * np.exp(g2 *0) - 1
	equation2 = c1 * np.exp(g1*1) + c2 * np.exp(g2 *1)     

	return equation1, equation2

guess = np.array([0.5,0.5])

answer = opt.fsolve(function,guess,xtol = 0.0000000001)

print 'c1 is ', answer[0]
print 'c2 is ', answer[1]

print 'answer', answer[0] * np.exp(g1*0) + answer[1] * np.exp(g2 *0)
print 'answer', answer[0] * np.exp(g1*1) + answer[1] * np.exp(g2 *1)
print 'gamma1', g1
print 'gamma2', g2

x = np.linspace(0,1,100)
y = answer[0] * np.exp(g1*x) + answer[1] * np.exp(g2 *x)
plt.plot(x,y)
plt.show()
