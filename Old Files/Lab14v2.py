import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import timeit
from timeit import Timer
import decimal
import time
import cymodule
 
 
print "Exercise 1"
np.set_printoptions(precision=30)
print "Created floating point class--see code"
 
class Float(object):
 
    def __init__(self,exponent=1,significand=10,number=0.0):
        self.exponent=exponent
        self.significand=significand
        self.float=number
 
    def convert(self,number):
        return number*self.significand**self.exponent
 
    def __repr__(self,number):
        newnumb=convert(self,number)
        return str(newnumb)
 
    def copy(self,number):
        return float(exponent=self.exponent,significand=self.significand, number=number)
 
    def __add__(self,other):
        if self.exponent != other.exponent:
            return "Error--different exponents!"
        if self.significand != other.significand:
            return "Error--different significands!"
        return float(exponent=self.exponent,significand=self.significand,number=self.float+other.float)
 
    def __mul__(self,other):
        if self.significand != other.significand:
            return "Error--different significands!"
        return float(exponent=self.exponent+other.exponent,significand=self.significand,number=self.float*other.float)
 
    def __truncate__(self,digits):
        return float(exponent=self.exponent,significand=self.significand,number=round(self.round,digits))
 
 
class Float_error(object):
 
    def __init__(self,exponent=1,significand=10,number=0.0,truenum=1,error=0):
        self.exponent=exponent
        self.significand=significand
        self.float=number
        self.truenum=truenum
        self.error=error
 
    def convert(self):
        error=abs(self.float*self.significand**self.exponent)-self.truenum
        self.error+=error
        return (self.float)*(self.significand)**self.exponent
 
    def __repr__(self):
        x=self.float*self.significand**self.exponent
        return str(x)
 
    def copy(self,number):
        return Float_error(exponent=self.exponent,significand=self.significand,number=number,truenum=self.truenum,error=self.error)
 
    def __add__(self, other):
        if self.exponent != other.exponent:
            return "Error--different exponents!"
        if self.significand != other.significand:
            return "Error--different significands!"
        newnum=self.convert()+other.convert()#convert(self.float+other.float)
        newerror=newnum-(self.truenum+other.truenum)
        return Float_error(exponent=self.exponent,significand=self.significand, \
        number=self.float+other.float,truenum=self.truenum+other.truenum, \
        error=newerror)
 
    def __sub__(self,other):
        if self.exponent != other.exponent:
            return "Error--different exponents!"
        if self.significand != other.significand:
            return "Error--different significands"
        newnum=convert(self.float-other.float)
        newerror=newnum-(self.truenum-other.truenum)
        return Float_error(exponent=self.exponent,significand=self.significand,number=self.float-other.float,truenum=self.truenum-other.truenum,error=newerror)
 
    def __mul__(self, other):
        if self.significand != other.significand:
            return "Error--different significands"
        newnum=Float_error(exponent=self.exponent+other.exponent,significand=\
        self.significand, number=self.float*other.float,truenum=self.truenum+other.truenum, error=0)
        newerror=abs(newnum.convert-(self.truenum*other.truenum))
        return Float_error(exponent=self.exponent+other.exponent,significand=\
        self.significand, number=self.float*other.float,truenum=\
        self.truenum*other.truenum, error=newerror)
  
    def truncate(self):
        i=-20
        while 10**i <=self.error:
            num=10**i
            i+=1
        return Float_error(exponent=self.exponent,significand=self.significand, \
        number=round(number,i-1), truenum=self.truenum,error=self.error)
 
bob1 = Float_error(exponent=2,significand=10,number=1.23,truenum=123.)
bob2 = Float_error(exponent=2,significand=10,number=1.99,truenum=199.)
#bob3 = bob1*bob2--doesn't work
#print bob3--doesn't work
 
'''
bob2 = Float_error(exponent=2,significand=10,number=1.35,truenum=123.)
 
bob3 = bob1+bob2
bob4 = bob3.copy
print bob4
print bob3.significand
print bob3.exponent
print bob3.float
print bob3.truenum
print bob3.error 
x = bob3.convert
print x 
'''
 
 
print "Exercise 2"
 
def pysqrt64(A, reps):
    Ac=A.copy()
    I=Ac.view(dtype=np.int64) # get an integer view of the array
    I >>=1 # divide by two using a binary bit flip
    I += (1<<61)-(1<<51) # scale by a constant value
    for i in xrange(reps):
        Ac=.5*(Ac+A/Ac) # use an iterative method to increase accuracy, reps is the number of times.
    return Ac
 
a=np.array([1234.])
print pysqrt64(a,10)
 
start=time.time()
print pysqrt64(a,10)
end=time.time()
Timer1=end-start
print "Floating point square root time:", Timer1
start=time.time()
print np.sqrt(a)
end=time.time()
Timer2=end-start
print "Numpy square root time:",Timer2
start=time.time()
print cymodule.pysqrt64(a,10)
Timer3=time.time()-start
print "Cython Function time:",Timer3
