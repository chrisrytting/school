#Chris Rytting
import sys
from matplotlib import pyplot as plt
import numpy as np

#Exercise 1.9
print ""
print "Exercise 1.9"
"First Example with integers"
print (450*40545**4 - 50*70226**4 + 702260**2)/10.**4
"Second Example with floats"
print 450*4054.5**4 - 50*7022.6**4 + 7022.6**2

print 3/2

#Exercise 1.10
print "Exercise 1.10"
print ""
print "Formula = b**{t+1}+1"
def firstintnotfloat():
    print "THIS IS THE FIRST INT NOT FLOAT FUNCTION"
    i = 1
    while i < 200:
        allintsgood = False
        num_bins = 2**i
        divisions = 2**52
        binsoverdivisions = divisions % num_bins
        if binsoverdivisions == 0:
            allintsgood = True
        elif binsoverdivisions != 0:
            allintsgood = False
        if allintsgood == False:
            firstint = 2**i+1
            return firstint
        i += 1

first = firstintnotfloat()
print "The first integer that can't be expressed in floating point is: ", first
print "Here is the check: "
print "N-3: ", first-3, "as int =", first-3, "as float?"
print first-3 == float(first - 3)
print "N-2: ", first-2, "as int =", first-2, "as float?"
print first-2 == float(first - 2)
print "N-1: ", first-1, "as int =", first-1, "as float?"
print first-1 == float(first - 1)
print "N: ", first, "as int =", first, "as float?"
print first == float(first)
print "N+1: ", first + 1, "as int =", first + 1, "as float?"
print first + 1 == float(first + 1)
print "N+2: ", first + 2, "as int =", first + 2, "as float?"
print first + 2 == float(first + 2)

#Exercise 1.11 (i)
print ""
print "Exercise 1.11 (i)"
print "We give as an example N =", first,"which we found in the last exercise."
print "Let x = N, and let y = N+1"
print "When expressed as integers, x + y =",first + 1 + first
print "But when expressed as floats, x + y =", float(first) + float(first + 1)
print "Whose difference is,",first + first + 1 - float(first) - float(first + 1)
print "The relative error is:",np.abs(((first +1 + first) - (float(first + 1) + float(first)))/(first+1+first))

#Exercise 1.11 (ii)
print ""
print "Exercise 1.11 (ii)"
print "We give as an example N,", first,"which we found in the last exercise." 
print "Let x = N" 
print "As an integer, 1/X =", 1/first
print "As a float, 1/X =", 1/float(first)
print "The relative error (which had to be computed by hand) is:, (((1/2**53 + 1) - (1/2*53)) /1/2*53 + 1)"

#Exercise 1.11 (iii)
print ""
print "Exercise 1.11 (iii)"
print "As an example, N, N+1, N+2"
print float(first) + (float(first+1) + float(first + 2)) == (float(first) + float(first+1))+ float(first + 2) 

#Exercise 1.11 (iv)
print ""
print "Exercise 1.11 (iv)"
print "As an example, N, N + 1.2, N + 101.2"
y = 1.2
z = 101.2
print (float(first) + float(first+y)) * float(first + z) == (float(first) * float(first+z)) + (float(first+y) * float(first + z) )

#Exercise 1.12

print ""
print "Exercise 1.12"
print 'Difference =', sum((1./n**6 for n in xrange(1,1001))) - sum((1./n**6 for n in xrange(1000,0,-1)))
print "This number is off because summing from the back, we hit different gaps (1/1000**6 will be summed with 1/999**6 instead of the sum of the entire summation from 1 to 999) than summing from the front"


#Exercise 1.13

print ""
print "Exercise 1.13"
x = np.linspace(-3e-15, 3e-15, 1000)
y1 = x
y2 = (1-x) -1
plt.plot(x,y1, 'blue')
plt.plot(x,y2, 'green')
plt.plot()
plt.savefig("Graph1.png")
plt.show()

y3 = ((1-x)-1)/x
plt.plot(x,y3, 'red')
plt.savefig("Graph2.png")
plt.show()
print "The graph of (1-x)-1 looks jagged because it is hitting gaps that x does not hit."
print "Error is getting larger as x approaches 0 because absolute error is staying the same but relative error is getting larger."

