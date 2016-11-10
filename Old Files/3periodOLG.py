from __future__ import division
from scipy import optimize
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

###3-Period Steady State###

print "\n\n EXERCISE 1 \n\n"

#Calibration
years = 20
beta = .96**years
delta = 1 - (1 - 0.05)**years
gamma = 3
A = 1
alpha = 0.35
Lbar = 2
T = 50

#Rental rate of capital function
def rental(k2,k3):
    return alpha*A*(Lbar/(k2+k3))**(1-alpha)

#Wage function
def wage(k2,k3):
    return (1-alpha)*A*((k2+k3)/Lbar)**(alpha)

#Derivative of utility CRRA function 
def uprime(c_s):
    return c_s**(-gamma)
    
def euler(kguess):
    k2 = kguess[0]
    k3 = kguess[1]
    eq1 = uprime(wage(k2,k3) - k2) - beta* (1 + rental(k2,k3) - delta) * uprime(wage(k2,k3) + (1 + rental(k2,k3) - delta)*k2 - k3)
    eq2 = uprime(wage(k2,k3) + (1 + rental(k2,k3) - delta) * k2 - k3) - beta * (1 + rental(k2,k3) - delta) * uprime((1 + rental(k2,k3) - delta) * k3)
    return eq1,eq2

k_opt_ss_guess = np.array([.5,.5])


k2_opt_ss, k3_opt_ss = optimize.fsolve(euler, k_opt_ss_guess)
print "Optimal steady state capital stock at age 2 = {}".format(k2_opt_ss)
print "Optimal steady state capital stock at age 3 = {}".format(k3_opt_ss)

w_opt_ss = wage(k2_opt_ss, k3_opt_ss)
r_opt_ss = rental(k2_opt_ss, k3_opt_ss)
print "Optimal steady state wage = {}".format(w_opt_ss)
print "Optimal steady state rental rate = {}".format(r_opt_ss)

c1 = lambda k2: w_opt_ss - k2
c2 = lambda k2, k3: w_opt_ss + (1 + r_opt_ss - delta) * k2 - k3
c3 = lambda k3: (1 + r_opt_ss - delta) * k3

c1_opt_ss = c1(k2_opt_ss)
c2_opt_ss = c2(k2_opt_ss, k3_opt_ss)
c3_opt_ss = c3(k3_opt_ss)

print "Optimal steady state level of consumption at age 1 = {}".format(c1_opt_ss)
print "Optimal steady state level of consumption at age 2 = {}".format(c2_opt_ss)
print "Optimal steady state level of consumption at age 3 = {}".format(c3_opt_ss)

###Exercise 2.2###

print "\n\n EXERCISE 2 \n\n"

#Calibration
years = 20
beta = .55 
delta = 1 - (1 - 0.05)**years
gamma = 3
A = 1
alpha = 0.35
Lbar = 2

#Rental rate of capital function
def rental(k2,k3):
    return alpha*A*(Lbar/(k2+k3))**(1-alpha)

#Wage function
def wage(k2,k3):
    return (1-alpha)*A*((k2+k3)/Lbar)**(alpha)

#Derivative of utility CRRA function 
def uprime(c_s):
    return c_s**(-gamma)
    
def euler(kguess):
    k2 = kguess[0]
    k3 = kguess[1]
    eq1 = uprime(wage(k2,k3) - k2) - beta* (1 + rental(k2,k3) - delta) * uprime(wage(k2,k3) + (1 + rental(k2,k3) - delta)*k2 - k3)
    eq2 = uprime(wage(k2,k3) + (1 + rental(k2,k3) - delta) * k2 - k3) - beta * (1 + rental(k2,k3) - delta) * uprime((1 + rental(k2,k3) - delta) * k3)
    return eq1,eq2

k_opt_ss_guess = np.array([.5,.5])


k2_opt_ss, k3_opt_ss = optimize.fsolve(euler, k_opt_ss_guess)
print "Optimal steady state capital stock at age 2 = {}".format(k2_opt_ss)
print "Optimal steady state capital stock at age 3 = {}".format(k3_opt_ss)

w_opt_ss = wage(k2_opt_ss, k3_opt_ss)
r_opt_ss = rental(k2_opt_ss, k3_opt_ss)
print "Optimal steady state wage = {}".format(w_opt_ss)
print "Optimal steady state rental rate = {}".format(r_opt_ss)

c1 = lambda k2: w_opt_ss - k2
c2 = lambda k2, k3: w_opt_ss + (1 + r_opt_ss - delta) * k2 - k3
c3 = lambda k3: (1 + r_opt_ss - delta) * k3

c1_opt_ss = c1(k2_opt_ss)
c2_opt_ss = c2(k2_opt_ss, k3_opt_ss)
c3_opt_ss = c3(k3_opt_ss)

print "Optimal steady state level of consumption at age 1 = {}".format(c1_opt_ss)
print "Optimal steady state level of consumption at age 2 = {}".format(c2_opt_ss)
print "Optimal steady state level of consumption at age 3 = {}".format(c3_opt_ss)

print "\nIntuition: With a higher level of patience, individuals are willing to save more money. This leads to an increased capital stock in all periods, which drives the interest rate     down, wages up, and consumption in all three periods up. People are better off with more patience."


print "\n\n EXERCISE 3 \n\n"

#Rental rate of capital function
def rental_aggregate(Kpath):
    return alpha*A*(Lbar/(Kpath))**(1-alpha)

#Wage function
def wage_aggregate(Kpath):
    return (1-alpha)*A*((Kpath)/Lbar)**(alpha)

#Determines starting place for Kpath
initial_value = 1.5

#Find steady state K bar
Kbar_ss = k2_opt_ss + k3_opt_ss

#Initialize guess for K path
Kpath_guess = np.linspace(initial_value * Kbar_ss, Kbar_ss, T)

def calculate_k32(k32_guess):
    equation_1 = uprime(wage_vector[0] + (1 + rental_vector[0] - delta) * k2_opt_ss * initial_value - k32_guess) - beta * (1 + rental_vector[1] - delta) * uprime((1 + rental_vector[1] - delta) * k32_guess)
    return equation_1

def calculate_kpaths(k_twist_guess, i):
    k22 = k_twist_guess[0]
    k33 = k_twist_guess[1]
    equation_1 = uprime(wage_vector[i] - k22) - beta * (1 + rental_vector[ i + 1 ] - delta) * uprime( wage_vector[ i + 1 ] + ( 1 + rental_vector[i + 1] - delta ) * k22 - k33)
    equation_2 = uprime(wage_vector[i+1] + (1 + rental_vector[i+1] - delta) * k22 - k33) - beta * (1 + rental_vector[i+2] - delta) * uprime((1 + rental_vector[i + 2] - delta) * k33)
    return equation_1, equation_2

difference = 10**10

counter = 0
while difference > 10**-9:
    counter += 1
    #Find wage and rental rate vectors using the aggregate capital stock path
    wage_vector = wage_aggregate(Kpath_guess)
    rental_vector = rental_aggregate(Kpath_guess)

    #Find k32
    k32_guess = .1
    k32 = optimize.fsolve(calculate_k32, k32_guess)

    #Lengthen rental_vector by one, appending the steady state rental rate(so as to not go out of range
    rental_vector = np.hstack([rental_vector, rental_vector[-1:]])

    #Start to fill out kmatrix (50,2)
    kmatrix = np.zeros((T+1,2))
    kmatrix[0,0] = k2_opt_ss * initial_value
    kmatrix[0,1] = k3_opt_ss * initial_value
    kmatrix[1,1] = k32
    k_twist_guess = np.array([.1, .1])

    for i in xrange(T-1):
        kmatrix[ i+1, 0 ], kmatrix[ i + 2, 1 ] = optimize.fsolve(calculate_kpaths, k_twist_guess, i)

    kmatrix = kmatrix[:-1,:]
    K_update = np.sum(kmatrix, axis = 1)
    difference = K_update - Kpath_guess 
    difference = la.norm(difference)
    Kpath_guess = K_update 
    

print "Best estimation of aggregate capital stock path: \n {}".format(K_update)
    
print "\n\n EXERCISE 4 \n\n"

x = np.linspace(0, T, T)
plt.plot(x, K_update)
plt.xlabel('Time')
plt.ylabel('K')
plt.title('Aggregate Capital Stock Over Time')
plt.show()

print "It took 29 iterations for the path to converge, and 17 periods for the economy to get within .0001 of the aggregate capital stock K bar"












