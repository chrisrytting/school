from __future__ import division
from scipy import optimize
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

#Calibration values
A = 1 #Productivity
alpha = 0.35 #cap. share of income
gamma = 3
periods = 7
sigma = 2.5
s = periods
T = 50
years = 60./periods
beta = 0.96**years
delta = 1 - (1-.05)**years 

'''
###EXERCISE 1###
l_tilde = 1

def g(b_k_v, n):
    b = b_k_v[0]
    k = b_k_v[1]
    v = b_k_v[2]
    return b * ( 1 - ( n / l_tilde ) ** v) ** ( 1 / v ) + k

def g_st(n, theta):
    return -(n**(1+theta))/(1+theta)

def opt(b_k_v, n, theta):
    eq1  =  g(b_k_v, n) - g_st(n, theta)
    return la.norm(eq1)

theta = 2/3
n = np.linspace(0 + 10**-9, l_tilde- 10**-9, 100)
guess_b_k_v = np.array([1,1,1]) 
bnds = ((0,None),(None,None),(0,None))
print optimize.minimize(opt, guess_b_k_v, args = (n, theta), method = 'L-BFGS-B', bounds = bnds)
results = optimize.minimize(opt, guess_b_k_v, args = (n, theta), method = 'L-BFGS-B', bounds = bnds).x
print results

good_b_k_v = results 

diff = (g(good_b_k_v,n) - g_st(n, theta))
print la.norm(diff)

fig, ax  =plt.subplots()
plt.plot(n,g(good_b_k_v, n), label = 'g')
plt.plot(n,g_st(n, theta), label = 'g_st')
plt.title('g(n) and g(n)st on n')
plt.ylabel('g(n) and g(n)st')
plt.xlabel('n')
legend = ax.legend(loc= "upper right", shadow =True)
plt.show()




'''
def wage(kvec,lvec):
    K = np.sum(kvec) 
    L = np.sum(lvec)
    return (1-alpha)*(((K)/L)**alpha)

def rental(kvec, lvec):
    K = np.sum(kvec) 
    L = np.sum(lvec)
    return alpha*((L/(K))**(1-alpha))

def uprime(c_s):
    return c_s**(-gamma)

def opt(KLvec):
    kvec = np.zeros(periods + 1)
    kvec[1:-1] = KLvec[:periods-1]
    lvec = KLvec[periods-1:]
    w = wage(kvec, lvec)
    r = rental(kvec, lvec)
    apprate = 1 + r - delta

    Kks = kvec[:-2]
    ##print "Kks: \n ", Kks, np.shape(Kks)
    Kks1 = kvec[1:-1]
    #print "Kks1: \n ", Kks1, np.shape(Kks1)
    Kks2 = kvec[2:]
    #print "Kks2: \n ", Kks2, np.shape(Kks2)

    Kls = lvec[:-1]
    #print "Kls: \n ", Kls, np.shape(Kls)
    Kls1 = lvec[1:]
    #print "Kls1: \n ", Kls1, np.shape(Kls1)

    Lks = kvec[:-1]
    #print "Lks: \n ", Lks, np.shape(Lks)
    Lks1 = kvec[1:]
    #print "Lks1: \n ", Lks1, np.shape(Lks1)
    Lls = lvec
    #print "Lls: \n ", Lls, np.shape(Lls)
    
    Keq = uprime(Kls*w + apprate*Kks - Kks1) - beta * apprate * uprime(Kls1 * w + apprate * Kks1 - Kks2)
    Leq = w*uprime(Lls*w + apprate * Lks - Lks1) - ( 1 - Lls) ** -sigma

    return np.append(Keq,Leq)

guess = np.array([.1,.1,.1,.1,.1,.1,.95,.95,.95,.95,.95,.95,.95])

#Steady state values for capital and labor.
ssvalues = optimize.fsolve(opt, guess)

kssvec = ssvalues[:s-1]
#Aggregate steady state capital
kbar = np.sum(kssvec)

lssvec = ssvalues[s-1:]
#Aggregate steady state labor
lbar = np.sum(lssvec)
        
###TPI###

initval = 2

def apprate(r, delta):
    apprate = 1 + r - delta
    return apprate

def wage(kvec):
    K = kvec 
    return (1-alpha)*(((K)/lbar)**alpha)

def rental(kvec):
    K = kvec
    return alpha*((lbar/(K))**(1-alpha))

Kguess = np.linspace(initval*kbar, kbar, T)
#print "k steady = \n {}".format(kssvec)
toprow = kssvec * initval 
#print "toprow = \n {}".format(toprow)

#Construct longer K vector after steady state is reached
Ksteady = np.ones(s)*kbar

#Construct longer w vector after steady state is reached
print Kguess
wsteady = wage(kbar)
wsteadyvec = np.ones(s)*wsteady

#Construct longer r vector after steady state is reached
rsteady = rental(kbar)
rsteadyvec = np.ones(s)*rsteady





def k_calc_upper(kvecout, wagevec, rentalvec, toprow, iteration):
    wagevectemp = wagevec[:tud+1]
    #print "wagevectemp = \n", wagevectemp
    rentalvectemp = rentalvec[:tud+1]
    #print "rentalvectemp = \n", rentalvectemp

    w1 = wagevectemp[:-1]
    #print "w1 = \n", w1
    w2 = wagevectemp[1:]
    #print "w2 = \n", w2

    r1 = rentalvectemp[:-1] 
    #print "r1 = \n", r1
    r2 = rentalvectemp[1:] 
    #print "r2 = \n", r2

    kvec = np.zeros(tud)
    k1 = np.copy(kvec)
    k1[0] = toprow[s-2-iteration]
    k1[1:] = kvecout[:-1]
    k2 = np.copy(kvecout)
    k3 = np.copy(kvec)
    k3[:-1] = kvecout[1:]
    #print "k1: \n", k1
    #print "k2: \n", k2
    #print "k3: \n", k3
    
    lssvectemp = lssvec[-(tud+1):]
#    print lssvectemp
    l1 = lssvectemp[:-1]
    #print l1
    l2 = lssvectemp[1:]
    #print l2
    eq = uprime(w1*l1 + apprate(r1, delta)*k1 - k2) - beta * apprate(r2, delta)*uprime(w2*l2 + apprate(r2, delta) * k2 - k3)
    #print "eq: \n", eq
    return eq

def k_calc_lower(kvecout, wagevec, rentalvec, iteration):
    #print "kvecout \n", kvecout
    wagevectemp = wagevec[iteration:iteration + s]
    #print "wage vector = \n", wagevectemp
    rentalvectemp = rentalvec[iteration:iteration + s]
    #print "rental vector = \n", rentalvectemp

    w1 = wagevectemp[:-1]
    #print "w1 vector = \n", w1
    w2 = wagevectemp[1:]
    #print "w2 vector = \n", w2

    r1 = rentalvectemp[:-1] 
    #print "r1 vector = \n", r1
    r2 = rentalvectemp[1:] 
    #print "r2 vector = \n", r2 

    kvec = np.zeros(s-1)
    #print "kvecout \n", kvecout
    #print "kvec \n", kvec
    k1 = np.copy(kvec)
    k1[1:] = kvecout[:-1]
    #print "k1 = ", k1
    k2 = np.copy(kvecout)
    #print "k2 = ", k2
    k3 = np.copy(kvec)
    k3[:-1] = kvecout[1:]
    #print "k3 = ", k3

    l1 = lssvec[:-1]
    #print l1
    l2 = lssvec[1:]
    #print l2
    eq = uprime(w1*l1 + apprate(r1, delta)*k1 - k2) - beta * apprate(r2, delta)*uprime(w2*l2 + apprate(r2, delta) * k2 - k3)
    return eq

difference = 10**10

iters = 0

while(difference > 10**-9):
    iters += 1
    print "\n\n ITERATION #{} \n\n".format(iters)
    Kvec = np.zeros(T + s)
    Kvec[:T] = Kguess
    Kvec[T:] = Ksteady
    w = wage(Kguess)
    wagevec = np.zeros(T + s)
    wagevec[:T] = w
    wagevec[T:] = wsteadyvec
    r = rental(Kguess)
    rentalvec = np.zeros(T + s)
    rentalvec[:T] = r
    rentalvec[T:] = rsteadyvec
    matrix = np.zeros((T+s, s-1))

    for i in xrange(s-2):
        #print "\n\n ITERATION #{} \n\n".format(i+1)
        tud = i+1
        kguess = np.ones(tud)*.1
        new_k_vec = optimize.fsolve(k_calc_upper, kguess, args=(wagevec, rentalvec, toprow, i))
        tempmatrix = sp.sparse.diags(new_k_vec, s-2-i, shape = (T + s, s - 1))
        matrix = matrix + tempmatrix
        #print "Solutions for k's: \n",new_k_vec

    matrix1 = np.vstack([toprow, matrix])

    matrix = np.zeros((T+s, s-1))

    counter = 1

    for i in xrange(1,T):
        #print "\n\n ITERATION #{} \n\n".format(i+1)
        kguess = np.ones(s-1)*.1
        new_k_vec = optimize.fsolve(k_calc_lower, kguess, args=(wagevec, rentalvec, i))
        new_k_vec = np.hstack([new_k_vec, 0])
        tempmatrix = sp.sparse.spdiags(new_k_vec, -counter,T + s, s - 1)
        matrix = matrix + tempmatrix
        counter += 1

    topzeros = np.zeros(6)
    matrix2 = np.vstack([matrix, topzeros])
    matrix = (matrix1 + matrix2)
    matrix = matrix[:T,:]
    print matrix
    Updated_K = (matrix.sum(axis = 1)).T
    Updated_K = np.ravel(Updated_K)
    #print "Kguess: \n {} \n Updated_K: \n {}".format(Kguess, Updated_K)
    difference = la.norm(Kguess-Updated_K)
    print difference
    Kguess = Updated_K
x = np.linspace(0,T,T)
plt.plot(x, Updated_K)
plt.title("Aggregate K path")
plt.xlabel("T")
plt.ylabel("K")
plt.show()


