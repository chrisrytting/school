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
sigma = 3
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
    return (n**(1-theta))/(1-theta)

def opt(b_k_v, n, theta):
    eq1  =  (g(b_k_v, n) - g_st(n, theta))/g_st(n,theta)
    return la.norm(eq1)

theta = 2./3
n = np.linspace(0 + 10**-9, l_tilde- 10**-9, 100)
guess_b_k_v = np.array([.67,-.65,1.3]) 
results = optimize.minimize(opt, guess_b_k_v, args = (n, theta), method = 'TNC')
print results

good_b_k_v = np.array([0.6701, -0.6548, 1.3499])

diff = (g_st(n, theta) - g(good_b_k_v,n))/g_st(n, theta)
print la.norm(diff)

'''

def wage(kvec,lvec):
    K = np.sum(kvec) 
    L = np.sum(lvec)
    return (1-alpha)*(((K)/L)**alpha)

def rental(kvec, lvec):
    K = np.sum(kvec) 
    L = np.sum(lvec)
    return alpha*((L/(K))**(1-alpha)) - delta

def uprime(c_s):
    return c_s**(-gamma)

def opt(KLvec):
    kvec = np.zeros(s + 1)
    kvec[1:-1] = KLvec[:s-1]
    lvec = KLvec[s-1:]
    w = wage(kvec, lvec)
    r = rental(kvec, lvec)
    apprate = 1 + r

    Kks = kvec[:-2]
    #print "Kks: \n ", Kks, np.shape(Kks)
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

#Steady state values for capital and labor.
guess = np.append(np.ones(s-1)*.5, np.ones(s)*.95)
ssvalues = optimize.fsolve(opt, guess)
print "ssvec:\n", ssvalues

kssvec = ssvalues[:s-1]
#Aggregate steady state capital
kbar = np.sum(kssvec)

lssvec = ssvalues[s-1:]
#Aggregate steady state labor
lbar = np.sum(lssvec)
        
###TPI###

initval = .9


print "kssvec:\n", kssvec
print "kssvec*initval:\n", kssvec*initval 
print "lssvec:\n", lssvec
print "lssvec*initval:\n", lssvec*initval 

def apprate(r, delta):
    apprate = 1 + r - delta
    return apprate

def wage(kvec,lvec):
    return (1-alpha)*(((kvec)/lvec)**alpha)

def rental(kvec,lvec):
    return alpha*((lvec/(kvec))**(1-alpha))

Kguess = np.linspace(initval*kbar, kbar, T)
Lguess = np.linspace(lbar, lbar, T)
#print "Kguess\n", Kguess
#print "Lguess\n", Lguess
#print "k steady = \n {}".format(kssvec)
toprow = kssvec * initval 
#print "toprow = \n {}".format(toprow)

print "wbar = ", wage(Kguess,Lguess)
print "rbar = ", rental(Kguess,Lguess)






def calc_upper(klvecout, wagevec, rentalvec, toprow, iteration):
    kvecouttemp = klvecout[:iteration]
    #print 'kvecouttemp ', kvecouttemp
    lvecout = klvecout[iteration:]
    #print 'lvecout ', lvecout
    lw1 = wagevec[:iteration + 1]
    #print 'lwagevectemp ', lw1
    lr1 = rentalvec[:iteration + 1]
    #print 'lrentalvectemp ', lr1
    kvecout = np.hstack((toprow[s-2-iteration],kvecouttemp, 0))
    #print 'kvecout ', kvecout
    lk1 = kvecout[:-1]
    #print 'lk1 ', lk1
    lk2 = kvecout[1:]
    #print 'lk2 ', lk2


    wagevectemp = wagevec[:iteration + 1] 
    rentalvectemp = rentalvec[:iteration +1]

    kw1 = wagevectemp[:-1]
    #print 'kw1 ', kw1
    kw2 = wagevectemp[1:]
    #print 'kw2 ', kw2

    kr1 = rentalvectemp[:-1]
    #print 'kr1 ', kr1
    kr2 = rentalvectemp[1:]
    #print 'kr2 ', kr2

    kl1 = lvecout[:-1]
    #print 'kl1 ', kl1
    kl2 = lvecout[1:]
    #print 'kl2 ', kl2

    kk1 = kvecout[:-2]
    #print 'kk1 ', kk1
    kk2 = kvecout[1:-1]
    #print 'kk2 ', kk2
    kk3 = kvecout[2:]
    #print 'kk3 ', kk3

    if iteration == 0:
        leq = lw1 * uprime(lvecout * lw1 + (1 + lr1 - delta)*lk1 - lk2) - uprime(1 - lvecout)
        return leq
    if iteration >= 1:
        leq = lw1 * uprime(lvecout * lw1 + (1 + lr1 - delta)*lk1 - lk2) - uprime(1 - lvecout)
        keq = uprime(kw1*kl1 + (1 + kr1 - delta)*kk1 - kk2) - beta * (1 + kr2 - delta)*uprime(kw2*kl2 + (1 + kr2 - delta) * kk2 - kk3)
        #print "keq, ", keq, "leq, ", leq
        return np.append(keq, leq)

def calc_lower(klvecout, wagevec, rentalvec, iteration):
    kvecouttemp = klvecout[:s-1]
    ##print 'kvecouttemp ', kvecouttemp
    lvecout = klvecout[s-1:]
    #print 'lvecout ', lvecout
    lw1 = wagevec[iteration:iteration + s]
    #print 'lwagevectemp ', lw1
    lr1 = rentalvec[iteration:iteration + s]
    #print 'lrentalvectemp ', lr1
    kvecout = np.hstack((0,kvecouttemp, 0))
    #print 'kvecout ', kvecout
    lk1 = kvecout[:-1]
    #print 'lk1 ', lk1
    lk2 = kvecout[1:]
    #print 'lk2 ', lk2

    wagevectemp = wagevec[iteration:iteration + s ] 
    rentalvectemp = rentalvec[iteration:iteration + s]

    kw1 = wagevectemp[:-1]
    #print 'kw1 ', kw1
    kw2 = wagevectemp[1:]
    #print 'kw2 ', kw2

    kr1 = rentalvectemp[:-1]
    #print 'kr1 ', kr1
    kr2 = rentalvectemp[1:]
    #print 'kr2 ', kr2

    kl1 = lvecout[:-1]
    #print 'kl1 ', kl1
    kl2 = lvecout[1:]
    #print 'kl2 ', kl2

    kk1 = kvecout[:-2]
    #print 'kk1 ', kk1
    kk2 = kvecout[1:-1]
    #print 'kk2 ', kk2
    kk3 = kvecout[2:]
    #print 'kk3 ', kk3

    leq = lw1 * uprime(lvecout * lw1 + (1 + lr1 - delta)*lk1 - lk2) - uprime(1 - lvecout)
    keq = uprime(kw1*kl1 + (1 + kr1 - delta)*kk1 - kk2) - beta * (1 + kr2 - delta)*uprime(kw2*kl2 + (1 + kr2 - delta) * kk2 - kk3)
    #print "keq, ", keq, "leq, ", leq
    return np.append(keq, leq)

kdifference = 10**10
ldifference = 10**10

iters = 0


while kdifference > 10**-9 or ldifference > 10**-9:
    iters += 1

    print "\n\n ITERATION #{} \n\n".format(iters)

    #Construct longer K vector after steady state is reached
    Ksteady = np.ones(s)*kbar
    Kvec = np.zeros(T + s)
    Kvec[:T] = Kguess
    Kvec[T:] = Ksteady
    #print Kvec

    #Construct longer w vector after steady state is reached
    w = wage(Kguess,Lguess)
    wsteady = w[-1:]
    wsteadyvec = np.ones(s)*wsteady
    wagevec = np.zeros(T + s)
    wagevec[:T] = w
    wagevec[T:] = wsteadyvec
    #print wagevec

    #Construct longer r vector after steady state is reached
    r = rental(Kguess,Lguess)
    rsteady = r[-1:]
    rsteadyvec = np.ones(s)*rsteady
    rentalvec = np.zeros(T + s)
    rentalvec[:T] = r
    rentalvec[T:] = rsteadyvec
    #print rentalvec

    kmatrix = np.zeros((T+s, s-1))
    lmatrix = np.zeros((T+s, s))

    for i in xrange(s-1):
        #print "\n\n ITERATION #{} \n\n".format(i+1)
        tud = i+1
        lguess = lssvec[-(i + 1):] 
        #print lguess
        if i == 0:
            new_l_vec = optimize.fsolve(calc_upper, lguess, args=(wagevec, rentalvec, toprow, i))
            #print "L solutions: ",new_l_vec
            lmatrix = lmatrix + sp.sparse.diags(new_l_vec, s-1, shape = (T+s, s))

        if i > 0:
            kguess = kssvec[-(i):] 
            klguess = np.hstack((kguess, lguess))
            #print kguess
            new_kl_vec = optimize.fsolve(calc_upper, klguess, args=(wagevec, rentalvec, toprow, i))
            new_k_vec = new_kl_vec[:tud-1]
            new_l_vec = new_kl_vec[tud-1:]
            kmatrix = kmatrix + sp.sparse.diags(new_k_vec, s-1-i, shape = (T+s, s-1))
            lmatrix = lmatrix + sp.sparse.diags(new_l_vec, s-1-i, shape = (T+s, s))
            #print "K solutions: ",new_k_vec,"\n L solutions: ", new_l_vec
    kmatrix = np.vstack([toprow, kmatrix])[:-1,:]
    matrix = np.zeros((T+s, s-1))
    counter = 1
    for i in xrange(0,T):
        #print "\n\n ITERATION #{} \n\n".format(i+1)
        kguess = kssvec
        lguess = lssvec
        klguess = np.append(kguess, lguess)
        new_kl_vec = optimize.fsolve(calc_lower, klguess, args = (wagevec, rentalvec, i))
        new_k_vec = new_kl_vec[:s-1]
        new_l_vec = new_kl_vec[s-1:]
        kmatrix = kmatrix + sp.sparse.spdiags(new_k_vec, -(i+1), T+s, s-1)
        lmatrix = lmatrix + sp.sparse.spdiags(new_l_vec, -i,T+s, s)
        counter += 1
    kmatrix = kmatrix[:T,:]
    lmatrix = lmatrix[:T,:]
    print 'kmatrix: \n', kmatrix
    print 'kssvec: \n', kssvec
    print 'lmatrix: \n', lmatrix
    print 'lssvec: \n', lssvec
    Updated_K = (kmatrix.sum(axis = 1)).T
    Updated_L = (lmatrix.sum(axis = 1)).T

    Updated_K = np.ravel(Updated_K)
    Updated_L = np.ravel(Updated_L)

    #print "Kguess: \n {} \n Updated_K: \n {}".format(Kguess, Updated_K)
    kdifference = la.norm(Kguess-Updated_K)
    ldifference = la.norm(Lguess-Updated_L)
    print "Kguess:\n", Kguess
    print "Lguess:\n", Lguess
    print "Kdifference:\n", kdifference
    print "Ldifference:\n", ldifference
    Kguess = Updated_K
    Lguess = Updated_L
    print iters

x = np.linspace(0,T,T)
plt.plot(x, Updated_K)
plt.title("Aggregate K path")
plt.xlabel("T")
plt.ylabel("K")
plt.show()

