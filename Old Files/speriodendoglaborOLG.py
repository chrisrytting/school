from __future__ import division
from scipy import optimize
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

#Calibration values

'''
        A - Firm Productivity
    alpha - capital share of income
    gamma - risk aversion
        s - periods of life
    sigma - Something else
        T - time periods
    years - years in each time period
     beta - discounting factor
    delta - depreciation rate
       xi - parameter to help TPI converge
'''

A = 1
alpha = 0.35
gamma = 2.9
s = 7
sigma = 2.9
T = 80
years = 60./s
beta = 0.96**years
delta = 1 - (1-.05)**years 
xi = .2
initval = 1.1

###Steady state with endogenous labor###

def sswage(kvec,lvec):
    K = np.sum(kvec) 
    L = np.sum(lvec)
    return (1-alpha)*(((K)/L)**alpha) 

def ssrental(kvec, lvec):
    K = np.sum(kvec) 
    L = np.sum(lvec)
    return alpha*((L/(K))**(1-alpha)) - delta

def uprime(c_s):
    return c_s**(-gamma)

#Function to calculate steady state labor and capital values
def opt(KLguess):
    kvec = np.zeros(s + 1)
    kvec[1:-1] = KLguess[:s-1]
    lvec = KLguess[s-1:]
    ssw = sswage(kvec, lvec)
    ssr = ssrental(kvec, lvec)
    ssapprate = 1 + ssr

    #Vectors for capital Euler equation
    Kks = kvec[:-2]
    Kks1 = kvec[1:-1]
    Kks2 = kvec[2:]
    Kls = lvec[:-1]
    Kls1 = lvec[1:]

    #Vectors for labor Euler equation
    Lks = kvec[:-1]
    Lks1 = kvec[1:]
    Lls = lvec
    
    #Capital Euler equation
    Keq = uprime(Kls*ssw + ssapprate *Kks - Kks1) - beta * ssapprate * uprime(Kls1 * ssw + ssapprate * Kks1 - Kks2)
    #Labor Euler equation
    Leq = ssw * uprime(Lls * ssw + ssapprate * Lks - Lks1) - ( 1 - Lls) ** -sigma

    return np.append(Keq,Leq)


#Calculate steady state values for capital and labor.
guess = np.append(np.ones(s-1)*.1, np.ones(s)*.95)
ssvalues = optimize.fsolve(opt, guess)

#Steady state vectors
kssvec = ssvalues[:s-1]
lssvec = ssvalues[s-1:]
print "Capital Steady States:\n", kssvec
print "Labor Steady States:\n", lssvec

#Steady state values of k, l, r, and w
kbar = np.sum(kssvec)
lbar = np.sum(lssvec)
rbar = ssrental(kbar,lbar)
wbar = sswage(kbar,lbar)
toprow = kssvec * initval 


###TPI###


#Initialize guesses for K path and L path
Kguess = np.linspace(initval*kbar, kbar, T)
Lguess = np.linspace(lbar, lbar, T)

#Calculate a vector of wages using L and K paths
def wage(kvec,lvec):
    return (1-alpha)*(((kvec)/lvec)**alpha)

#Calculate a vector of rental rates using L and K paths
def rental(kvec,lvec):
    return alpha*((lvec/(kvec))**(1-alpha)) - delta

#This function calculates the numbers to fill in the capital and labor matrices on or above the main diagonal
def calc_upper(klguess, wagevec, rentalvec, toprow, iteration):
    #Split up guess into capital guesses and labor guesses
    kguess = klguess[:iteration]
    lguess = klguess[iteration:]

    #Initialize vectors for labor Euler equation
    #   Wage, rental rate, k1, and k2
    lw1 = wagevec[:iteration + 1]
    lr1 = rentalvec[:iteration + 1]
    kvec = np.hstack((toprow[s-2-iteration],kguess, 0))
    lk1 = kvec[:-1]
    lk2 = kvec[1:]


    #Initialize wage and rental vectors to be used in capital Euler equation
    wagevectemp = wagevec[:iteration + 1] 
    rentalvectemp = rentalvec[:iteration +1]

    #   Wage 1 and 2
    kw1 = wagevectemp[:-1]
    kw2 = wagevectemp[1:]

    #   Rental rate 1 and 2
    kr1 = rentalvectemp[:-1]
    kr2 = rentalvectemp[1:]

    #   Labor 1 and 2
    kl1 = lguess[:-1]
    kl2 = lguess[1:]

    #   Capital 1, 2, and 3
    kk1 = kvec[:-2]
    kk2 = kvec[1:-1]
    kk3 = kvec[2:]

    #On the first iteration, we will only calculate the top right entry of the labor matrix
    if iteration == 0:
        #Labor Euler equation
        leq = lw1 * uprime(lguess * lw1 + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
        return leq

    #Starting on the second iteration, we will calculate the labor and capital unknowns at once in a system of equations.
    if iteration >= 1:
        #Labor Euler equation
        leq = lw1 * uprime(lguess * lw1 + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
        #Capital Euler equation
        keq = uprime(kw1*kl1 + (1 + kr1)*kk1 - kk2) - beta * (1 + kr2)*uprime(kw2*kl2 + (1 + kr2) * kk2 - kk3)
        return np.append(keq, leq)

#This function calculates the rest of the labor and capital matrices
def calc_lower(klguess, wagevec, rentalvec, iteration):
    #Split up klguess
    kguess = klguess[:s-1]
    lguess = klguess[s-1:]

    #Initialize arrays for labor Euler equation
    #   Wage, rental rate, savings 1 and savings 2
    lw1 = wagevec[iteration:iteration + s]
    lr1 = rentalvec[iteration:iteration + s]
    kvecout = np.hstack((0,kguess, 0))
    lk1 = kvecout[:-1]
    lk2 = kvecout[1:]

    #Initialize arrays for capital Euler equation
    wagevectemp = wagevec[iteration:iteration + s] 
    rentalvectemp = rentalvec[iteration:iteration + s]

    #   Wage 1 and 2
    kw1 = wagevectemp[:-1]
    kw2 = wagevectemp[1:]

    #   Rental rate 1 and 2
    kr1 = rentalvectemp[:-1]
    kr2 = rentalvectemp[1:]

    #   Labor 1 and 2
    kl1 = lguess[:-1]
    kl2 = lguess[1:]

    #   Capital 1, 2, and 3
    kk1 = kvecout[:-2]
    kk2 = kvecout[1:-1]
    kk3 = kvecout[2:]

    #   Labor and capital Euler equations
    leq = lw1 * uprime(lguess * lw1 + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
    keq = uprime(kw1*kl1 + (1 + kr1)*kk1 - kk2) - beta * (1 + kr2)*uprime(kw2*kl2 + (1 + kr2) * kk2 - kk3)

    return np.append(keq, leq)

#Initialize iterations and distance between K and L guesses and updates.
kdifference = 10**10
ldifference = 10**10
iters = 0

'''
print "\n\n\n BEGINNING TPI \n\n\n"

while kdifference > 10**-9 or ldifference > 10**-9:

    iters += 1
    print "\n\nIteration #{}:\n\n".format(iters)

    #Construct longer K vector after steady state is reached
    Kguess = np.append(Kguess, np.ones(s) * Kguess[-1])
    Lguess = np.append(Lguess, np.ones(s) * Lguess[-1])
    
    #Construct longer w vector after steady state is reached
    wagevec = wage(Kguess,Lguess)

    #Construct longer r vector after steady state is reached
    rentalvec = rental(Kguess,Lguess)

    #Initialize capital and labor matrices
    kmatrix = np.zeros((T+s, s-1))
    lmatrix = np.zeros((T+s, s))

    #Fill out matrices on and above the main diagonal
    for i in xrange(s-1):
        #tud = time until death
        tud = i+1
        lguess = lssvec[-(i + 1):] 
        #On first iteration, solve only for the top right entry of the l matrix, which is unknown
        if i == 0:
            new_l_vec = optimize.fsolve(calc_upper, lguess, args=(wagevec, rentalvec, toprow, i))
            lmatrix = lmatrix + sp.sparse.diags(new_l_vec, s-1, shape = (T+s, s))
        #Thereafter, solve for both capital and labor unknowns
        if i > 0:
            kguess = kssvec[-(i):] 
            klguess = np.hstack((kguess, lguess))
            new_kl_vec = optimize.fsolve(calc_upper, klguess, args=(wagevec, rentalvec, toprow, i))
            new_k_vec = new_kl_vec[:tud-1]
            new_l_vec = new_kl_vec[tud-1:]
            kmatrix = kmatrix + sp.sparse.diags(new_k_vec, s-1-i, shape = (T+s, s-1))
            lmatrix = lmatrix + sp.sparse.diags(new_l_vec, s-1-i, shape = (T+s, s))


    #Stack shocked individual capital steady state values on capital matrix begun above
    kmatrix = np.vstack([toprow, kmatrix])[:-1,:]

    #Now fill out the matrix below the main diagonal
    for i in xrange(0,T):
        klguess = np.append(kssvec, lssvec)
        new_kl_vec = optimize.fsolve(calc_lower, klguess, args = (wagevec, rentalvec, i))

        new_k_vec = new_kl_vec[:s-1]
        new_l_vec = new_kl_vec[s-1:]

        kmatrix = kmatrix + sp.sparse.spdiags(new_k_vec, -(i+1), T+s, s-1)
        lmatrix = lmatrix + sp.sparse.spdiags(new_l_vec, -i,T+s, s)
    #K and l matrices completed at this point, now
    kmatrix = kmatrix[:T,:]
    lmatrix = lmatrix[:T,:]

    #sum up columns of k and l,
    Updated_K = (kmatrix.sum(axis = 1)).T
    Updated_L = (lmatrix.sum(axis = 1)).T
    Updated_K = np.ravel(Updated_K)
    Updated_L = np.ravel(Updated_L)

    #take the norm between the two,
    kdifference = la.norm(Kguess[:T]-Updated_K)
    ldifference = la.norm(Lguess[:T]-Updated_L)
    print kdifference
    print ldifference

    #if the norm between update and initial guess is too big, then update guess and go through loop again.
    Kguess = xi*Updated_K + ((1-xi)*Kguess[:T])
    Lguess = xi*Updated_L + ((1-xi)*Lguess[:T])
    print iters
    
#When the error is small enough, graph the latest K and L updates on T

#K graph
x = np.linspace(0,T,T)
plt.plot(x, Updated_K)
plt.title("Aggregate K path")
plt.xlabel("T")
plt.ylabel("K")
plt.show()

#L graph
x = np.linspace(0,T,T)
plt.plot(x, Updated_L)
plt.title("Aggregate L path")
plt.xlabel("T")
plt.ylabel("L")
plt.show()
'''
###Chapter 4###

ability_types = np.genfromtxt("e_js.txt", delimiter=",")
ej_1 = ability_types[:,0]
ej_2 = ability_types[:,1]
ej_3 = ability_types[:,2]
ej_4 = ability_types[:,3]
ej_5 = ability_types[:,4]
ej_6 = ability_types[:,5]
ej_7 = ability_types[:,6]
x = np.linspace(1,80,80)

fig, ax  = plt.subplots()
plt.plot(x,ej_1, label = '0-24%')
plt.plot(x,ej_2, label = '25-49%')
plt.plot(x,ej_3, label = '50-69%')
plt.plot(x,ej_4, label = '70-79%')
plt.plot(x,ej_5, label = '80-89%')
plt.plot(x,ej_6, label = '90-98%')
plt.plot(x,ej_7, label = '99-100%')
plt.title('Ability types over lifetime')
plt.ylabel('Ability')
plt.xlabel('S')
legend = ax.legend(loc= "upper right", shadow =True)
plt.show()



