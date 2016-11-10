from __future__ import division
from scipy import optimize
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

 # -------------Calibration values---------------#

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
listolegend=['0-24%','25-49%', '50-69%', '70-79%', '80-89%', '90-98%','99-100%']
e_js=np.loadtxt('e_js.txt', delimiter=',')
s=np.linspace(0,80,80)
# fig, ax  = plt.subplots()
# for i in xrange(7):
#     plt.plot(s, e_js[:,i], label=listolegend[i])
# legend=ax.legend(loc= "upper right", shadow =True, title='ability types')
# plt.xlabel('periods (s)')
# plt.ylabel('ability levels')
# plt.title('ability levels over lifetime')
# plt.show()

J=np.shape(e_js)[1]
A = 1
alpha = 0.35
gamma = 2.9
s = 80
sigma = 2.9
T = 100
years = 80/s
beta = 0.96**years
delta = 1 - (1-.05)**years 
xi = .2
initval = 1.005
periods=s
d=delta
g=gamma
b=beta
###Steady state with endogenous labor###

def wage (klist,lList):
    w=(1-alpha)*((np.sum(klist))/(np.sum(lList)))**(alpha)
    return w

def rate (klist,lList):
    
    r= (alpha)*((np.sum(lList))/(np.sum(klist)))**(1-alpha)-d
    return r


def uprime(c_s):
    return c_s**(-gamma)

#Function to calculate steady state labor and capital values
def opt(listit,e):
    #------------- K vectors-----------#
    klist=np.copy(listit[:periods-1])
    # print "klist:{}".format(klist)
    lList=np.copy(listit[periods-1:])
    e1=e[:-1]
    e2=e[1:]
    # print "lList:{}".format(lList)
    ks=np.append(0,klist[:-1])
    # print "ks:{}".format(ks)
    ks1=np.copy(klist)
    # print "ks1:{}".format(ks1)
    ks2=np.append(klist[1:],0)
    # print "ks2:{}".format(ks2)
    labors=np.copy(lList[:-1])
    # print "labors:{}".format(labors)
    labors1=np.copy(lList[1:])
    # print "labors1:{}".format(labors1)

    # ------------wage and rental rate--------------#
    
    w=wage(klist, lList)
    r=rate(klist, lList)
    # print 1+r-d
    # labor vectors
    lLabors=np.copy(lList)
    # print "lLabors:{}".format(lLabors)
    Klabors=np.append(0,klist[:periods-1])
    # print "Klabors:{}".format(Klabors)
    Klabors1=np.append(klist[:periods-1],0)
    # print "Klabors1:{}".format(Klabors1)

    eq1 =(labors*w*e1+(1+r)*ks-ks1)**(-gamma)-(b*(1+r))*(labors1*w*e2+(1+r)*ks1-ks2)**(-gamma)
    eq2 =((w*lLabors*e+(1+r)*Klabors-Klabors1)**(-g))*w-((1-lLabors)**(-sigma))
    return np.append(eq1,eq2)

def checkerror(listit,e):
    ssvalues = optimize.fsolve(opt, listit,args=(e), xtol=1e-13)
    klist=np.copy(ssvalues[:periods-1])
    e1=e[:-1]
    e2=e[1:]
    # print "klist:{}".format(klist)
    lList=np.copy(ssvalues[periods-1:])
    # print "lList:{}".format(lList)
    ks=np.append(0,klist[:-1])
    # print "ks:{}".format(ks)
    ks1=np.copy(klist)
    # print "ks1:{}".format(ks1)
    ks2=np.append(klist[1:],0)
    # print "ks2:{}".format(ks2)
    labors=np.copy(lList[:periods-1])
    # print "labors:{}".format(labors)
    labors1=np.copy(lList[1:])
    # print "labors1:{}".format(labors1)

    # ------------wage and rental rate--------------#
    
    w=wage(klist, lList)
    r=rate(klist, lList)
    # print 1+r-d
    # labor vectors
    lLabors=np.copy(lList)
    # print "lLabors:{}".format(lLabors)
    Klabors=np.append(0,klist)
    # print "Klabors:{}".format(Klabors)
    Klabors1=np.append(klist,0)
    eq1 =(labors*w*e1+(1+r)*ks-ks1)**(-gamma)-(b*(1+r))*(labors1*w*e2+(1+r)*ks1-ks2)**(-gamma)
    eq2 =((w*lLabors*e+(1+r)*Klabors-Klabors1)**(-g))*w-((1-lLabors)**(-sigma))
    return np.append(ssvalues, np.max(np.abs(np.append(eq1,eq2))))

#Calculate steady state values for capital and labor.
guess = np.append(np.ones(s-1)*.025, np.ones(s)*.975)
ssmat=np.zeros(((2*s-1),J))
error=np.array([])

for j in xrange(J):
    ssvalues=checkerror(guess, e_js[:, j])
    ssmat[:, j]=ssvalues[:-1]
    error=np.append(error, ssvalues[-1])

print "ssmat values: {} and shape {}".format(ssmat, np.shape(ssmat))
print "here is the vector of errors \n {}".format(error)
ind=np.argmax(np.abs(ssmat))
print "here is the biggest value in ssmat: {}".format(ssmat.flatten()[ind])
#Steady state vectors
kssmat = ssmat[:s-1, :]
kssmatuse=kssmat
lssmatuse=ssmat[s-1:,:]
lssmat = ssmat[s-1:,:]*e_js
kssvec = kssmat.sum(1)
lssvec = lssmat.sum(1)
kssmat = np.vstack((np.zeros((1, J)),kssmat, np.zeros((1, J))))
print "Capital Steady States:\n", kssvec
print "Labor Steady States:\n", lssvec

#Steady state values of k, l, r, and w
kbar = np.sum(kssvec)
lbar = np.sum(lssvec)
rbar = rate(kbar,lbar)
wbar = wage(kbar,lbar)
print "Here is wbar and rbar {} {}".format(wbar, rbar)
cmat=np.zeros((s,J))
for j in xrange(J):
    cvec=wbar*lssmat[:, j]+(1+rbar)*kssmat[:-1,j]-kssmat[1:,j]
    cmat[:, j]=cvec
print "Here is the matrix of consumptions: {}".format(cmat)
print "Here is kbar and lbar: {} {}".format(kbar, lbar)
cbar=cmat.sum(1).sum(0)
y=cbar+delta*kbar
print y
y=(kbar**alpha)*(lbar**(1-alpha))
print y
toprow = kssmatuse * initval 
toprow=np.reshape(toprow,(1,s-1,J))
if np.max(np.abs(error))>1e-5:
    print "error too high for steady state!: {}".format(error)



#TPI###

if np.max(np.abs(error))<1e-5:
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
    def calc_upper(klguess, wagevec, rentalvec, toprow, iteration,ejs):
        #Split up guess into capital guesses and labor guesses
        kguess = klguess[:iteration]
        lguess = klguess[iteration:]
        length=len(lguess)
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
            leq = lw1 * uprime(lguess * lw1*ejs + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
            return leq

        #Starting on the second iteration, we will calculate the labor and capital unknowns at once in a system of equations.
        if iteration >= 1:
            ejs=ejs[(-length):, j]
            e1=ejs[:-1]
            e2=ejs[1:]
            #Labor Euler equation
            leq = lw1 * uprime(lguess*lw1*ejs + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
            #Capital Euler equation
            keq = uprime(kw1*kl1*e1 + (1 + kr1)*kk1 - kk2) - beta * (1 + kr2)*uprime(kw2*kl2*e2 + (1 + kr2) * kk2 - kk3)
            return np.append(keq, leq)

    def check_upper(klguess, wagevec, rentalvec, toprow, iteration,ejs):
        solvec=optimize.fsolve(calc_upper, klguess, args=(wagevec, rentalvec, toprow, iteration,ejs))
        kguess = solvec[:iteration]
        lguess = solvec[iteration:]

        length=len(lguess)

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
            leq = lw1 * uprime(lguess * lw1* ejs + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
            return np.append(solvec, np.max(np.abs(leq)))

        #Starting on the second iteration, we will calculate the labor and capital unknowns at once in a system of equations.
        if iteration >= 1:
            ejs=ejs[(-length):, j]
            e1=ejs[:-1]
            e2=ejs[1:]
            #Labor Euler equation
            leq = lw1 * uprime(lguess * lw1*ejs + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
            #Capital Euler equation
            keq = uprime(kw1*kl1*e1 + (1 + kr1)*kk1 - kk2) - beta * (1 + kr2)*uprime(kw2*kl2*e2 + (1 + kr2) * kk2 - kk3)
            return np.append(solvec, np.max(np.abs(np.append(keq,leq))))


    #This function calculates the rest of the labor and capital matrices
    def calc_lower(klguess, wagevec, rentalvec, iteration,ejs):
        #Split up klguess
        kguess = klguess[:s-1]
        lguess = klguess[s-1:]

        length=len(lguess)
        ejs=ejs[(-length):, j]

        e1=ejs[:-1]
        e2=ejs[1:]

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
        leq = lw1 * uprime(lguess * lw1*ejs + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
        keq = uprime(kw1*kl1*e1 + (1 + kr1)*kk1 - kk2) - beta * (1 + kr2)*uprime(kw2*kl2*e2 + (1 + kr2) * kk2 - kk3)

        return np.append(keq, leq)

    def check_lower(klguess, wagevec, rentalvec, iteration,ejs):
        solvec=optimize.fsolve(calc_lower, klguess, args=(wagevec, rentalvec, iteration,ejs))
        kguess = solvec[:s-1]
        lguess = solvec[s-1:]
        
        length=len(lguess)
        ejs=ejs[(-length):, j]

        e1=ejs[:-1]
        e2=ejs[1:]

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
        leq = lw1 * uprime(lguess * lw1*ejs + (1 + lr1)*lk1 - lk2) - uprime(1 - lguess)
        keq = uprime(kw1*kl1*e1 + (1 + kr1)*kk1 - kk2) - beta * (1 + kr2)*uprime(kw2*kl2*e2 + (1 + kr2) * kk2 - kk3)
        return np.append(solvec, np.max(np.abs(np.append(keq,leq))))

    #Initialize iterations and distance between K and L guesses and updates.
    kdifference = 10**10
    ldifference = 10**10
    iters = 0
    too_high=False
    ind=np.arange(s)

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
        kmatrix = np.zeros((T+s, s-1,J))
        lmatrix = np.zeros((T+s, s,J))
        for j in xrange(J):
            print j
            toprowuse=np.reshape(toprow[:,:,j],(s-1,1))
            #Fill out matrices on and above the main diagonal
            for i in xrange(s-1):
                #tud = time until death
                tud = i+1
                lguess = lssmatuse[-(i + 1):,j]
                #On first iteration, solve only for the top right entry of the l matrix, which is unknown
                if i == 0:
                    new_l_vec = check_upper(lguess,wagevec, rentalvec, toprowuse, i, e_js[-1,j])
                    error=new_l_vec[-1]
                    if error>1e-5:
                        too_high=True
                        print "here is the error: {}".format(error)
                        break
                    new_l_vec=new_l_vec[:-1]
                    lmatrix[0,-1, j]= new_l_vec
                #Thereafter, solve for both capital and labor unknowns
                if i > 0:
                    indk=np.arange(i)
                    indl=np.arange(i+1)
                    kguess = kssmatuse[-(i):,j]
                    klguess = np.hstack((kguess, lguess))
                    new_kl_vec = check_upper(klguess,wagevec, rentalvec, toprowuse, i,e_js)
                    error=new_kl_vec[-1]
                    if error>1e-5:
                        too_high=True
                        print "here is the error and iteration: {} {}".format(error, i)
                        break
                    new_kl_vec=new_kl_vec[:-1]
                    new_k_vec = new_kl_vec[:tud-1]
                    new_l_vec = new_kl_vec[tud-1:]
                    kmatrix[indk, s-(i+1)+indk, j] = new_k_vec
                    lmatrix[indl, s-(i+1)+indl, j] = new_l_vec


            #Stack shocked individual capital steady state values on capital matrix begun above
            kmatrix = np.vstack([toprow, kmatrix])[:-1,:]
            # print "here are k and l matrices: {} \n {}".format(kmatrix[:s-1,:,0], lmatrix[:s,:,0])


            #Now fill out the matrix below the main diagonal
            for i in xrange(0,T):
                indk=np.arange(s-1)
                indl=np.arange(s)
                klguess = np.append(kssmatuse[:,j], lssmatuse[:,j])
                new_kl_vec = check_lower(klguess,wagevec, rentalvec, i,e_js)
                error=new_kl_vec[-1]
                if error>1e-5:
                    too_high=True
                    print "here is the error: {}".format(error)
                    break
                new_kl_vec=new_kl_vec[:-1]
                new_k_vec = new_kl_vec[:s-1]
                new_l_vec = new_kl_vec[s-1:]
                kmatrix[i+1+indk, indk, j] = new_k_vec
                lmatrix[i+indl, indl, j] = new_l_vec
        #K and l matrices completed at this point, now
        kmatrix = kmatrix[:T,:,:]#use for guesses
        lmatrix = lmatrix[:T,:,:]#use for guesses again
        placehold_L=lssmatuse
        placehold_K=kssmatuse
        lssmatuse=lmatrix[0,:,:].reshape(s,J)
        kssmatuse=kmatrix[0,:,:].reshape(s-1,J)
        lssmatuse=xi*lssmatuse+((1-xi)*placehold_L)
        kssmatuse=xi*kssmatuse+((1-xi)*placehold_K)
        lmatrix= e_js.reshape(1, s, J) * lmatrix
        if (kmatrix.any()==0) or (lmatrix.any()==0):
            print "empty spots exist!"

        #sum up columns of k and l,
        Updated_K = kmatrix.sum(2).sum(1)
        Updated_L = lmatrix.sum(2).sum(1)
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


    if too_high==True:
        print "There was an error that was too high!!!"
        
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





