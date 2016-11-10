from __future__ import division
import numpy as np
import scipy.optimize as opt
import math
import pandas as pd
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import math as math
import sys
sys.path.insert(0, 'C:\Users\Parker\Documents\MCL\parkermcl\Econ')
import RBCnumerderiv as RBC1
import RBCnumerderiv2 as RBC2
import uhlig as uhlig
#import UhligDeriv as UhligDeriv
from numpy import exp, array, zeros
from numpy import random as rand
import scipy.stats as stats

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize as opt
from matplotlib import pyplot as plt
import random as rand
import tabulate


'''
problem 8.1
'''
beta=.98
alpha=.35
zbar=0.
rho=.95
sigma=.02

Kbar= (beta* alpha)**(1/(1- alpha))
Rbar= alpha* Kbar**( alpha-1)
Cbar= Kbar** alpha - Kbar

F= alpha*Kbar**(alpha-1)/Cbar
G= - alpha*Kbar**(alpha-1) * (alpha + Kbar**(alpha-1) ) /Cbar
H= alpha**2 * Kbar**(2*alpha-2) / Cbar
L= - alpha* Kbar**(2*alpha-1) / Cbar
M= alpha**2 * Kbar**(2*alpha-2)/ Cbar
N= rho

P= ( -G + ( G**2- 4*F*H )**0.5 )/ 2*F #-2.67262262106
Q= (-L*N-M )/( F*N+F*P+G ) #-0.315305225874

#print P, Q

K = np.linspace(.5*Kbar,1.5*Kbar,26)
z = np.linspace(-5*sigma,5*sigma,26)

def Psi_lin(K, z):
    Psi= P* (K -Kbar) + Q*(z -zbar) + Kbar
    return Psi

K_prime_1= Psi_lin(K, z)

X, Y= np.meshgrid(K, z)
fig1 = plt.figure(1)
ax = fig1.gca(projection = '3d')
ax.set_xlabel("K")
ax.set_ylabel("z")
ax.set_zlabel("K'(K,z)")
plt.title('8.1 Policy Function Graph')
ax.plot_surface(X, Y, K_prime_1, rstride=5)
plt.show()
print "Problem 8.2"
'''
problem 8.2
'''

gamma=2.5 #not used
beta=.98
alpha=.35
delta=.10 #not used
zbar=0.
tau=.05 #not used
rho=.95
sigma=.02

Kbar= (beta* alpha)**(1/(1- alpha))
Rbar= alpha* Kbar**( alpha-1)
Cbar= Kbar** alpha - Kbar

F= Kbar**2/Cbar
G= ( alpha*Kbar**(alpha+1)-Kbar**(-alpha) )/Cbar + (alpha-1)*Kbar
H= alpha* Kbar**(1+alpha)/ Cbar
L= ( Kbar - Kbar**(alpha+1) )/ Cbar
M= Kbar**(alpha+1)/ Cbar
N= rho

P= ( -G + ( G**2- 4*F*H )**0.5 )/ 2*F
Q= (-L*N-M )/( F*N+F*P+G )

print P #0.4869957137
print Q #0.108734140615

K = np.linspace(.5*Kbar,1.5*Kbar,26)
z = np.linspace(-5*sigma,5*sigma,26)

def Psi(K, z):
	Psi= P* (K -Kbar) + Q*(z -zbar) *Kbar + Kbar
	return Psi

K_prime= Psi(K, z)
print K_prime
X, Y= np.meshgrid(K, z)
fig1 = plt.figure(1)
ax = fig1.gca(projection = '3d')
ax.set_xlabel("K")
ax.set_ylabel("z")
ax.set_zlabel("K'(K,z)")
plt.title('Policy Function')
ax.plot_surface(X, Y, K_prime, rstride=5)
plt.show()

'''
problem 8.3
'''

'''
problem 8.4
this is exactly like 6.5
'''

'''
problem 8.5
'''
print "Problem 8.5"

param=np.array([.1,.05, 0, .4, 2.5,1.5,.98,.5])

def state_defs6(kbar, lbar, parameters=param):
    delta, tau, zbar, alpha, gamma, xi, beta, a = parameters
    y = (kbar ** alpha) * (np.exp(zbar) * lbar )** (1 - alpha)
    i = kbar - (1 - delta) * kbar
    r = (alpha) * (lbar/kbar)**(1-alpha)
    w = (1 - alpha) * (kbar/lbar)**alpha
    T = tau * (w * lbar + (r - delta) * kbar)
    c = (1 - tau) * (w * lbar + (r-delta) * kbar) + T
 
    return np.array([y, i, c, r, w, T])
 
 
def opt_func6(guess, params=param):
    delta, tau, zbar, alpha, gamma, xi, beta, a = params
    kbar, lbar = guess
    y, i, c, r, w, T = state_defs6(kbar, lbar, params)
    eul_err1 = beta * (c ** (-gamma)*((r - delta) * (1 - tau) + 1)) - c**(-gamma)
    eul_err2 = w * c ** (-gamma)*(1-tau)-a*(1-lbar)**-xi

    return np.array([eul_err1, eul_err2])
 
 
initial_guess = np.array([1.0, .4])
ss = opt.fsolve(opt_func6, initial_guess, args=param)
k_ss, l_ss = ss
y_ss, i_ss, c_ss, r_ss, w_ss, T_ss = state_defs6(k_ss, l_ss)



original=np.array([ k_ss, c_ss, r_ss, w_ss, l_ss, T_ss, y_ss, i_ss ])
deriv=original
for i in range (len(param)):
	param[i]=param[i]+.00001
	ss = opt.fsolve(opt_func6, initial_guess, args=param)
	k_ss, l_ss = ss
	y_ss, i_ss, c_ss, r_ss, w_ss, T_ss = state_defs6(k_ss, l_ss)
	adjust=np.array([ k_ss, c_ss, r_ss, w_ss, l_ss, T_ss, y_ss, i_ss ])
	diff=(adjust-original)/.00001
	deriv=np.vstack((deriv,diff))
	param[i]=param[i]-.00001
deriv=np.delete(deriv,(0), axis=0)
deriv=deriv.round(decimals=3, out=None)
print np.shape(deriv)
deriv=np.append(np.reshape(np.array(['one','two','three','four','five','six','seven','eight']), (8,1)), deriv, 1)
headers=['one','two','three','four','five','six','seven','eight', 'nine']
print tabulate.tabulate(deriv, headers, tablefmt="latex")

print "problem 8.6"
'''
problem 8.6
'''
gamma = 2.5
xsi = 1.5
beta = 0.98
alpha = 0.40
a = 0.5
delta = 0.10
zbar = 0.0
tau = 0.05
rho = 0.9


print('Exercise 8.6')

nx=2
ny=0
nz=1

def state_defs(kt1, kt, lt, zt):

    y = kt**alpha*(lt*np.exp(zt))**(1-alpha)
    i = kt1-kt*(1-delta)
    r = alpha*kt**(alpha-1)*(lt*np.exp(zt))**(1-alpha)
    w = (1-alpha)*kt**alpha*(lt*np.exp(zt))**(-alpha)*np.exp(zt)
    T = tau*(w*lt+ (r-delta)*kt)
    c = (1-tau)*(w*lt + (r-delta)*kt)+kt +T-kt1
    return np.array([y, i, c, r, w, T])


def opt_func(guess, want_ss=0):
    if want_ss == 1:
    	kbar, lbar = guess[0:2]
    	y, i, c, r, w, T = state_defs(kbar, kbar, lbar, zbar)

    	Eul1 = c **(-gamma) - beta * c **(-gamma) * ((r - delta) * (1 - tau) + 1)
    	Eul2 = -a * (1 - lbar) ** (-xsi)+ c **(-gamma) * w * (1 - tau)
    
    else:
        kt2, lt1, kt1, lt, kt, lt_1, zt1, zt = guess

        y, i, c, r, w, T = state_defs(kt1, kt, lt, zt)
        y1, i1, c1, r1, w1, T1 = state_defs(kt2, kt1, lt1, zt1)

        Eul1 = c **(gamma) * beta * c1 **(-gamma) * ((r1 - delta) * (1 - tau) + 1) -1
        Eul2 = ( w * (1 - tau) * (1 - lt)** xsi * c**(-gamma) ) / a -1
    	
    return np.array([Eul1, Eul2])

# Solving for the steady state
initial_guess = np.array([3.5, 0.4])
Xbar  = kbar, lbar = opt.fsolve(opt_func, initial_guess, args=(1))

SS_1 = ybar, ibar, cbar, rbar, wbar, Tbar = state_defs(kbar, kbar, lbar, zbar)
SS = np.append(Xbar, SS_1)

# Solving for A_M matricies.
theta0 = np.array([kbar, lbar, kbar, lbar, kbar, lbar, zbar, zbar])
#theta0=np.array([k_ss, c_ss,r_ss, w_ss, l_ss, T_ss, y_ss, i_ss, zbar,zbar])

NN = np.zeros((nz,nz))
NN = NN+ rho

AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, TT, WW  = RBC2.numeric_deriv2(opt_func, theta0, nx, ny, nz) 

print 'Empty matrices are ', AA, BB, CC, DD, JJ, KK
print 'F is \n', FF, '\nG is \n', GG, '\nH is \n', HH, '\nL is \n', LL, '\nand M is \n', MM
PP, QQ, RR, SS = uhlig.solvePQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN)
print 'P is \n', PP, '\nand Q is \n', QQ


print('Exercise 8.7')

K_prime_7 = PP[0,0]* (K -kbar) + QQ[0]*(z -zbar) + kbar

X, Y= np.meshgrid(K, z)
fig2 = plt.figure(2)
ax = fig2.gca(projection = '3d')
ax.set_xlabel("K")
ax.set_ylabel("z")
ax.set_zlabel("K'(K,z)")
plt.title('8.7 Policy Function Graph')
ax.plot_surface(X, Y, K_prime_7, rstride=5)
plt.show()

print('Exercise 8.8')

sigma=.02

#generate one time-series
def time_series():
    # initialization
    k_period = np.zeros(250)
    k_period[0] = kbar 
    z_period = np.zeros(250)
    z_period[0] = zbar 
    l_period = np.zeros(250)
    l_period[0] = lbar
    k_period1 = np.zeros(250)

    e_period= np.random.normal(0, sigma, 250)
    for i in xrange(250): #250
        if i < 249:
            z_period[i+1]= rho * (z_period[i]) + e_period[i]
            k_period[i+1] = PP[0,0]* (kbar- k_period[i]) + QQ[0]*(z_period[i]) + kbar
            k_period1[i] = k_period[i+1]
            l_period[i+1] = PP[1,0]* (lbar- l_period[i]) + QQ[1]*(z_period[i]) + lbar
        else:
            k_period1[i] = PP[0,0]* (kbar- k_period[i]) + QQ[0]*(z_period[i]) + kbar

    return k_period, l_period, z_period, k_period1

#simulate and get mean

def simulation(n):
    # initialization
    y_mat= np.zeros((n, 250))
    c_mat= np.zeros((n, 250))
    i_mat= np.zeros((n, 250))
    l_mat= np.zeros((n, 250))
    
    for i in xrange(n):

        k_period, l_period, z_period, k_period1= time_series()

        l_mat[i,:] = l_period 
        y_mat[i,:] = k_period**alpha*(l_period*np.exp(z_period))**(1-alpha)
        i_mat[i,:] = k_period1-k_period*(1-delta)
        r = alpha*k_period**(alpha-1)*(l_period*np.exp(z_period))**(1-alpha)
        w = (1-alpha)*k_period**alpha*(l_period*np.exp(z_period))**(-alpha)*np.exp(z_period)
        T = tau*(w*l_period+ (r-delta)*k_period)   
        c_mat[i,:] = (1-tau)*(w*l_period + (r-delta)*k_period)+k_period +T-k_period1
        
    return y_mat, c_mat, i_mat, l_mat

#take means

y_mat, c_mat, i_mat, l_mat = simulation(500)

y_mean= np.mean(y_mat, axis=0)
c_mean= np.mean(c_mat, axis=0)
i_mean= np.mean(i_mat, axis=0)
l_mean= np.mean(l_mat, axis=0)

y_5= np.percentile(y_mat, 5, axis=0)
c_5= np.percentile(c_mat, 5, axis=0)
i_5= np.percentile(i_mat, 5, axis=0)
l_5= np.percentile(l_mat, 5, axis=0)

y_95= np.percentile(y_mat, 95, axis=0)
c_95= np.percentile(c_mat, 95, axis=0)
i_95= np.percentile(i_mat, 95, axis=0)
l_95= np.percentile(l_mat, 95, axis=0)

#graph
mean= np.array([y_mean,c_mean,i_mean,l_mean])
upper_inter= np.array([y_95,c_95,i_95,l_95])
lower_inter= np.array([y_5,c_5,i_5,l_5])
confi_inter= upper_inter - lower_inter

titiles=['GDP','Consumption','Investment','Labor Output']

period= np.linspace(1,250,250)
for i in xrange(4):
    plt.subplot(4,1,i+1)
    plt.title(titiles[i])
    plt.plot(period, mean[i], label='The Path')
    plt.plot(period, upper_inter[i], label ='Upper Confidence Band')
    plt.plot(period, lower_inter[i], label ='Lower Confidence Band')
    #plt.errorbar(period, mean[i], yerr=confi_inter[i])
plt.legend(loc='lower right')

plt.show()


'''
problem 8.8 and 8.9
'''
print "Exercises 8.8 and 8.9"

def generate_time_series(m):
	epsilon_vector = np.random.normal(0,sigma**2,(m,1))
	zpath = np.zeros((m))
	kpath = np.zeros((m))
	lpath = np.zeros((m))
	ypath = np.zeros((m))
	ipath = np.zeros((m))
	cpath = np.zeros((m))
	rpath = np.zeros((m))
	wpath = np.zeros((m))
	Tpath = np.zeros((m))

	zpath[0]=zbar
	kpath[0]=k_ss
	lpath[0]=l_ss
	ypath[0]=y_ss
	ipath[0]=i_ss
	cpath[0]=c_ss
	rpath[0]=r_ss
	wpath[0]=w_ss
	Tpath[0]=T_ss

	for i in xrange(1,m):
		zpath[i] = rho*zpath[i-1]+epsilon_vector[i]
		kpath[i]=k_ss+PP[0,0]*(kpath[i-1]-k_ss)+QQ[0]*zpath[i]
		lpath[i]=l_ss+PP[1,0]*(lpath[i-1]-l_ss)+QQ[1]*zpath[i]
		ypath[i],ipath[i],cpath[i],rpath[i],wpath[i],Tpath[i]=state_defs(kpath[i],kpath[i-1],lpath[i],zpath[i])

	return zpath,kpath,lpath,ypath,ipath,cpath,rpath,wpath,Tpath



def time_series_simulate(n,m):
	
	zmatrix = np.zeros((n,m))
	kmatrix = np.zeros((n,m))
	lmatrix = np.zeros((n,m))
	ymatrix = np.zeros((n,m))
	imatrix = np.zeros((n,m))
	cmatrix = np.zeros((n,m))
	rmatrix = np.zeros((n,m))
	wmatrix = np.zeros((n,m))
	Tmatrix = np.zeros((n,m))
	kcorr = np.zeros(n)
	ccorr = np.zeros(n)
	icorr = np.zeros(n)
	lcorr = np.zeros(n)
	ycorr = np.zeros(n)
	kauto = np.zeros(n)
	cauto = np.zeros(n)
	iauto = np.zeros(n)
	lauto = np.zeros(n)
	yauto = np.zeros(n)

	for i in xrange(0,n):
		zmatrix[i,:],kmatrix[i,:],lmatrix[i,:],ymatrix[i,:],imatrix[i,:],cmatrix[i,:],rmatrix[i,:],wmatrix[i,:],Tmatrix[i,:]=generate_time_series(m)
		kcorr[i],p=stats.pearsonr(kmatrix[i,:],ymatrix[i,:])
		kauto[i],p=stats.pearsonr(kmatrix[i,:-1],kmatrix[i,1:])
		ccorr[i],p=stats.pearsonr(cmatrix[i,:],ymatrix[i,:])
		cauto[i],p=stats.pearsonr(cmatrix[i,:-1],cmatrix[i,1:])
		icorr[i],p=stats.pearsonr(imatrix[i,:],ymatrix[i,:])
		iauto[i],p=stats.pearsonr(imatrix[i,:-1],imatrix[i,1:])
		lcorr[i],p=stats.pearsonr(lmatrix[i,:],ymatrix[i,:])
		lauto[i],p=stats.pearsonr(lmatrix[i,:-1],lmatrix[i,1:])
		ycorr[i],p=stats.pearsonr(ymatrix[i,:],ymatrix[i,:])
		yauto[i],p=stats.pearsonr(ymatrix[i,:-1],ymatrix[i,1:])

	lvector5=np.percentile(lmatrix, 5, axis = 0)
	lvector95=np.percentile(lmatrix, 95, axis = 0)
	lvector50=np.mean(lmatrix,axis=0)
	kvector5=np.percentile(kmatrix, 5, axis = 0)
	kvector95=np.percentile(kmatrix, 95, axis = 0)
	kvector50=np.mean(kmatrix,axis=0)
	yvector5=np.percentile(ymatrix, 5, axis = 0)
	yvector95=np.percentile(ymatrix, 95, axis = 0)
	yvector50=np.mean(ymatrix,axis=0)
	cvector5=np.percentile(cmatrix, 5, axis = 0)
	cvector95=np.percentile(cmatrix, 95, axis = 0)
	cvector50=np.mean(cmatrix,axis=0)
	ivector5=np.percentile(imatrix, 5, axis = 0)
	ivector95=np.percentile(imatrix, 95, axis = 0)
	ivector50=np.mean(imatrix,axis=0)

	ymean = np.mean(ymatrix, axis=1)
	ystd = np.std(ymatrix, axis = 1)
	ycv = ymean/ystd
	yrv = ystd/ystd

	lmean = np.mean(lmatrix, axis=1)
	lstd = np.std(lmatrix, axis = 1)
	lcv = lmean/lstd
	lrv = lstd/ystd

	kmean = np.mean(kmatrix, axis=1)
	kstd = np.std(kmatrix, axis = 1)
	kcv = kmean/kstd 
	krv = kstd/ystd

	cmean = np.mean(cmatrix, axis=1)
	cstd = np.std(cmatrix, axis = 1)
	ccv = cmean/cstd
	crv = cstd/ystd

	imean = np.mean(imatrix, axis=1)
	istd = np.std(imatrix, axis = 1)
	icv = imean/istd
	irv = imean/ystd

	average_value_vector = np.array([yvector50,cvector50,ivector50,lvector50,kvector50])
	lower_band_vector = np.array([yvector5,cvector5, ivector5, lvector5, kvector5])
	upper_band_vector = np.array([yvector95,cvector95,ivector95,lvector95,kvector95])

	y_stats = np.array([ymean,ystd,ycv,yrv,yauto,ycorr])
	c_stats = np.array([cmean,cstd,ccv,crv,cauto,ccorr])
	i_stats = np.array([imean,istd,icv,irv,iauto,icorr])
	l_stats = np.array([lmean,lstd,lcv,lrv,lauto,lcorr])
	k_stats = np.array([kmean,kstd,kcv,krv,kauto,kcorr])

	return average_value_vector,lower_band_vector,upper_band_vector,y_stats,c_stats,i_stats,l_stats,k_stats

#m-number of time periods
#n-number of simulations

m=250
n=100

average_value_vector,lower_band_vector,upper_band_vector,y_stats,c_stats,i_stats,l_stats,k_stats = time_series_simulate(n,m)
time_grid = np.linspace(1,m,m)
print y_ss
print np.mean(average_value_vector[0])
plt.plot(time_grid,average_value_vector[0], label="Mean of output path")
plt.plot(time_grid,lower_band_vector[0], label="Lower confidence band")
plt.plot(time_grid,upper_band_vector[0], label="Upper confidence band")
plt.legend(loc="lower right")
plt.show()
print c_stats

y_mean_mean = np.mean(y_stats[0])
y_mean_std = np.std(y_stats[0])/(n**(1/2))
y_std_mean = np.mean(y_stats[1])
y_std_std = np.std(y_stats[1])/(n**(1/2))
y_cv_mean = np.mean(y_stats[2])
y_cv_std = np.std(y_stats[2])/(n**(1/2))
y_rv_mean = np.mean(y_stats[3])
y_rv_std = np.std(y_stats[3])/(n**(1/2))
y_autocorr_mean = np.mean(y_stats[4])
y_autocorr_std = np.std(y_stats[4])/(n**(1/2))
y_corr_mean = np.mean(y_stats[5])
y_corr_std = np.std(y_stats[5])/(n**(1/2))


c_mean_mean = np.mean(c_stats[0])
c_mean_std = np.std(c_stats[0])/(n**(1/2))
c_std_mean = np.mean(c_stats[1])
c_std_std = np.std(c_stats[1])/(n**(1/2))
c_cv_mean = np.mean(c_stats[2])
c_cv_std = np.std(c_stats[2])/(n**(1/2))
c_rv_mean = np.mean(c_stats[3])
c_rv_std = np.std(c_stats[3])/(n**(1/2))
c_autocorr_mean = np.mean(c_stats[4])
c_autocorr_std = np.std(c_stats[4])/(n**(1/2))
c_corr_mean = np.mean(c_stats[5])
c_corr_std = np.std(c_stats[5])/(n**(1/2))

i_mean_mean = np.mean(i_stats[0])
i_mean_std = np.std(i_stats[0])/(n**(1/2))
i_std_mean = np.mean(i_stats[1])
i_std_std = np.std(i_stats[1])/(n**(1/2))
i_cv_mean = np.mean(i_stats[2])
i_cv_std = np.std(i_stats[2])/(n**(1/2))
i_rv_mean = np.mean(i_stats[3])
i_rv_std = np.std(i_stats[3])/(n**(1/2))
i_autocorr_mean = np.mean(i_stats[4])
i_autocorr_std = np.std(i_stats[4])/(n**(1/2))
i_corr_mean = np.mean(i_stats[5])
i_corr_std = np.std(i_stats[5])/(n**(1/2))


l_mean_mean = np.mean(l_stats[0])
l_mean_std = np.std(l_stats[0])/(n**(1/2))
l_std_mean = np.mean(l_stats[1])
l_std_std = np.std(l_stats[1])/(n**(1/2))
l_cv_mean = np.mean(l_stats[2])
l_cv_std = np.std(l_stats[2])/(n**(1/2))
l_rv_mean = np.mean(l_stats[3])
l_rv_std = np.std(l_stats[3])/(n**(1/2))
l_autocorr_mean = np.mean(l_stats[4])
l_autocorr_std = np.std(l_stats[4])/(n**(1/2))
l_corr_mean = np.mean(l_stats[5])
l_corr_std = np.std(l_stats[5])/(n**(1/2))

k_mean_mean = np.mean(k_stats[0])
k_mean_std = np.std(k_stats[0])/(n**(1/2))
k_std_mean = np.mean(k_stats[1])
k_std_std = np.std(k_stats[1])/(n**(1/2))
k_cv_mean = np.mean(k_stats[2])
k_cv_std = np.std(k_stats[2])/(n**(1/2))
k_rv_mean = np.mean(k_stats[3])
k_rv_std = np.std(k_stats[3])/(n**(1/2))
k_autocorr_mean = np.mean(k_stats[4])
k_autocorr_std = np.std(k_stats[4])/(n**(1/2))
k_corr_mean = np.mean(k_stats[5])
k_corr_std = np.std(k_stats[5])/(n**(1/2))

'''
8.10
'''

'''
8.11
'''

print "Exercise 4.1"

def ss_equations(klist,parameters):
	
	k2 = klist[0]
	k3 = klist[1]
	beta = parameters[0]
	delta = parameters[1]
	gamma = parameters[2]
	A = parameters[3]
	alpha = parameters[4]
	L = 2
	K = k2 + k3
	wt = (1-alpha)*A*(K/L)**alpha
	rt = alpha*A*(L/K)**(1-alpha)
	lhs1 = (wt+(1+rt-delta)*k2-k3)**(-gamma)
	rhs1 = beta*(1+rt-delta)*((1+rt-delta)*k3)**(-gamma)
	lhs2 = (wt-k2)**(-gamma)
	rhs2 = beta*(1+rt-delta)*(wt+(1+rt-delta)*k2-k3)**(-gamma)

	difference1 = lhs1 - rhs1
	difference2 = lhs2 - rhs2
	difference = np.array([difference1,difference2])
	return difference

parameters = np.array([.96**20,1 - (1 - 0.05)**20,3,1,.35])
print parameters

beta = parameters[0]
delta = parameters[1]
gamma = parameters[2]
A = parameters[3]
alpha = parameters[4]


steady_state = opt.fsolve(ss_equations,[.1,.1], args=(parameters), xtol = .00000001)

k2_steady = steady_state[0]
k3_steady = steady_state[1]
Kss = k2_steady+k3_steady
L = 2
wss = (1-alpha)*A*(Kss/L)**(alpha)
rss = alpha*A*(L/Kss)**(1-alpha)
c1ss = wss - k2_steady
c2ss = wss-k3_steady+(1+rss-delta)*k2_steady
c3ss = (1+rss-delta)*k3_steady

print "c1 steady state: ", c1ss
print "c2 steady state: ", c2ss
print "c3 steady state: ", c3ss
print "k2 steady state: ", k2_steady
print "k3 steady state: ", k3_steady
print "w steady state: ", wss
print "r rate steady state: ", rss

print "Exercise 4.2"
parameters = np.array([.55,1 - (1 - 0.05)**20,3,1,.35])
beta = parameters[0]
delta = parameters[1]
gamma = parameters[2]
A = parameters[3]
alpha = parameters[4]

steady_state = opt.fsolve(ss_equations,[.1,.1], args=(parameters), xtol = .00000001)

k2_steady = steady_state[0]
k3_steady = steady_state[1]
Kss = k2_steady+k3_steady
L = 2
wss = (1-alpha)*A*(Kss/L)**(alpha)
rss = alpha*A*(L/Kss)**(1-alpha)
c1ss = wss - k2_steady
c2ss = wss-k3_steady+(1+rss-delta)*k2_steady
c3ss = (1+rss-delta)*k3_steady

print "c1 steady state (beta = .55): ", c1ss
print "c2 steady state (beta = .55): ", c2ss
print "c3 steady state (beta = .55): ", c3ss
print "k2 steady state (beta = .55): ", k2_steady
print "k3 steady state (beta = .55): ", k3_steady
print "w steady state (beta = .55): ", wss
print "r rate steady state (beta = .55): ", rss
print "\nIncreasing beta caused all savings to go up. "
print "It also increased the wage and decreased the interest rate. This makes "
print "sense because increasing beta decreases the discount rate "
print "(or the 'impatience' parameter) and therefore consumption would be "
print "expected to go up."
print "\nExercise 4.3"

T=25
'''
re-initializing the appropriate parameters
'''
parameters = np.array([.96**20,1 - (1 - 0.05)**20,3,1,.35])
steady_state = opt.fsolve(ss_equations,[.1,.1], args=(parameters), xtol = .00000001)
k2_steady = steady_state[0]
k3_steady = steady_state[1]
Kss = k2_steady+k3_steady
L = 2
wss = (1-alpha)*A*(Kss/L)**(alpha)
rss = alpha*A*(L/Kss)**(1-alpha)
c1ss = wss - k2_steady
c2ss = wss-k3_steady+(1+rss-delta)*k2_steady
c3ss = (1+rss-delta)*k3_steady





#pathguess = np.array([k3guess,k2guess,wguess,rguess])

'''
'''


"""
Begin ch. 8 stuff here!!


"""

print "8.11"

parameters = np.array([.96**20,1 - (1 - 0.05)**20,3,1,.35])


beta = parameters[0]
delta = parameters[1]
gamma = parameters[2]
A = parameters[3]
alpha = parameters[4]

def state_defs(k2t,k3t,k2t1,k3t1,zt):
	Kt=k2t + k3t   
	rt = alpha*Kt**(alpha-1)*2**(1-alpha)*exp(zt)
	wt = (1-alpha)*Kt**(alpha)*2**(-alpha)*exp(zt)
	c1t = wt - k2t1
	c2t = wt+(1+rt-delta)*k2t-k3t1
	c3t = (1+rt-delta)*k3t
	return rt,wt, c1t, c2t, c3t

def opt_func(guess):
 
	#k3t2, k2t2, k3t1, k2t1, k3t, k2t, zt1, zt = guess
	k2t2,k3t2,k2t1,k3t1,k2t,k3t,zt1,zt = guess 

	rt, wt, c1t, c2t, c3t = state_defs(k2t,k3t,k2t1,k3t1,zt)
	rt1, wt1, c1t1, c2t1, c3t1 = state_defs(k2t1,k3t1,k2t2,k3t2,zt1)

	Eul1 = c1t**(-1) - beta*c2t1**(-1)*(1+rt1-delta) 
	Eul2 = c2t**(-1) - beta*c3t1**(-1)*(1+rt1-delta) 
	
	return np.array([Eul1,Eul2])
	
rho = .9
nx=2
ny=0
nz=1
NN = np.zeros((nz,nz))
NN = NN + rho
zbar = 0
#theta0 = np.array([k3_steady,k2_steady,k3_steady,k2_steady,k3_steady,k2_steady,zbar,zbar])
theta0 = np.array([k2_steady,k3_steady,k2_steady,k3_steady,k2_steady,k3_steady,zbar,zbar])
AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, TT, WW  = RBC2.numeric_deriv2(opt_func, theta0, nx, ny, nz)

PP, QQ, RR, SS = uhlig.solvePQRS(AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, NN)
print "P", PP
print "Q", QQ

k2_ss = k2_steady
k3_ss = k3_steady
sigma = .02


def generate_time_series(m, shock = 0):
	epsilon_vector = np.random.normal(0,sigma**2,(m,1))
	
	zpath = np.zeros((m))
	Kpath = np.zeros((m))
	k2path = np.zeros((m))
	k3path = np.zeros((m))
	ypath = np.zeros((m))
	ipath = np.zeros((m))
	c1path = np.zeros((m))
	c2path = np.zeros((m))
	c3path = np.zeros((m))
	rpath = np.zeros((m))
	wpath = np.zeros((m))
	
	zpath[0]=zbar
	Kpath[0]=.8*k2_ss+1.1*k3_ss
	k2path[0]=.8*k2_ss
	k3path[0]=1.1*k3_ss
	rpath[0]=alpha*A*(L/Kpath[0])**(1-alpha)
	wpath[0]=(1-alpha)*A*(Kpath[0]/L)**(alpha)
	c1path[0]=wpath[0] - k2path[0]
	c2path[0]=wpath[0]-k3path[0]+(1+rss-delta)*k2path[0]
	c3path[0]=(1+rpath[0]-delta)*k3path[0]
	ipath[0]=Kpath[0]


	if shock==0:

		for i in xrange(1,m):
			k2path[i]=k2_ss+PP[0,0]*(k2path[i-1]-k2_ss)+PP[0,1]*(k3path[i-1]-k3_ss)+QQ[0]*zpath[i]
			k3path[i]=k3_ss+PP[1,0]*(k3path[i-1]-k3_ss)+PP[1,1]*(k2path[i-1]-k2_ss)+QQ[1]*zpath[i]
			Kpath[i]=k2path[i]+k3path[i]
			ipath[i] = Kpath[i]-Kpath[i-1]*(1-delta)
			rpath[i],wpath[i],c1path[i],c2path[i],c3path[i]=state_defs(k2path[i-1],k3path[i-1],k2path[i],k3path[i],zpath[i])

	else:
		
		for i in xrange(1,m):
			zpath[i] = (rho)*zpath[i-1]+epsilon_vector[i]
			k2path[i]=k2_ss+PP[0,0]*(k2path[i-1]-k2_ss)+PP[0,1]*(k3path[i-1]-k3_ss)+QQ[0]*zpath[i]
			k3path[i]=k3_ss+PP[1,0]*(k3path[i-1]-k3_ss)+PP[1,1]*(k2path[i-1]-k2_ss)+QQ[1]*zpath[i]
			Kpath[i]=k2path[i]+k3path[i]
			ipath[i] = Kpath[i]-Kpath[i-1]*(1-delta)
			rpath[i],wpath[i],c1path[i],c2path[i],c3path[i]=state_defs(k2path[i-1],k3path[i-1],k2path[i],k3path[i],zpath[i])
	
	
	ypath = A*Kpath**(alpha)*2**(1-alpha)
	return Kpath,k2path, k3path, ypath, ipath, rpath,wpath,c1path,c2path,c3path,zpath

Kpath,k2path, k3path, ypath, ipath, rpath,wpath,c1path,c2path,c3path,zpath=generate_time_series(25)

time_grid = np.linspace(0,25,25)

plt.plot(Kpath)
plt.title("Path of aggregate capital stock")
plt.show()
plt.plot(k2path)
plt.title("Path of capital stock for middle-aged")
plt.show()
plt.plot(k3path)
plt.title("Path of capital stock for elderly")
plt.show()

print "8.12"

def time_series_simulate(n,m):
	
	
	Kmatrix = np.zeros((n,m))
	k2matrix = np.zeros((n,m))
	k3matrix = np.zeros((n,m))
	ymatrix = np.zeros((n,m))
	imatrix = np.zeros((n,m))
	c1matrix = np.zeros((n,m))
	c2matrix = np.zeros((n,m))
	c3matrix = np.zeros((n,m))
	Cmatrix = np.zeros((n,m))
	rmatrix = np.zeros((n,m))
	wmatrix = np.zeros((n,m))
	zmatrix = np.zeros((n,m))

	for i in xrange(0,n):
		Kmatrix[i,:],k2matrix[i,:], k3matrix[i,:], ymatrix[i,:], imatrix[i,:], rmatrix[i,:],wmatrix[i,:],c1matrix[i,:],c2matrix[i,:],c3matrix[i,:],zmatrix[i,:]=generate_time_series(m,1)
	
	Cmatrix=c1matrix+c2matrix+c3matrix 

	Kvector5=np.percentile(Kmatrix, 5, axis = 0)
	Kvector95=np.percentile(Kmatrix, 95, axis = 0)
	Kvector50=np.mean(Kmatrix,axis=0)
	k2vector5=np.percentile(k2matrix, 5, axis = 0)
	k2vector95=np.percentile(k2matrix, 95, axis = 0)
	k2vector50=np.mean(k2matrix,axis=0)
	k3vector5=np.percentile(k3matrix, 5, axis = 0)
	k3vector95=np.percentile(k3matrix, 95, axis = 0)
	k3vector50=np.mean(k3matrix,axis=0)
	yvector5=np.percentile(ymatrix, 5, axis = 0)
	yvector95=np.percentile(ymatrix, 95, axis = 0)
	yvector50=np.mean(ymatrix,axis=0)
	c1vector5=np.percentile(c1matrix, 5, axis = 0)
	c1vector95=np.percentile(c1matrix, 95, axis = 0)
	c1vector50=np.mean(c1matrix,axis=0)
	c2vector5=np.percentile(c2matrix, 5, axis = 0)
	c2vector95=np.percentile(c2matrix, 95, axis = 0)
	c2vector50=np.mean(c2matrix,axis=0)
	c3vector5=np.percentile(c2matrix, 5, axis = 0)
	c3vector95=np.percentile(c2matrix, 95, axis = 0)
	c3vector50=np.mean(c2matrix,axis=0)
	Cvector5=np.percentile(Cmatrix, 5, axis = 0)
	Cvector95=np.percentile(Cmatrix, 95, axis = 0)
	Cvector50=np.mean(Cmatrix,axis=0)
	ivector5=np.percentile(imatrix, 5, axis = 0)
	ivector95=np.percentile(imatrix, 95, axis = 0)
	ivector50=np.mean(imatrix,axis=0)

	average_value_vector = np.array([yvector50,c1vector50,c2vector50,c3vector50, Cvector50, ivector50, k2vector50,k3vector50,Kvector50])
	lower_band_vector = np.array([yvector5,c1vector5,c2vector5,c3vector5, Cvector5, ivector5, k2vector5,k3vector5,Kvector5])
	upper_band_vector = np.array([yvector95,c1vector95,c2vector95,c3vector95, Cvector95, ivector95, k2vector95,k3vector95,Kvector95])

	return average_value_vector,lower_band_vector,upper_band_vector

n=100
m=25
average_value_vector,lower_band_vector,upper_band_vector=time_series_simulate(n,m)

plt.plot(average_value_vector[0], label="Mean of output path")
plt.plot(lower_band_vector[0], label="Lower confidence band")
plt.plot(upper_band_vector[0], label="Upper confidence band")
plt.title("Output Path With Random Shock")
plt.legend(loc="lower right")
plt.show()

plt.plot(average_value_vector[4], label="Mean of consumption path")
plt.plot(lower_band_vector[4], label="Lower confidence band")
plt.plot(upper_band_vector[4], label="Upper confidence band")
plt.title("Consumption Path With Random Shock")
plt.legend(loc="lower right")
plt.show()

plt.plot(average_value_vector[5], label="Mean of investment path")
plt.plot(lower_band_vector[5], label="Lower confidence band")
plt.plot(upper_band_vector[5], label="Upper confidence band")
plt.title("Investment Path With Random Shock")
plt.legend(loc="upper right")
plt.show()