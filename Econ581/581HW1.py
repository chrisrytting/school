import numpy as np
from scipy.stats.stats import pearsonr   
from matplotlib import pyplot as plt
import random
import scipy as sp

alpha, delta, s, n, a, rho, sigma = .33, .02, .05, .0025, .005, .9, .02
params1 = alpha, delta, s, n, a, rho, sigma

##SKETCHY
'''
def generateseries(params):
    alpha, delta, s, n, a, rho, sigma = params 
    z = np.zeros(250)
    z[0] = 0
    k_tilde = np.zeros(250)
    k_tilde[0] = 1
    A_tilde = np.zeros(250)
    A_tilde[0] = 1
    H_tilde = np.zeros(250)
    H_tilde[0] = 1
    y_tilde = np.zeros(250)
    epsilon = np.random.normal(0, sigma, 250)

    for i in xrange(1,250):
        z[i] = rho*z[i-1] + epsilon[i] 
        k_tilde[i] = ((1-delta)*k_tilde[i-1] + s*k_tilde[i-1]**alpha*np.exp((1-alpha)*z[i-1]))/np.exp(n+a)
        A_tilde[i] = (np.exp(a+z[i-1])*A_tilde[i-1])
        ###Also could be np.exp(z)
        H_tilde[i] = (np.exp(n)*H_tilde[i-1])
    y_tilde = k_tilde**alpha
    i_tilde = y_tilde * s
    c_tilde = y_tilde * (1-s)
    return z, k_tilde, A_tilde, H_tilde, y_tilde, epsilon, i_tilde, c_tilde

def sumstats(seriesname, series, ytilde, Atilde):
    series_mean = np.mean(series)
    series_stdv = np.std(series)
    series_coeffvar = series_mean/series_stdv
    relativecoeffvar = np.mean(ytilde)/np.std(ytilde)
    corr_y_tilde, pval = pearsonr(series, ytilde)
    corr_A_tilde, pval = pearsonr(series, Atilde)
    autocorr, pval = pearsonr(series[:-1],series[1:])
    print '\nFor {}, Mean = {}\n Standard Deviation {}= \n Coefficient of variation = {} \n Relative coefficient of variation = {} \n Correlation with y-tilde = {} \n Correlation with A-tilde = {}\n'.format(seriesname, series_mean, series_stdv, series_coeffvar, relativecoeffvar, corr_y_tilde, corr_A_tilde, autocorr)

z, k_tilde, A_tilde, H_tilde, y_tilde, epsilon, i_tilde, c_tilde = generateseries(params1)
sumstats('k-tilde',k_tilde,y_tilde,A_tilde)
sumstats('y-tilde',y_tilde,y_tilde,A_tilde)
sumstats('c-tilde',c_tilde,y_tilde,A_tilde)
sumstats('i-tilde',i_tilde,y_tilde,A_tilde)
sumstats('A-tilde',A_tilde,y_tilde,A_tilde)
    
sumstats('ln(k-tilde)',np.log(k_tilde),np.log(y_tilde),np.log(A_tilde))
sumstats('ln(y-tilde)',np.log(y_tilde),np.log(y_tilde),np.log(A_tilde))
sumstats('ln(c-tilde)',np.log(c_tilde),np.log(y_tilde),np.log(A_tilde))
sumstats('ln(i-tilde)',np.log(i_tilde),np.log(y_tilde),np.log(A_tilde))
sumstats('z', z,np.log(y_tilde),np.log(A_tilde))

def MCsumstats(series, ytilde, Atilde):
    series_mean = np.mean(series)
    series_stdv = np.std(series)
    series_coeffvar = series_mean/series_stdv
    relativecoeffvar = np.mean(ytilde)/np.std(ytilde)
    corr_y_tilde, pval = pearsonr(series, ytilde)
    corr_A_tilde, pval = pearsonr(series, Atilde)
    autocorr, pval = pearsonr(series[:-1],series[1:])
    return series_mean, series_stdv, series_coeffvar, relativecoeffvar, corr_y_tilde, corr_A_tilde, autocorr

def MCseriesstats(params):
    summarystats = np.zeros((7,5))
    for i in xrange(1000):
        z, k_tilde, A_tilde, H_tilde, y_tilde, epsilon, i_tilde, c_tilde = generateseries(params)
        series = [k_tilde, y_tilde, c_tilde, i_tilde, A_tilde]
        for j in xrange(5):
            series_mean, series_stdv, series_coeffvar, relativecoeffvar, corr_y_tilde, corr_A_tilde, autocorr = MCsumstats(series[j], y_tilde, A_tilde)
            summarystats[0,j] += series_mean
            summarystats[1,j] += series_stdv
            summarystats[2,j] += series_coeffvar
            summarystats[3,j] += relativecoeffvar
            summarystats[4,j] += corr_y_tilde
            summarystats[5,j] += corr_A_tilde
            summarystats[6,j] += autocorr
    return summarystats / 1000

alpha, delta, s, n, a, rho, sigma = .33, .02, .05, .0025, .005, .9, .078
params1 = alpha, delta, s, n, a, rho, sigma
summarystats1 = MCseriesstats(params1)
print '\nRound 1: ',np.round(summarystats1, decimals = 3)
    
alpha, delta, s, n, a, rho, sigma = .33, .02, .20, .0025, .005, .9, .078
params2 = alpha, delta, s, n, a, rho, sigma
summarystats2 = MCseriesstats(params2)
print '\nRound 2: ',np.round(summarystats2, decimals = 3)

alpha, delta, s, n, a, rho, sigma = .33, .02, .05, .0025, .005, 0.0, .078
params3 = alpha, delta, s, n, a, rho, sigma
summarystats3 = MCseriesstats(params3)
print '\nRound 3: ',np.round(summarystats3, decimals = 3)

'''
def RCK(params, shock):
    a, n, delta, alpha, gamma, beta = params 
    H0,A0 = 1, 1
    params = a, n, delta, alpha, gamma, beta
    kbar = ((delta + (((np.exp(a))**gamma)/beta) - 1)/alpha)**(1/(alpha-1))
    ybar = kbar**alpha 
    cbar = -kbar * np.exp(a+n) + kbar**alpha + kbar*(1-delta)
    rbar = alpha * kbar**(alpha - 1)
    wbar = (1-alpha)*kbar**alpha
    T = 40
    c0guess = .3*cbar
    print kbar, ybar, cbar, rbar, wbar

    def findc(cguess):
        k0 = (shock)*kbar
        ktilde = np.zeros(T)
        ktilde[0] = k0
        rtilde = np.zeros(T)
        rtilde[0] = alpha*ktilde[0]**(alpha-1)
        dtilde = np.ones(T)
        dtilde[0] = np.exp(a)/(1+rtilde[0] - delta)
        ctilde = np.zeros(T)
        ctilde[0] = cguess
        wtilde = np.zeros(T)
        wtilde[0] = (1-alpha)*ktilde[0]**(alpha)
        ytilde = np.zeros(T)
        ytilde[0] = ktilde[0]**alpha
        for i in xrange(1, T):
            ktilde[i] = (((1-delta) * ktilde[i-1] + ktilde[i-1]**alpha - ctilde[i-1])/(np.exp(n+a)))
            rtilde[i] = alpha*(1/ktilde[i])**(1-alpha)
            wtilde[i] = (1-alpha)*ktilde[i]**(alpha)
            ytilde[i] = ktilde[i]**alpha
            for j in xrange(0,i):
                dtilde[i] *= (np.exp(a)/(1+rtilde[j] - delta))
            ctilde[3] = (beta*(1+rtilde[i] - delta))**(1/gamma)*np.exp(-a)*ctilde[i-1]
            c0guess = ctilde[1]
        return np.abs(ktilde[-1] - kbar)
    def bestcarrays(cguess):
        k0 = .7*kbar
        ktilde = np.zeros(T)
        ktilde[0] = k0
        rtilde = np.zeros(T)
        rtilde[0] = alpha*ktilde[0]**(alpha-1)
        dtilde = np.ones(T)
        dtilde[0] = np.exp(a)/(1+rtilde[0] - delta)
        ctilde = np.zeros(T)
        ctilde[0] = cguess
        wtilde = np.zeros(T)
        wtilde[0] = (1-alpha)*ktilde[0]**(alpha)
        ytilde = np.zeros(T)
        ytilde[0] = ktilde[0]**alpha
        for i in xrange(1, T):
            ktilde[i] = (((1-delta) * ktilde[i-1] + ktilde[i-1]**alpha - ctilde[i-1])/(np.exp(n+a)))
            rtilde[i] = alpha*(1/ktilde[i])**(1-alpha)
            wtilde[i] = (1-alpha)*ktilde[i]**(alpha)
            ytilde[i] = ktilde[i]**alpha
            for j in xrange(0,i):
                dtilde[i] *= (np.exp(a)/(1+rtilde[j] - delta))
            ctilde[i] = (beta*(1+rtilde[i] - delta))**(1/gamma)*np.exp(-a)*ctilde[i-1]
            c0guess = ctilde[1]
        return ktilde, ytilde, ctilde, wtilde, rtilde
    bestcguess = sp.optimize.fsolve(findc, c0guess)
    ktilde, ytilde, ctilde, wtilde, rtilde = bestcarrays(bestcguess)

    '''
    t=np.linspace(1,T,T)
    plt.subplot(231)
    plt.scatter(t,ktilde)
    plt.title('Capital')
    plt.xlim(0,40)

    plt.subplot(232)
    plt.scatter(t,ytilde)
    plt.title('Output')
    plt.xlim(0,40)

    plt.subplot(233)
    plt.scatter(t,ctilde)
    plt.title('Consumption')
    plt.xlim(0,40)

    plt.subplot(234)
    plt.scatter(t,wtilde)
    plt.title('Wages')
    plt.xlim(0,40)

    plt.subplot(235)
    plt.scatter(t,rtilde)
    plt.title('Interest Rate')
    plt.xlim(0,40)

    plt.subplot(236)
    plt.scatter(ktilde,ctilde)
    plt.title('Saddle path')
    plt.show()
    '''
                







params1 = .005, .0025, .02, .33, 2.5, .995
shock = .7
RCK(params1, shock)

params1 = .005, .0025, .02, .33, 2.5, .995
shock = 1.5
RCK(params1, shock)

params2 = .005, .0025, .02, .33, 1.0, .995
shock = .7
RCK(params2, shock)

params2 = .005, .0025, .02, .33, 1.0, .995
shock = 1.5
RCK(params2, shock)

params3 = .005, .0025, .02, .33, 2.5, .95
shock = .7
RCK(params3, shock)

params3 = .005, .0025, .02, .33, 2.5, .95
shock = 1.5
RCK(params3, shock)
