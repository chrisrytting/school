'''
------------------------------------------------------------------------
Last updated 8/5/2015

This program runs the steady state solver as well as the time path
iteration solution for the model with S-period lived agents and
exogenous labor from Chapter 4 of the OG textbook.

This Python script calls the following other file(s) with the associated
functions:
    Sec2ssfuncs.py
        feasible
        get_L
        get_K
        get_Y
        get_C
        get_r
        get_w
        get_cvec_ss
        get_b_n_errors
        EulerSys
        SS
    Sec2tpfuncs.py
        TPI
------------------------------------------------------------------------
'''
# Import packages

import numpy as np
import Chap4ssfuncs as c4ssf
reload(c4ssf)
import Chap4tpfuncs as c4tpf
reload(c4tpf)
import sys
import matplotlib.pyplot as plt

'''
------------------------------------------------------------------------
Declare parameters
------------------------------------------------------------------------
S            = integer in [3,80], number of periods an individual lives
T            = integer > S, number of time periods until steady state
beta_annual  = scalar in [0,1), discount factor for one year
beta         = scalar in [0,1), discount factor for each model period
sigma        = scalar > 0, coefficient of relative risk aversion
L            = scalar > 0, exogenous aggregate labor
A            = scalar > 0, total factor productivity parameter in firms'
               production function
alpha        = scalar in (0,1), capital share of income
delta_annual = scalar in [0,1], one-year depreciation rate of capital
delta        = scalar in [0,1], model-period depreciation rate of
               capital
SS_tol       = scalar > 0, tolerance level for steady-state fsolve
SS_graphs    = boolean, =True if want graphs of steady-state objects
TPI_solve    = boolean, =True if want to solve TPI after solving SS
TPI_tol      = scalar > 0, tolerance level for fsolve's in TPI
maxiter_TPI  = integer >= 1, Maximum number of iterations for TPI
mindist_TPI  = scalar > 0, Convergence criterion for TPI
xi           = scalar in (0,1], TPI path updating parameter
TPI_graphs   = boolean, =True if want graphs of TPI objects
------------------------------------------------------------------------
'''
# Household parameters
S = int(80)
T = int(round(3. * S))
beta_annual = .96
beta = beta_annual ** (80 / S)
sigma = 3.0
# Firm parameters
A = 1.0
alpha = .35
delta_annual = .05
delta = 1 - ((1-delta_annual) ** (80 / S))
# SS parameters
SS_tol = 1e-13
SS_graphs = False
# TPI parameters
TPI_solve = False
TPI_tol = 1e-13
maxiter_TPI = 200
mindist_TPI = 1e-13
xi = 0.20
TPI_graphs = False

listolegend=['0-24%','25-49%', '50-69%', '70-79%', '80-89%', '90-98%','99-100%']
ability_types=np.loadtxt('e_js.txt', delimiter=',')
s=np.linspace(0,80,80)
fig, ax  = plt.subplots()
'''
for i in xrange(7):
    plt.plot(s, ability_types[:,i], label=listolegend[i])
legend=ax.legend(loc= "upper right", shadow =True, title='ability types')
plt.xlabel('periods (s)')
plt.ylabel('ability levels')
plt.title('ability levels over lifetime')
plt.show()

'''
e_1 = ability_types[:, 0]
e_2 = ability_types[:, 1]
e_3 = ability_types[:, 2]
e_4 = ability_types[:, 3]
e_5 = ability_types[:, 4]
e_6 = ability_types[:, 5]
e_7 = ability_types[:, 6]

'''
------------------------------------------------------------------------
Compute the steady state
------------------------------------------------------------------------
b_guess       = [S-1,] vector, initial guess for steady-state
                distribution of savings
n_guess       = [S,] vector, initial guess for steady-state
                distribution of labor
feas_params   = tuple of length 5, parameters for feasible function
                [S, A, alpha, delta, L]
GoodGuess     = boolean, =True if initial steady-state guess is feasible
K_constr_init = boolean, =True if K<=0 for initial guess b_guess
c_constr_init = [S,] boolean vector, =True if c<=0 for initial b_guess
ss_params     = length 8 tuple, parameters to be passed in to SS
                function: [S, beta, sigma, A, alpha, delta, L, SS_tol]
b_ss          = [S-1,] vector, steady-state distribution of savings
c_ss          = [S,] vector, steady-state distribution of consumption
w_ss          = scalar > 0, steady-state real wage
r_ss          = scalar > 0, steady-state real interest rate
K_ss          = scalar > 0, steady-state aggregate capital stock
Y_ss          = scalar > 0, steady-state aggregate output (GDP)
C_ss          = scalar > 0, steady-state aggregate consumption
EulErr_ss     = [S-1,] vector, steady-state Euler errors
ss_time       = scalar, number of seconds to compute SS solution
rcdiff_ss     = scalar, steady-state difference in goods market clearing
                (resource constraint)
------------------------------------------------------------------------
'''
# Make initial guess of the steady-state
b_guess = np.zeros(S-1)
b_guess[:int(round(2 * S / 3))] = (np.linspace(0.003, 0.3,
                                   int(round(2 * S / 3))))
b_guess[int(round(2 * S / 3)):] = (np.linspace(0.3, 0.003,
                                   S - 1 - int(round(2 * S / 3))))
print np.shape(b_guess)
b_guess_matrix = np.tile(b_guess.T, (1,1,7)) 
print np.shape(b_guess_matrix)
sys.exit()
print b_guess.reshape(79,T,7)
n_guess = np.linspace(.9,.5,S)
n_guess_1 = np.copy(n_guess)*e_1
n_guess_2 = np.copy(n_guess)*e_2
n_guess_3 = np.copy(n_guess)*e_3
n_guess_4 = np.copy(n_guess)*e_4
n_guess_5 = np.copy(n_guess)*e_5
n_guess_6 = np.copy(n_guess)*e_6
n_guess_7 = np.copy(n_guess)*e_7
n_guesses = (n_guess_1,n_guess_2,n_guess_3,n_guess_4,n_guess_5,n_guess_6,n_guess_7)
n_guess_matrix = np.reshape(np.vstack(n_guesses).T, (80,1,7))

# Make sure initial guess is feasible
feas_params = (S, A, alpha, delta)
GoodGuess, K_constr_init, L_constr_init, c_constr_init = c4ssf.feasible(feas_params, b_guess, n_guesses)
if K_constr_init == True and L_constr_init.max() == True:
    print 'Initial guess is not feasible because K<=0 and n < 0 at some point. Some element(s) of bvec must increase and some element(s) of nvec must increase.'
elif K_constr_init == True and L_constr_init.max() == False:
    print 'Initial guess is not feasible because K<=0. Some element(s) of bvec must increase.'
elif K_constr_init == False and L_constr_init.max() == True:
    print 'Initial guess is not feasible because some elements of n <= 0 at some point. Some element(s) of nvec must increase.'
elif GoodGuess == True and c_constr_init.max() == True:
    print 'Initial guess is not feasible because some element of c<=0. However, K > 0 and nvec is positive for all elements.'
elif GoodGuess == True and c_constr_init.max() == False:
    print 'Initial guess is feasible.'

    # Compute steady state
    print 'BEGIN STEADY STATE COMPUTATION'
    ss_params = (S, beta, sigma, A, alpha, delta, SS_tol)
    b_ss, n_ss, c_ss, w_ss, r_ss, K_ss, L_ss, Y_ss, C_ss, EulErr_ss, ss_time = \
        c4ssf.SS(ss_params, b_guess, n_guess, SS_graphs)

    # Print diagnostics
    print 'The maximum absolute steady-state Euler error is: ', np.absolute(EulErr_ss).max()
    print 'The steady-state distribution of capital is:'
    print b_ss
    print 'The steady-state distribution of labor is:'
    print n_ss
    print 'The steady-state distribution of consumption is:'
    print c_ss
    print 'The steady-state wage and interest rate are:'
    print np.array([w_ss, r_ss])
    print 'Aggregate output, capital stock and consumption are:'
    print np.array([Y_ss, K_ss, C_ss])
    rcdiff_ss = Y_ss - C_ss - delta * K_ss
    print 'The difference Ybar - Cbar - delta * Kbar is: ', rcdiff_ss

    # Print SS computation time
    if ss_time < 60: # seconds
        secs = round(ss_time, 3)
        print 'SS computation time: ', secs, ' sec'
    elif ss_time >= 60 and ss_time < 3600: # minutes
        mins = int(ss_time / 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', mins, ' min, ', secs, ' sec'
    elif ss_time >= 3600 and ss_time < 86400: # hours
        hrs = int(ss_time / 3600)
        mins = int(((ss_time / 3600) - hrs) * 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'
    elif ss_time >= 86400: # days
        days = int(ss_time / 86400)
        hrs = int(((ss_time / 86400) - days) * 24)
        mins = int(((ss_time / 3600) - hrs) * 60)
        secs = round(((ss_time / 60) - mins) * 60, 1)
        print 'SS computation time: ', days, ' days,', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'

    '''
    --------------------------------------------------------------------
    Compute the equilibrium time path by TPI
    --------------------------------------------------------------------
    Gamma1        = [S-1,] vector, initial period savings distribution
    K1            = scalar > 0, initial period aggregate capital stock
    K_constr_tpi1 = boolean, =True if K1<=0 for given Gamma1
    Kpath_init    = [T+S-2,] vector, initial guess for the time path of
                    the aggregate capital stock
    aa            = scalar, parabola coefficient for Kpath_init
                    Kpath_init = aa*t^2 + bb*t + cc for 0<=t<=T-1
    bb            = scalar, parabola coefficient for Kpath_init
    cc            = scalar, parabola coefficient for Kpath_init
    Lpath         = [T+S-2,] vector, exogenous time path for aggregate
                    labor
    tpi_params    = length 15 tuple, (S, T, beta, sigma, L, A, alpha,
                    delta, K1, K_ss, C_ss, maxiter_TPI, mindist_TPI, xi,
                    TPI_tol)
    b_path        = [S-1, T+S-2] matrix, equilibrium time path of the
                    distribution of savings. Period 1 is the initial
                    exogenous distribution
    c_path        = [S, T+S-2] matrix, equilibrium time path of the
                    distribution of consumption.
    w_path        = [T+S-2,] vector, equilibrium time path of the wage
    r_path        = [T+S-2,] vector, equilibrium time path of the
                    interest rate
    K_path        = [T+S-2,] vector, equilibrium time path of the
                    aggregate capital stock
    Y_path        = [T+S-2,] vector, equilibrium time path of aggregate
                    output (GDP)
    C_path        = [T+S-2,] vector, equilibrium time path of aggregate
                    consumption
    EulErr_path   = [S-1, T+S-2] matrix, equilibrium time path of the
                    Euler errors for all the savings decisions
    tpi_time      = scalar, number of seconds to compute TPI solution
    ResDiff       = [T-1,] vector, errors in the resource constraint
                    from period 1 to T-1. We don't use T because we are
                    missing one individual's consumption in that period
    --------------------------------------------------------------------
    '''
    if TPI_solve == True:
        print 'BEGIN EQUILIBRIUM TIME PATH COMPUTATION'
        Gamma1 = 0.9 * b_ss
        # Make sure init. period distribution is feasible in terms of K
        K1, K_constr_tpi1 = c4ssf.get_K(Gamma1)
        if K1 <= 0:
            print 'Initial savings distribution is not feasible because K1<=0. Some element(s) of Gamma1 must increase.'
        else:
            # Choose initial guess of path of aggregate capital stock.
            # Use parabola specification aa*x^2 + bb*x + cc
            Kpath_init = np.zeros(T+S-1)
            # Kpath_init[:T] = np.linspace(K1, K_ss, T)
            aa = (K1 - K_ss) / ((T - 1) ** 2)
            bb = - 2 * (K1 - K_ss) / (T - 1)
            cc = K1
            aa = -bb / (2 * (T - 1))
            Kpath_init[:T] = (aa * (np.arange(0, T) ** 2) +
                             (bb * np.arange(0, T)) + cc)
            Kpath_init[T:] = K_ss
            # Generate path of aggregate labor
            L, L_constr = c4ssf.get_L(n_ss)
            Lpath_init = np.linspace(L, L, T+S-1)

            # Run TPI
            tpi_params = (S, T, beta, sigma, L, A, alpha, delta, K1,
                         K_ss, L_ss, C_ss, maxiter_TPI, mindist_TPI, xi,
                         TPI_tol)
            (b_path, c_path, w_path, r_path, K_path, Y_path, C_path,
                EulErr_path, tpi_time) = c4tpf.TPI(tpi_params, Kpath_init,
                Lpath_init, Gamma1, n_ss, b_ss, TPI_graphs)

            # Print diagnostics
            print 'The max. absolute difference in Yt-Ct-K{t+1}+(1-delta)*Kt is:'
            ResDiff = (Y_path[:T-1] - C_path[:T-1] - K_path[1:T] +
                      (1 - delta) * K_path[:T-1])
            print np.absolute(ResDiff).max()

            # Print TPI computation time
            if tpi_time < 60: # seconds
                secs = round(tpi_time, 3)
                print 'TPI computation time: ', secs, ' sec'
            elif tpi_time >= 60 and tpi_time < 3600: # minutes
                mins = int(tpi_time / 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', mins, ' min, ', secs, ' sec'
            elif tpi_time >= 3600 and tpi_time < 86400: # hours
                hrs = int(tpi_time / 3600)
                mins = int(((tpi_time / 3600) - hrs) * 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation time: ', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'
            elif tpi_time >= 86400: # days
                days = int(tpi_time / 86400)
                hrs = int(((tpi_time / 86400) - days) * 24)
                mins = int(((tpi_time / 3600) - hrs) * 60)
                secs = round(((tpi_time / 60) - mins) * 60, 1)
                print 'TPI computation timeu ', days, ' days,', hrs, ' hrs, ', mins, ' min, ', secs, ' sec'

