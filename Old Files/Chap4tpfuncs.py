'''
------------------------------------------------------------------------
Last updated 8/5/2015

All the functions for the TPI computation from Chapter 4 of the OG
textbook
    get_cvec_lf
    LfEulerSys
    paths_life
    get_cbepath
    TPI
------------------------------------------------------------------------
'''
# Import Packages
import time
import numpy as np
import scipy.optimize as opt
import Chap4ssfuncs as c4ssf
reload(c4ssf)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import sys

'''
------------------------------------------------------------------------
    Functions
------------------------------------------------------------------------
'''

def get_cvec_lf(p, rpath, wpath, nvec, bvec):
    '''
    Generates vector of remaining lifetime consumptions from individual
    savings and labor supply, and the time path of interest rates and the real wages

    Inputs:
        p     = integer in [2,80], number of periods remaining in
                individual life
        rpath = [p,] vector, remaining interest rates
        wpath = [p,] vector, remaining wages
        nvec  = [p,] vector, remaining exogenous labor supply
        bvec  = [p,] vector, remaining savings including initial savings

    Functions called: None

    Objects in function:
        c_constr = [p,] boolean vector, =True if element c_s <= 0
        b_s      = [p,] vector, bvec
        b_sp1    = [p,] vector, last p-1 elements of bvec and 0 in last
                   element
        cvec     = [p,] vector, remaining consumption by age c_s

    Returns: cvec, c_constr
    '''
    c_constr = np.zeros(p, dtype=bool)
    b_s = bvec
    b_sp1 = np.append(bvec[1:], [0])
    '''
    print np.shape(rpath)
    print np.shape(wpath)
    print np.shape(b_s)
    print np.shape(b_sp1)
    print np.shape(nvec)
    '''
    cvec = (1 + rpath) * b_s + wpath * nvec - b_sp1
    if cvec.min() <= 0:
        # print 'initial guesses and/or parameters created c<=0 for some agent(s)'
        c_constr = cvec <= 0
    return cvec, c_constr

def LfEulerSys(b_n_vec, *objs):
    '''
    Generates vector of all Euler errors for a given bvec and nvec, which errors
    characterize all optimal lifetime decisions

    Inputs:
        b_n_vec    = [2p-1,] vector, consisting of bvec and nvec
        bvec       = [p-1,] vector, remaining lifetime savings decisions
                     where p is the number of remaining periods
        nvec       = [p,] vector, remaining lifetime labor decisions
                     where p is the number of remaining periods
        objs       = length 6 tuple,
                     (p, beta, sigma, beg_wealth, rpath, wpath)
        p          = integer in [2,S], remaining periods in life
        beta       = scalar in [0,1), discount factor
        sigma      = scalar > 0, coefficient of relative risk aversion
        beg_wealth = scalar, wealth at the beginning of first age

    Functions called:
        get_cvec_lf
        c4ssf.get_b_n_errors

    Objects in function:
        bvec2        = [p, ] vector, remaining savings including initial
                       savings
        cvec         = [p, ] vector, remaining lifetime consumption
                       levels implied by bvec2
        c_constr     = [p, ] boolean vector, =True if c_{s,t}<=0
        b_n_err_params = length 2 tuple, parameters to pass into
                       get_b_errors (beta, sigma)
        b_n_err_vec    = [p-1,] vector, Euler errors from lifetime
                       consumption vector

    Returns: b_n_err_vec
    '''
    p, beta, sigma, beg_wealth, rpath, wpath = objs
    bvec=b_n_vec[:p-1]
    #print "BVEC", bvec
    nvec=b_n_vec[p-1:]
    #print "NVEC", nvec
    bvec2 = np.append(beg_wealth, bvec)
    cvec, c_constr = get_cvec_lf(p, rpath, wpath, nvec, bvec2)
    b_n_err_params = (beta, sigma)
    b_n_err_vec = c4ssf.get_b_n_errors(b_n_err_params, rpath[1:], wpath, cvec, nvec,
                                   c_constr, diff=True)
    return b_n_err_vec


def paths_life(params, beg_age, beg_wealth, rpath, wpath,
               b_n_init):
    '''
    Solve for the remaining lifetime savings decisions of an individual
    who enters the model at age beg_age, with corresponding initial
    wealth beg_wealth.

    Inputs:
        params     = length 4 tuple, (S, beta, sigma, TPI_tol)
        S          = integer in [3,80], number of periods an individual
                     lives
        beta       = scalar in [0,1), discount factor for each model
                     period
        sigma      = scalar > 0, coefficient of relative risk aversion
        TPI_tol    = scalar > 0, tolerance level for fsolve's in TPI
        beg_age    = integer in [1,S-1], beginning age of remaining life
        beg_wealth = scalar, beginning wealth at beginning age
        rpath      = [S-beg_age+1,] vector, remaining lifetime interest
                     rates
        wpath      = [S-beg_age+1,] vector, remaining lifetime wages
        b_init     = [S-beg_age,] vector, initial guess for remaining
                     lifetime savings
        n_init     = [S-beg_age+1,] vector, initial guess for remaining
                     lifetime labor

    Functions called:
        LfEulerSys
        get_cvec_lf
        c4ssf.get_b_errors

    Objects in function:
        p            = integer in [2,S], remaining periods in life
        b_guess      = [p-1,] vector, initial guess for lifetime savings
                       decisions
        eullf_objs   = length 6 tuple, objects to be passed in to
                       LfEulerSys
                       (p, beta, sigma, beg_wealth, nvec, rpath, wpath)
        bpath        = [p-1,] vector, optimal remaining lifetime savings
                       decisions
        cpath        = [p,] vector, optimal remaining lifetime
                       consumption decisions
        c_constr     = [p,] boolean vector, =True if c_{p}<=0,
        b_n_err_params = length 2 tuple, parameters to pass into
                       c4ssf.get_b_errors (beta, sigma)
        b_n_err_vec    = [p-1,] vector, Euler errors associated with
                       optimal savings decisions

    Returns: b_n_path, cpath, b_n_err_vec
    '''
    S, beta, sigma, TPI_tol = params
    p = int(S - beg_age + 1)
    b_init = b_n_init[:p-1]
    n_init = b_n_init[p-1:]
    if beg_age == 1 and beg_wealth != 0:
        sys.exit("Beginning wealth is nonzero for age s=1.")
    if len(rpath) != p:
        #print len(rpath), S-beg_age+1
        sys.exit("Beginning age and length of rpath do not match.")
    if len(wpath) != p:
        sys.exit("Beginning age and length of wpath do not match.")
    '''
    if len(nvec) != p:
        sys.exit("Beginning age and length of nvec do not match.")
    '''
    b_guess = 1.01 * b_init
    n_guess = 1.01 * n_init
    b_n_guess = np.append(b_guess, n_guess)
    eullf_objs = (p, beta, sigma, beg_wealth, rpath, wpath)
    b_n_path = opt.fsolve(LfEulerSys, b_n_guess, args=(eullf_objs),
                       xtol=TPI_tol)
    cpath, c_constr = get_cvec_lf(p, rpath, wpath, b_n_path[p-1:],
            np.append(beg_wealth, b_n_path[:p-1]))
    b_n_err_params = (beta, sigma)
    b_n_err_vec = c4ssf.get_b_n_errors(b_n_err_params, rpath[1:], wpath, cpath,
            b_n_path[p-1:], c_constr, diff=True)

    return b_n_path, cpath, b_n_err_vec


def get_cbnepath(params, n_ss, rpath, wpath, Gamma1, b_ss):
    '''
    Generates matrices for the time path of the distribution of
    individual savings, individual consumption, and the Euler errors
    associated with the savings decisions.

    Inputs:
        params  = length 5 tuple, (S, T, beta, sigma, TPI_tol)
        S       = integer in [3,80], number of periods an individual
                  lives
        T       = integer > S, number of time periods until steady state
        beta    = scalar in [0,1), discount factor for each model period
        sigma   = scalar > 0, coefficient of relative risk aversion
        TPI_tol = scalar > 0, tolerance level for fsolve's in TPI
        n_ss    = [S,] vector, steady-state labor supply
        rpath   = [T+S-2,] vector, equilibrium time path of the interest
                  rate
        wpath   = [T+S-2,] vector, equilibrium time path of the real
                  wage
        Gamma1  = [S-1,] vector, initial period savings distribution
        b_ss    = [S-1,] vector, steady-state savings distribution

    Functions called:
        paths_life

    Objects in function:
        cpath      = [S, T+S-1] matrix,
        bpath      = [S-1, T+S-1] matrix,
        EulErrPath = [S-1, T+S-1] matrix,
        pl_params  = length 4 tuple, parameters to pass into paths_life
                     (S, beta, sigma, TPI_tol)
        p          = integer >= 2, represents number of periods
                     remaining in a lifetime, used to solve incomplete
                     lifetimes
        b_guess    = [p-1,] vector, initial guess for remaining lifetime
                     savings, taken from previous cohort's choices
        bveclf     = [p-1,] vector, optimal remaining lifetime savings
                     decisions
        cveclf     = [p,] vector, optimal remaining lifetime consumption
                     decisions
        b_err_veclf = [p-1,] vector, Euler errors associated with
                      optimal remaining lifetime savings decisions
        DiagMaskb   = [p-1, p-1] boolean identity matrix
        DiagMaskc   = [p, p] boolean identity matrix

    Returns: cpath, bpath, EulErrPath
    '''
    S, T, beta, sigma, TPI_tol = params
    cpath = np.zeros((S, T+S-1))
    bpath = np.append(Gamma1.reshape((S-1,1)), np.zeros((S-1, T+S-2)),
            axis=1)
    npath = np.zeros((S, T+S-1))
    EulErrPath_b = np.zeros((S-1, T+S-1))
    EulErrPath_n = np.zeros((S, T+S-1))
    # Solve the incomplete remaining lifetime decisions of agents alive
    # in period t=1 but not born in period t=1
    pl_params = (S, beta, sigma, TPI_tol)

    
    ''' 
    ###BEGIN SLOPPY CODE
    def solvefirstn(n1, *objs):
        b1, b2, w, r, beta, sigma = objs
        return (w * ( w * n1 + ( 1 + r ) * b1 - b2 ) ** -sigma) - (( 1 - n1 ) ** (-sigma))

    n_first = .9
    n_first_objs = Gamma1[-2], 0, wpath[0], rpath[0], beta, sigma
    npath[S-1, 0] = opt.fsolve(solvefirstn, n_first, args=(n_first_objs),
                       xtol=TPI_tol)
    cpath[S-1, 0] = (1 + rpath[0]) * Gamma1[S-2] + wpath[0] * npath[S-1,0]
    ###END SLOPPY CODE
    '''

    for p in xrange(1, S):
        # b_guess = b_ss[-p+1:]
        b_guess = np.diagonal(bpath[S-p:, :p-1])
        n_guess = n_ss[-p:]
        b_n_guess = np.append(b_guess, n_guess)
        b_n_veclf, cveclf, b_n_err_veclf = paths_life(pl_params, S-p+1, Gamma1[S-p-1], rpath[:p], wpath[:p], b_n_guess)
        bveclf = b_n_veclf[:p-1]
        nveclf = b_n_veclf[p-1:]
        '''
        ###CHECK INDIVIDUAL VECTORS OF b AND n
        print "Iteration ", p-1
        print "bvec, ", bveclf
        print "nvec, ", nveclf
        '''
        b_err_veclf = b_n_err_veclf[:p-1]
        n_err_veclf = b_n_err_veclf[p-1:]

        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        DiagMaskb = np.eye(p-1, dtype=bool)
        DiagMaskn = np.eye(p, dtype=bool)
        DiagMaskc = np.eye(p, dtype=bool)
        bpath[S-p:, 1:p] = DiagMaskb * bveclf + bpath[S-p:, 1:p]
        npath[S-p:, :p] = DiagMaskn * nveclf + npath[S-p:, :p]
        cpath[S-p:, :p] = DiagMaskc * cveclf + cpath[S-p:, :p]
        EulErrPath_b[S-p:, 1:p] = (DiagMaskb * b_err_veclf +
                                EulErrPath_b[S-p:, 1:p])
        EulErrPath_n[S-p:, :p] = (DiagMaskn * n_err_veclf +
                                EulErrPath_n[S-p:, :p])
        '''
        ###TO TEST COMPUTATION OF MATRICES
        ###m and n to test only
        m = 25
        n = 10
        print "bpath, ", bpath, np.shape(bpath), bpath[m,n]
        print "npath, ", npath, np.shape(npath), npath[m,n]
        print "cpath, ", cpath, np.shape(cpath), cpath[m,n]
        print "n_ss, ", n_ss
        print "b_ss, ", b_ss
        '''
    # Solve for complete lifetime decisions of agents born in periods
    # 1 to T and insert the vector lifetime solutions diagonally (twist
    # donut) into the cpath, bpath, and EulErrPath matrices
    DiagMaskb = np.eye(S-1, dtype=bool)
    DiagMaskc = np.eye(S, dtype=bool)
    DiagMaskn = np.eye(S, dtype=bool)
    for t in xrange(1, T+1): # Go from periods 1 to T-1
        # b_guess = b_ss
        b_guess = np.diagonal(bpath[:, t-1:t+S-1])
        n_guess = n_ss
        b_n_guess = np.append(b_guess, n_guess)
        b_n_veclf, cveclf, b_n_err_veclf = paths_life(pl_params, 1, 0,
            rpath[t-1:t+S-1], wpath[t-1:t+S-1], b_n_guess)
        # Insert the vector lifetime solutions diagonally (twist donut)
        # into the cpath, bpath, and EulErrPath matrices
        bveclf = b_n_veclf[:S-1]
        nveclf = b_n_veclf[S-1:]
        b_err_veclf = b_n_err_veclf[:S-1]
        n_err_veclf = b_n_err_veclf[S-1:]
        bpath[:, t:t+S-1] = DiagMaskb * bveclf + bpath[:, t:t+S-1]
        npath[:, t-1:t+S-1] = DiagMaskn * nveclf + npath[:, t-1:t+S-1]
        cpath[:, t-1:t+S-1] = DiagMaskc * cveclf + cpath[:, t-1:t+S-1]
        EulErrPath_b[:, t:t+S-1] = (DiagMaskb * b_err_veclf +
                                 EulErrPath_b[:, t:t+S-1])
        EulErrPath_n[:, t-1:t+S-1] = (DiagMaskn * n_err_veclf +
                                 EulErrPath_n[:, t-1:t+S-1])

    return cpath, bpath, npath, np.append(EulErrPath_n, EulErrPath_b)


def TPI(params, Kpath_init, Lpath_init, Gamma1, n_ss, b_ss, graphs):
    '''
    Generates steady-state time path for all endogenous objects from
    initial state (K1, Gamma1) to the steady state.

    Inputs:
        params      = length 15 tuple, (S, T, beta, sigma, L, A, alpha,
                      delta, K1, K_ss, C_ss maxiter_TPI, mindist_TPI,
                      xi, TPI_tol)
        S           = integer in [3,80], number of periods an individual
                      lives
        T           = integer > S, number of time periods until steady
                      state
        beta        = scalar in [0,1), discount factor for each model
                      period
        sigma       = scalar > 0, coefficient of relative risk aversion
        L           = scalar > 0, exogenous aggregate labor
        A           = scalar > 0, total factor productivity parameter in
                      firms' production function
        alpha       = scalar in (0,1), capital share of income
        delta       = scalar in [0,1], model-period depreciation rate of
                      capital
        K1          = scalar > 0, initial period aggregate capital stock
        K_ss        = scalar > 0, steady-state aggregate capital stock
        maxiter_TPI = integer >= 1, Maximum number of iterations for TPI
        mindist_TPI = scalar > 0, Convergence criterion for TPI
        xi          = scalar in (0,1], TPI path updating parameter
        TPI_tol     = scalar > 0, tolerance level for fsolve's in TPI
        Kpath_init  = [T+S-1,] vector, initial guess for the time path
                      of the aggregate capital stock
        Lpath_init  = [T+S-1,] vector, initial guess for the time path
                      of the aggregate labor supply
        Gamma1      = [S-1,] vector, initial period savings distribution
        n_ss        = [S,] vector, steady-state labor supply
        b_ss        = [S-1,] vector, steady-state savings distribution
        graphs      = boolean, =True if want graphs of TPI objects

    Functions called:
        c4ssf.get_r
        c4ssf.get_w
        get_cbepath
        c4ssf.get_K

    Objects in function:
        start_time   = scalar, current processor time in seconds (float)
        iter_TPI     = integer >= 0, current iteration of TPI
        dist_TPI     = scalar >= 0, distance measure for fixed point
        Kpath_new    = [T+S-2,] vector, new path of the aggregate
                       capital stock implied by household and firm
                       optimization
        r_params     = length 3 tuple, parameters passed in to get_r
        w_params     = length 2 tuple, parameters passed in to get_w
        cbne_params   = length 5 tuple. parameters passed in to
                       get_cbnepath
        rpath        = [T+S-2,] vector, equilibrium time path of the
                       interest rate
        wpath        = [T+S-2,] vector, equilibrium time path of the
                       real wage
        cpath        = [S, T+S-2] matrix, equilibrium time path values
                       of individual consumption c_{s,t}
        bpath        = [S-1, T+S-2] matrix, equilibrium time path values
                       of individual savings b_{s+1,t+1}
        EulErrPath   = [S-1, T+S-2] matrix, equilibrium time path values
                       of Euler errors corresponding to individual
                       savings b_{s+1,t+1} (first column is zeros)
        Kpath_constr = [T+S-2,] boolean vector, =True if K_t<=0
        Kpath        = [T+S-2,] vector, equilibrium time path of the
                       aggregate capital stock
        Y_params     = length 2 tuple, parameters to be passed to get_Y
        Ypath        = [T+S-2,] vector, equilibrium time path of
                       aggregate output (GDP)
        Cpath        = [T+S-2,] vector, equilibrium time path of
                       aggregate consumption
        elapsed_time = scalar, time to compute TPI solution (seconds)

    Returns: bpath, cpath, wpath, rpath, Kpath, Ypath, Cpath,
             EulErrpath, elapsed_time
    '''
    start_time = time.clock()
    (S, T, beta, sigma, L, A, alpha, delta, K1, K_ss, L_ss, C_ss, maxiter_TPI,
        mindist_TPI, xi, TPI_tol) = params
    iter_TPI = int(0)
    dist_TPI_K = 10.
    dist_TPI_L = 10.
    Kpath_new = Kpath_init
    Lpath_new = Lpath_init
    r_params = (A, alpha, delta)
    w_params = (A, alpha)
    cbne_params = (S, T, beta, sigma, TPI_tol)
    b_n_ss = np.append(b_ss, n_ss)

    while (iter_TPI < maxiter_TPI) and (dist_TPI_L >= mindist_TPI) and (dist_TPI_K >= mindist_TPI):
        iter_TPI += 1
        Kpath_init = xi * Kpath_new + (1 - xi) * Kpath_init
        Lpath_init = xi * Lpath_new + (1 - xi) * Lpath_init
        rpath = c4ssf.get_r(r_params, Kpath_init, Lpath_init)
        wpath = c4ssf.get_w(w_params, Kpath_init, Lpath_init)
        cpath, bpath, npath, EulErrPath = get_cbnepath(cbne_params, n_ss, rpath,
                                   wpath, Gamma1, b_ss)
        Kpath_new = np.zeros(T+S-1)
        Kpath_new[:T], Kpath_constr = c4ssf.get_K(bpath[:, :T])
        Kpath_new[T:] = K_ss * np.ones(S-1)
        Kpath_constr = np.append(Kpath_constr, np.zeros(S-1, dtype=bool))
        Kpath_new[Kpath_constr] = 1
        Lpath_new = np.zeros(T+S-1)
        Lpath_new[:T], Lpath_constr = c4ssf.get_L(npath[:, :T])
        Lpath_new[T:] = L_ss * np.ones(S-1)
        Lpath_constr = np.append(Lpath_constr, np.zeros(S, dtype=bool))
        Lpath_new[Lpath_constr] = 1

        # Check the distance of Kpath_new1
        dist_TPI_K = np.absolute((Kpath_new[1:T] - Kpath_init[1:T]) /
                   Kpath_init[1:T]).max()
        dist_TPI_L = np.absolute((Lpath_new[1:T] - Lpath_init[1:T]) /
                   Lpath_init[1:T]).max()
        print 'Iteration: ', iter_TPI, ', Kpath distance: ', dist_TPI_K, 'Lpath distance: ', dist_TPI_L, ' max Eul err: ', np.absolute(EulErrPath).max()
    print bpath
    print npath
    print cpath

    if iter_TPI == maxiter_TPI and dist_TPI > mindist_TPI:
        print 'TPI reached maxiter and did not converge.'
    elif iter_TPI == maxiter_TPI and dist_TPI <= mindist_TPI:
        print 'TPI converged in the last iteration. Should probably increase maxiter_TPI.'
    Kpath = Kpath_new
    Lpath = Lpath_new
    Y_params = (A, alpha)
    Ypath = c4ssf.get_Y(Y_params, Kpath, Lpath)
    Cpath = np.zeros(T+S-1)
    Cpath[:T-1] = c4ssf.get_C(cpath[:, :T-1])
    Cpath[T-1:] = C_ss * np.ones(S)
    elapsed_time = time.clock() - start_time

    if graphs == True:
        # Plot time path of aggregate capital stock
        tvec = np.linspace(1, T+S-1, T+S-1)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Kpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate capital stock')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate capital $K_{t}$')
        # plt.savefig('Kt_Sec2')
        plt.show()

        # Plot time path of aggregate labor supply
        tvec = np.linspace(1, T+S-1, T+S-1)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Kpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate labor supply')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate labor $L_{t}$')
        # plt.savefig('Lt_Sec2')
        plt.show()

        # Plot time path of aggregate output (GDP)
        tvec = np.linspace(1, T+S-1, T+S-1)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Ypath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate output (GDP)')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate output $Y_{t}$')
        # plt.savefig('Yt_Sec2')
        plt.show()

        # Plot time path of aggregate consumption
        tvec = np.linspace(1, T+S-1, T+S-1)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, Cpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for aggregate consumption')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Aggregate consumption $C_{t}$')
        # plt.savefig('Ct_Sec2')
        plt.show()

        # Plot time path of real wage
        tvec = np.linspace(1, T+S-1, T+S-1)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, wpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for real wage')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real wage $w_{t}$')
        # plt.savefig('wt_Sec2')
        plt.show()

        # Plot time path of real interest rate
        tvec = np.linspace(1, T+S-1, T+S-1)
        minorLocator   = MultipleLocator(1)
        fig, ax = plt.subplots()
        plt.plot(tvec, rpath)
        # for the minor ticks, use no labels; default NullFormatter
        ax.xaxis.set_minor_locator(minorLocator)
        plt.grid(b=True, which='major', color='0.65',linestyle='-')
        plt.title('Time path for real interest rate')
        plt.xlabel(r'Period $t$')
        plt.ylabel(r'Real interest rate $r_{t}$')
        # plt.savefig('rt_Sec2')
        plt.show()

        # Plot time path of individual savings distribution
        tgrid = np.linspace(1, T, T)
        sgrid = np.linspace(2, S, S - 1)
        tmat, smat = np.meshgrid(tgrid, sgrid)
        cmap_bp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual savings $b_{s,t}$')
        strideval = max(int(1), int(round(S/10)))
        ax.plot_surface(tmat, smat, bpath[:, :T], rstride=strideval,
            cstride=strideval, cmap=cmap_bp)
        # plt.savefig('bpath')
        plt.show()

        # Plot time path of individual savings distribution
        tgrid = np.linspace(1, T-1, T-1)
        sgrid = np.linspace(1, S, S)
        tmat, smat = np.meshgrid(tgrid, sgrid)
        cmap_cp = matplotlib.cm.get_cmap('summer')
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel(r'period-$t$')
        ax.set_ylabel(r'age-$s$')
        ax.set_zlabel(r'individual consumption $c_{s,t}$')
        strideval = max(int(1), int(round(S/10)))
        ax.plot_surface(tmat, smat, cpath[:, :T-1], rstride=strideval,
            cstride=strideval, cmap=cmap_cp)
        # plt.savefig('bpath')
        plt.show()

    return (bpath, cpath, wpath, rpath, Kpath, Ypath, Cpath, EulErrPath,
        elapsed_time)
