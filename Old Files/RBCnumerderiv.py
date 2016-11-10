"""
Created July 18, 2012

Author: Spencer Lyon

Created for "Macroeconomic Theory and Methods" by:
    David Spencer: Brigham Young University
    Kerk Phillips: Brigham Young University
    Rick Evans: Brigham Young University
    Ben Tengensen: Carnegie Mellon University
    Bryan Perry: Massachusetts Institute of Technology

Original MatLab code by Kerk P. (2012) was referenced in creating this file.
"""
from numpy import tile, array, empty, ndarray, zeros, log, asarray



def numeric_deriv(func, theta0, nx, ny, nz):
    """
    This function computes the matricies AA-MM in the log-linearizatin of
    the equations in the function 'func'.

    Parameters
    ----------
    func: function
        The function that generates a vector from the dynamic equations that are
        to be linearized. This must be written to evaluate to zero in the
        steady state. Often these equations are the Euler equations defining
        the model

    theta0: array, dtype=float
        A vector of steady state values for state parameters. Place the values
        of X in the front, then the Y values, followed by the Z's.

    nx: number, int
        The number of elements in X

    ny: number, int
        The number of elements in Y

    nz: number, int
        The number of elements in Z

    Returns
    -------
    AA - MM : 2D array, dtype=float:
        The equaitons described by Uhlig in the log-linearization.
    """
    theta0 = asarray(theta0)
    incr = 1e-7
    T0 = func(theta0)  # should be very close to zero.

    length = 3 * nx + 2 * (ny + nz)
    height = T0.size

    dev = tile(theta0.reshape(1, theta0.size), (length, 1))

    for i in range(length):
        dev[i, i] += incr

    bigMat = empty((height, length))
    for i in range(0,length):
        if i < 3 * nx + 2 * ny:
            bigMat[:, i] = theta0[i] * (func(dev[i, :]) - T0) / (1.0 + T0)
        else:
            bigMat[:, i] = (func(dev[i, :]) - T0) / (1.0 + T0)

    bigMat /= incr

    AA = array(bigMat[0:ny, nx:2 * nx])
    BB = array(bigMat[0:ny, 2 * nx:3 * nx])
    CC = array(bigMat[0:ny, 3 * nx + ny:3 * nx + 2 * ny])
    DD = array(bigMat[0:ny, 3 * nx + 2 * ny + nz:length])
    FF = array(bigMat[ny:ny + nx, 0:nx])
    GG = array(bigMat[ny:ny + nx, nx:2 * nx])
    HH = array(bigMat[ny:ny + nx, 2 * nx:3 * nx])
    JJ = array(bigMat[ny:ny + nx, 3 * nx:3 * nx + ny])
    KK = array(bigMat[ny:ny + nx, 3 * nx + ny:3 * nx + 2 * ny])
    LL = array(bigMat[ny:ny + nx, 3 * nx + 2 * ny:3 * nx + 2 * ny + nz])
    MM = array(bigMat[ny:ny + nx, 3 * nx + 2 * ny + nz:length])
    TT = log(1 + T0)
    WW = array(TT[0:ny])
    TT = array(TT[ny:ny + nx])

    AA = AA if AA else zeros((ny, nx))
    BB = BB if BB else zeros((ny, nx))
    CC = CC if CC else zeros((ny, ny))
    DD = DD if DD else zeros((ny, nz))




    return [AA, BB, CC, DD, FF, GG, HH, JJ, KK, LL, MM, TT, WW]
