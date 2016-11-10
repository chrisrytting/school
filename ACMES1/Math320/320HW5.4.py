import numpy as np
from matplotlib import pyplot as plt


def calc_wj(j, x_vec):
    wj = []
    for k in xrange(len(x_vec)):
        if k != j:
            wj.append(1./(x_vec[j]-x_vec[k]))
    return np.prod(wj)

def polynomial(x, x_vec,y,w):
    num = []
    den = []
    for j in xrange(len(x)):
        num.append(w[j]*y[j]/(x_vec-x[j]))
        den.append(w[j]/(x_vec-x[j]))
    num = sum(num)
    den = sum(den)
    return(num/den)

def f(x):
    return np.abs(x)

def prob_17():
    x = [-1., -1./3, 1./3, 1.]
    y = [np.sin(-np.pi), np.sin(-np.pi/3), np.sin(np.pi/3), np.sin(np.pi)]
    w = []
    for j in xrange(len(x)):
        w.append(calc_wj(j,x))
    x_vec = np.linspace(-1.1,1.1,100)
    a = polynomial(x,x_vec,y,w)
    plt.plot(x_vec, a)
    #plt.plot(x,y)
    plt.plot(x_vec, np.sin(np.pi*x_vec))
    plt.show()

def prob_18():
    x = [np.cos(0*np.pi/3), np.cos(1*np.pi/3), np.cos(2*np.pi/3), np.cos(3*np.pi/3)]
    y = [np.sin(j * np.pi) for j in x]
    w = []
    for j in xrange(len(x)):
        w.append(calc_wj(j,x))
    x_vec = np.linspace(-1.1,1.1,100)
    a = polynomial(x,x_vec,y,w)
    print a
    plt.plot(x_vec, a)
    #plt.plot(x,y)
    plt.plot(x_vec, np.sin(np.pi*x_vec))
    plt.show()
