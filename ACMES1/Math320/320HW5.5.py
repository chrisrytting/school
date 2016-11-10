import numpy as np
from numpy import fft
from matplotlib import pyplot as plt
from copy import copy

def cheb_interp(f,n):
    extrema = np.cos((np.pi * np.arange(2*n)) / n)
    samples = f(extrema)
    coeffs = np.real(fft.fft(samples/n))
    coeffs[0] = coeffs[0]/2
    coeffs[n] = coeffs[n]/2
    return coeffs[:n+1]

degrees = [2,4,8,16,32,64,128]

x = np.linspace(-1,1,500)

def f(x):
    if x < 0:
        return 1 + x
    elif x == 0:
        return 0
    elif x > 0:
        return x

vf = np.vectorize(f)
f_x = vf(x)
a = 1
for degree in degrees:
    plt.subplot(7,1,a)
    plt.ylabel('D' + str(degree))
    plt.yticks(np.arange(0,1.1,.5))
    if a < 7:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off') # labels along the bottom edge are off
    else:
        plt.xticks(np.arange(-1,1.1,1))
    plt.plot(x, f_x)
    coeff = cheb_interp(vf, degree)
    cheb = np.polynomial.chebyshev.Chebyshev(coeff)
    poly = np.polynomial.Polynomial(coeff)
    plt.plot(x, cheb(x))
    a+=1
plt.suptitle('Polynomials of different degrees')
plt.show()
