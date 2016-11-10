import numpy as np
from matplotlib import pyplot as plt


def g_n(x,n):
    total = 0
    for k in xrange(1,n+1):
        total += (2./k)*np.sin(k*x)
    f = np.pi - x
    total -= f
    return total

print g_n(5, 100)
x = np.linspace(0,2*np.pi, 100)
plt.plot(g_n(x,1))
plt.plot(g_n(x,2))
plt.plot(g_n(x,3))
plt.plot(g_n(x,10))
plt.plot(g_n(x,100))
plt.plot(g_n(x,1000))
plt.plot(g_n(x,10000))
plt.show()








