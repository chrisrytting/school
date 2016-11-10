import numpy as np
import csv
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

tbill = np.array([])
infl = np.array([])
wilshare = np.array([])
consumption = np.array([])
nasdaq = np.array([])
gold = np.array([])
home = np.array([])

"""
with open ('CPIAUCSL.csv', 'r') as csv_file:
    inflation_reader = csv.reader(csv_file)
    for d, x in  inflation_reader:
        inflation = np.append(inflation, float(x))

with open ('TB3MS.csv', 'r') as csv_file:
    tbill_reader = csv.reader(csv_file)
    for d, x in  tbill_reader:
        tbill = np.append(tbill, float(x))

with open ('WILL5000INDFC.csv', 'r') as csv_file:
    wilshare_reader = csv.reader(csv_file)
    for d, x in  wilshare_reader:
        wilshare = np.append(wilshare, float(x))

with open ('PCECC96-2.csv', 'r') as csv_file:
    consumption_reader = csv.reader(csv_file)
    for d, x in  consumption_reader:
        consumption = np.append(consumption, float(x))
"""
with open ('revisedNGH.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for d, wil, inf, bill, cons, nas, gld, hm  in  csv_reader:
        wilshare = np.append(wilshare, float(wil))
        infl = np.append(infl, float(inf))
        tbill = np.append(tbill, float(bill))
        consumption = np.append(consumption, float(cons))
        nasdaq = np.append(nasdaq, float(nas))
        gold = np.append(gold, float(gld))
        home = np.append(home, float(hm))

wilshare = wilshare[1:]
infl = infl[1:]
tbill = tbill[1:]
consumption = consumption[1:]
nasdaq = nasdaq[1:]
gold = gold[1:]
home = home[1:]

inflation = infl[1:]/infl[:-1]

gold = 1 + gold[1:]/gold[:-1] - inflation

m = consumption[1:]/consumption[:-1]

tbill = (1 + tbill[1:] / 100) ** (.25) - inflation + 1

wilshare = wilshare[1:]  / wilshare[:-1] - inflation + 1

nasdaq = 1 + nasdaq[1:] / nasdaq[:-1] - inflation

home = home[1:]/home[:-1] - inflation + 1


print m.size
print tbill.size
print wilshare.size
print gold.size
print nasdaq.size
print home.size


print m
print tbill
print wilshare
print inflation

def func(input):
    n = len(m)
    
    beta = input[0]

    gamma = input[1]
    
    eq1 = (np.dot((beta * m.T ** -gamma), wilshare)/(n - 1) ) - 1

    eq2 = (np.dot((beta * m.T ** -gamma), tbill)/(n - 1) ) - 1

    I = np.identity(2)

    N = np.array([eq1, eq2])
    
    N2 = np.dot(N, I)

    result = np.dot(N2, N)

    return result

solution = optimize.minimize(func, [.5, 5], method = "BFGS")
'''
values =  solution['x']

beta = np.linspace(-1,2,100)
gamma = np.linspace(-10,100,100)
X,Y = np.meshgrid(beta, gamma)

def f2(x,y):
    temp = np.array([m**x[i] for i in xrange(len(x))])
    n1 = np.array([x*np.dot(m**-y[i], wilshare)/(len(m)*1.0-1.0) - 1 for i in xrange(len(y))])
    n2 = np.array([x*np.dot(m**-y[i], tbill)/(len(m)*1.0-1.0) - 1 for i in xrange(len(y))])
    return n1**2 + n2**2
 
Z = f2(beta,gamma)
Z = np.array(Z)

fig1 = plt.figure(1)
ax = fig1.gca(projection = '3d')
ax.set_xlabel("Beta")
ax.set_ylabel("Gamma")
plt.title('Problem 1')
ax.plot_surface(X, Y, Z)
 
Gx, Gy = np.gradient(Z) # gradients with respect to x and y
G = (Gx**2+Gy**2)**.5  # gradient magnitude
N = G/G.max()  # normalize 0..1
surf = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1,
    facecolors=cm.jet(N),
    linewidth=0, antialiased=False, shade=False)
plt.show()
'''

#PROBLEM 2


def func2(input):
    beta = input[0]
    
    gamma = input[1]
    
    alpha = input[2]

    phi = input[3]

    m = beta * ( (alpha * consumption[1:] + phi * gamma) / (alpha * consumption[:-1] + phi * gamma) ) ** -gamma

    n = len(m)
    
    I = np.identity(5)

    eq1 = (np.dot(m, wilshare)/(n - 1) ) - 1

    eq2 = (np.dot(m, tbill)/(n - 1) ) - 1

    eq3 = (np.dot(m, gold)/(n - 1) ) - 1

    eq4 = (np.dot(m, home)/(n - 1) ) - 1

    eq5 = (np.dot(m, nasdaq)/(n - 1) ) - 1

    N = np.array([eq1, eq2, eq3, eq4, eq5])

    N2 = np.dot(N, I)

    result = np.dot(N2, N)

    return result


solution = optimize.minimize(func2, [.01, .01, .01, .01])

print solution


