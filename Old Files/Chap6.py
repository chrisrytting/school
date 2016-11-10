import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#5
import numpy as np
import scipy as sp
from scipy.optimize import fsolve

delta = 0.10
tau = 0.05
zbar = 0.0
gamma = 2.5 
beta = 0.98
alpha = 0.40

param_vec = [delta, tau, zbar, gamma, beta, alpha]
 
def state_defs5(kbar, lbar, parameters=param_vec):
    delta, tau, zbar, gamma, beta, alpha = parameters
    y = (kbar ** alpha) * (np.exp(zbar))
    i = kbar - (1 - delta) * kbar
    r = (alpha) * y / kbar
    w = (1 - alpha) * y
    T = tau * (w + (r - delta) * kbar)
    c = (1 - tau) * (w + (r-delta) * kbar) + T
 
    return np.array([y, i, c, r, w, T])
 
def opt_func5(guess, params=param_vec):
    delta, tau, zbar, gamma, beta, alpha = params
    kbar = guess
    y, i, c, r, w, T = state_defs5(kbar, 1, params)
    eul_err1 = beta * ((r - delta) * (1 - tau) + 1) - 1
 
    return np.array(eul_err1)
 
initial_guess = np.array(4.5)
ss = fsolve(opt_func5, initial_guess, args=(param_vec))
k_ss = ss
y_ss, i_ss, c_ss, r_ss, w_ss, T_ss = state_defs5(k_ss, 1)
'''
print 'k = ', k_ss

print 'y = ',y_ss, 
print 'i = ', i_ss, 
print 'c = ',c_ss, 
print 'r = ',r_ss,
print 'w = ',w_ss,
print 'T = ',T_ss

'''
#6
 
xi = 1.5
a = 0.5

param_vec = [delta, tau, zbar, gamma, beta, alpha, xi, a]
 
def state_defs6(kbar, lbar, parameters=param_vec):
    delta, tau, zbar, gamma, beta, alpha, xi, a = parameters
    y = (kbar ** alpha) * (np.exp(zbar) * lbar ** (1 - alpha))
    i = kbar - (1 - delta) * kbar
    r = (alpha * (kbar ** (alpha-1)) * (lbar * np.exp(zbar)) ** (1-alpha))
    w = (1-alpha)*(kbar**alpha)*(lbar**-alpha)*np.exp(zbar*(1-alpha)) 
    T = tau * (w * lbar + (r - delta) * kbar)
    c = (1 - tau) * (w * lbar + (r-delta) * kbar) + T
 
    return np.array([y, i, c, r, w, T])
 
 
def opt_func6(guess, params=param_vec):
    delta, tau, zbar, gamma, beta, alpha, xi, a = params
    kbar, lbar = guess
    y, i, c, r, w, T = state_defs6(kbar, lbar, params)
    eul_err1 = beta * ((r - delta) * (1 - tau) + 1) - 1
    eul_err2 = (-a * ((1-lbar) ** -xi))  + ((c**-gamma) * w * (1-tau)) 
 
    return np.array([eul_err1, eul_err2])
 
 
initial_guess = np.array([4.5, 0.4])
ss = fsolve(opt_func6, initial_guess, args=(param_vec))
k_ss, l_ss = ss
y_ss, i_ss, c_ss, r_ss, w_ss, T_ss = state_defs6(k_ss, l_ss)

print 'k = ', k_ss
print 'l= ', l_ss
print 'y = ',y_ss, 
print 'i = ', i_ss, 
print 'c = ',c_ss, 
print 'r = ',r_ss,
print 'w = ',w_ss,
print 'T = ',T_ss


#8

alpha = .35
beta = .98
rho = .95
sigma = .02
kbar = (beta*alpha)**(1/(1-alpha))

k = np.linspace(.5*kbar, 1.5*kbar, 26)
kprime = np.linspace(.5*kbar, 1.5*kbar, 26)
z = np.linspace(-5*sigma, 5*sigma, 26)
eps = np.ones((1,26))/26.0

utilmat = np.zeros((26,26,26))

def dT(vec1, vec2):
    diff = vec1-vec2
    diff = diff.reshape((1, 26**2))
    diffT = np.transpose(diff)
    dT = np.dot(diff, diffT)
    return dT

for i in range(26):
    for j in range(26):
        for x in range(26):
            utilmat[i][j][x] = ((np.exp(z[j])*(k[i]**alpha))-kprime[x])
utilmat = np.log(utilmat)
valuemat = np.zeros((26,26))
eps = np.transpose(eps)
valeps = np.dot(valuemat, eps)
rightbox = np.tile(valeps.reshape(1,1,26), (26,26,1))
cbox = utilmat + beta*rightbox
valuemat1 = np.max(cbox, axis = 2)
polmat1 = np.argmax(cbox, axis = 2)
oldvalue = valuemat1
oldpolicy = polmat1
norm = 5

while norm > 10**-9:
    valeps = np.dot(oldvalue, eps)
    rightbox = np.tile(valeps.reshape(1,1,26), (26,26,1))
    cbox = utilmat + beta*rightbox
    newvalue = np.max(cbox, axis = 2)
    newpolicy = np.argmax(cbox, axis = 2)
    norm = dT(oldvalue, newvalue)
    x += 1
    oldvalue = newvalue
print newvalue
print newpolicy
print x

newpolicy = .096 + (((newpolicy + 1)*.1927)/26)

fig = plt.figure()
X, Y = np.meshgrid(z, k)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,newpolicy,rstride=5)
ax.set_xlabel("Taste Shock Today")
ax.set_ylabel("Size of Cake Today")
ax.set_zlabel("Policy Function")
plt.show()

#closed form
closedform = (alpha*beta) *(np.exp(z)) * (k ** alpha)
fig = plt.figure()
X, Y = np.meshgrid(z, k)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,closedform,rstride=5)
ax.set_xlabel("Taste Shock Today")
ax.set_ylabel("Size of Cake Today")
ax.set_zlabel("Policy Function")
plt.show()

#9

alpha = .35
beta = .98
rho = .95
sigma = .02
kbar = ((beta*alpha)**(1/(1-alpha)))

k = np.linspace(np.log(.5*kbar), np.log(1.5*kbar), 26)
z = np.linspace(-5*sigma, 5*sigma, 26)
eps = np.linspace(1./26, 1./26, 26)
epsT = np.transpose(eps)

utilmat = np.zeros((26,26,26))

def dT(vec1, vec2):
    diff = vec1-vec2
    diff = diff.reshape((1, 26**2))
    diffT = np.transpose(diff)
    dT = np.dot(diff, diffT)
    return dT

for i in range(26):
    for j in range(26):
        for x in range(26):
            utilmat[i][j][x] = np.exp(z[j])*np.exp(k[i])**alpha-np.exp(k[x])

utilmat = np.log(utilmat)
valuemat = np.zeros((26,26))
valeps = np.dot(valuemat, epsT)
rightbox = np.tile(valeps.reshape(1,1,26), (26,26,1))
cbox = utilmat + beta*rightbox
valuemat1 = np.max(cbox, axis = 2)
oldvalue = valuemat1
oldpolicy = polmat1
norm = 5

while norm > 10**-9:
    valeps = np.dot(oldvalue, eps)
    rightbox = np.tile(valeps.reshape(1,1,26), (26,26,1))
    cbox = utilmat + beta*rightbox
    newvalue = np.max(cbox, axis = 2)
    norm = dT(oldvalue, newvalue)
    x += 1
    oldvalue = newvalue

print newvalue
print newpolicy
index = np.argmax(cbox, axis = 2)
policy = np.zeros((26,26))

for i in range(26):
    policy[i] = k[index[i]]

fig = plt.figure()
X, Y = np.meshgrid(z, k)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,policy,rstride=5)
ax.set_xlabel("Taste Shock Today")
ax.set_ylabel("Size of Cake Today")
ax.set_zlabel("Policy Function")
plt.show()

#closed form
closedform = np.zeros((26,26))
for i in range(26):
    for j in range(26):
        closedform[i][j] = np.log(alpha) + np.log(beta) + z[j] + alpha*k[i]
fig = plt.figure()
X, Y = np.meshgrid(z, k)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,closedform,rstride=5)
ax.set_xlabel("Taste Shock Today")
ax.set_ylabel("Size of Cake Today")
ax.set_zlabel("Policy Function")
plt.show()
