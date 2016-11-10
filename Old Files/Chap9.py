import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
 
np.set_printoptions(precision=6,suppress=True)
print "Exercise 9.2"
 
alpha=.33
k=5.
z=1.
b=2.
t=.1
h=24.
guess=.5
epsilon=.00001
param=np.array([alpha,k,z,b,t,h])
def marketclearingwagew(w, paramaters=param):
    alpha,k,z,b,t,h=param
    demand=((1-alpha)*z/w)**(1/alpha)*k
    pi=z*k**alpha*(demand)**(1-alpha)-w*demand
    supply=h-(b/((1+b)*w))*(w*h+pi-t)
    eq=supply-demand
    return eq
 
 
param=np.array([alpha,k,z,b,t,h])
w = fsolve(marketclearingwagew,guess)
k = k+epsilon
param=np.array([alpha,k,z,b,t,h])
w2=fsolve(marketclearingwagew,guess)
wprime=-(w-w2)/epsilon
print w
print wprime
k = k+epsilon
param=np.array([alpha,k,z,b,t,h])
wprime2=-(w2-fsolve(marketclearingwagew,guess))/epsilon
print wprime2
wprimeprime=-(wprime-wprime2)/epsilon
print wprimeprime
 
K=np.linspace(1,15,100)
Y1=np.zeros((100))
for i in xrange(100):
    param=np.array([alpha,K[i],z,b,t,h])
    Y1[i]=fsolve(marketclearingwagew,guess)
Y2=w+wprime*(K-5)
Y3=w+wprime*(K-5)+.5*wprimeprime*(K-5)**2
axes = plt.gca()
plt.plot(K,Y1, label="Grid Solution")
plt.plot(K,Y2,'r--', label="Linear Approximation")
plt.plot(K,Y3,'g--',label="Quadratic Approximation")
plt.legend(loc='upper left')
plt.show()
 
 
k=10.
param=np.array([alpha,k,z,b,t,h])
w = fsolve(marketclearingwagew,guess)
k = k+epsilon
param=np.array([alpha,k,z,b,t,h])
w2=fsolve(marketclearingwagew,guess)
wprime=-(w-w2)/epsilon
print w
print wprime
k = k+epsilon
param=np.array([alpha,k,z,b,t,h])
wprime2=-(w2-fsolve(marketclearingwagew,guess))/epsilon
print wprime2
wprimeprime=-(wprime-wprime2)/epsilon
print wprimeprime
 
K=np.linspace(1,15,100)
Y1=np.zeros((100))
for i in xrange(100):
    param=np.array([alpha,K[i],z,b,t,h])
    Y1[i]=fsolve(marketclearingwagew,guess)
Y2=w+wprime*(K-10)
Y3=w+wprime*(K-10)+.5*wprimeprime*(K-10)**2
axes = plt.gca()
plt.plot(K,Y1, label="Grid Solution")
plt.plot(K,Y2,'r--', label="Linear Approximation")
plt.plot(K,Y3,'g--',label="Quadratic Approximation")
plt.legend(loc='upper left')
plt.show()
 
print "Exercise 9.3"
X=100
def cubeit (y,x):
    eq=(x**.35+.9*x-y)**(-2.5)-.95*(y**.35+.9*y)**(-2.5)
    return eq
guess=10.5156
x0= fsolve(cubeit,10, args=X)
X=X+epsilon
x1= fsolve(cubeit,10, args=X)
xprime0=(x1-x0)/epsilon
X=X+epsilon
x2= fsolve(cubeit,10, args=X)
xprime1=(x2-x1)/epsilon
xprimeprime0=(xprime1-xprime0)/epsilon
X=X+epsilon
x3= fsolve(cubeit,10, args=X)
xprime2=(x3-x2)/epsilon
xprimeprime1=(xprime2-xprime1)/epsilon
xprimeprimeprime=(xprimeprime1-xprimeprime0)/epsilon
 
X=np.linspace(99,101,100)
Y1=np.zeros((100))
for i in xrange(100):
    Y1[i]=fsolve(cubeit,10,args=X[i])
 
Y2=x0+xprime0*(X-100)
Y3=x0+xprime0*(X-100)+xprimeprime0*.5*(X-100)**2
Y4=x0+xprime0*(X-100)+xprimeprime0*.5*(X-100)**2+xprimeprimeprime/6.*(X-100)**3
dif1=Y2-Y1
dif2=Y3-Y1
dif3=Y4-Y1
print xprime0
print xprimeprime0
print xprimeprimeprime
plt.plot(X,dif1,'r', label="Linear Approximation")
plt.plot(X,dif2,'g--',label="Quadratic Approximation")
plt.plot(X,dif3,label="Cubic Approximation")
 
plt.legend(loc='lower right')
plt.show()
print "Exercise 4"
 
gamma = 2.5
xsi = 1.5
beta = 0.98
alpha = 0.40
a = 0.5
delta = 0.10
zbar = 0.0
tau = 0.05
rho = 0.9
epsilon=.0001
param_vec=np.array([.1,.05,2.5,.98,.4])
 
def state_defs5(kbar, parameters=param_vec):
    delta, tau, gamma, beta, alpha = parameters
    r = (alpha) * kbar ** (alpha-1)
    w = (1 - alpha) * kbar ** alpha
    T = tau * (w  + (r - delta) * kbar)
    c = (1 - tau) * (w + (r-delta) * kbar) + T
    y = (kbar ** alpha) 
    i = kbar - (1 - delta) * kbar
 
    return np.array([c, r, w, T, y, i])
  
  
def opt_func5(guess, params=param_vec):
    delta, tau, gamma, beta, alpha= params
    kbar = guess
    c, r, w, T, y, i = state_defs5(kbar, params)
    eul_err1 = beta * (c ** (-gamma) *((r - delta) * (1 - tau) + 1)) - c ** (-gamma)
    return eul_err1
 
initial_guess = np.array([1])
ss = fsolve(opt_func5, initial_guess, args=(param_vec))
k_ss = ss
def opt_func(kt,zt,kt1,zt1,kt2):
    eul_err1 = beta * (alpha*np.exp(zt1)*kt1**(alpha-1)*(np.exp(zt)*kt**alpha-kt1))/(np.exp(zt1)*kt1**alpha-kt2)-1
    return eul_err1
 
zbar=0
Hx=(opt_func(k_ss+epsilon,zbar,k_ss+epsilon,zbar,k_ss+epsilon)-opt_func(k_ss,zbar,k_ss,zbar,k_ss))/epsilon
Hz=(opt_func(k_ss,epsilon,k_ss,epsilon,k_ss)-opt_func(k_ss,zbar,k_ss,zbar,k_ss))/epsilon
Hx2=(opt_func(k_ss+2*epsilon,zbar,k_ss+2*epsilon,zbar,k_ss+2*epsilon)-opt_func(k_ss+epsilon,zbar,k_ss+epsilon,zbar,k_ss+epsilon))/epsilon
Hz2=(opt_func(k_ss,2*epsilon,k_ss,2*epsilon,k_ss)-opt_func(k_ss,epsilon,k_ss,epsilon,k_ss))/epsilon
Hzz=(Hz-Hz2)/epsilon
Hxx=(Hx-Hx2)/epsilon
Hxz=(opt_func(k_ss+epsilon,epsilon,k_ss+epsilon,epsilon,k_ss+2*epsilon)-opt_func(k_ss+epsilon,zbar,k_ss+epsilon,zbar,k_ss+epsilon))/epsilon
Hv=(opt_func(k_ss,zbar,k_ss,epsilon,k_ss)-opt_func(k_ss,zbar,k_ss,zbar,k_ss))/epsilon
Hv1=(opt_func(k_ss,zbar,k_ss,2*epsilon,k_ss)-opt_func(k_ss,zbar,k_ss,epsilon,k_ss))/epsilon
Hvv=(Hv1-Hv)/epsilon
 
print Hx
print Hz
print Hxx
print Hxz
print Hzz
print Hvv
