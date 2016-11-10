import numpy as np
from matplotlib import pyplot as plt
import random


B = .9


#9
#possible cake sizes

w = np.linspace(0.01, 1, 100)

#10

#w-w' matrix
wmat = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        if (i - j)>0:
            wmat[i][j] = w[i]- w[j]
        else:
            wmat[i][j] = 0.0000000001 
        
#Jprint wmat
zerovec = np.zeros((1,100))

#w' transposed and stacked
valuemat = np.tile(zerovec, (100,1))

#utility of w-w'
utilmat = np.log(wmat)
#print utilmat
cvmat = utilmat + B*valuemat

#max of consumption
contraction = np.max(cvmat,axis=1)

#policy function
policy = (np.argmax(cvmat, axis=1))/100.0

#11
diff = contraction - zerovec
difft = np.transpose(diff)

def dT(vec1, vec2):
    diff = vec1-vec2
    diffT = np.transpose(diff)
    dT = np.dot(diff, diffT)
    return dT

#12
valuemat2 = np.tile(contraction, (100,1))
cvmat2 = utilmat + B*valuemat2
contraction2 = np.max(cvmat2,axis = 1)
policy2 = (np.argmax(cvmat2, axis = 1))/100.0
dT2 =  dT(contraction, contraction2)
#print "delta2 = ", dT2

#13
valuemat3 = np.tile(contraction2, (100,1))
cvmat3 = utilmat + B*valuemat3
contraction3 = np.max(cvmat3,axis = 1)
policy3 = (np.argmax(cvmat3, axis = 1))/100.0
dT3 =  dT(contraction2, contraction3)
#print dT3
#14
contraction1i = contraction3
dTi = 5
x = 0
while dTi>(10**-9):


    valuemati = np.tile(contraction1i, (100,1))
    cvmati = utilmat + B*valuemati
    contractioni = np.max(cvmati, axis=1)
    policyi = (np.argmax(cvmati, axis=1))/100.0
    dTi = dT(contractioni, contraction1i)
    contraction1i = contractioni
    x += 1
    print dTi

#15

policyi[0] = 0
plt.plot(w, policyi)
plt.show()

#16


eps = np.linspace(.5, 3.5, 7)
#print eps






import scipy.stats
import sys
sys.path.insert(0, 'C:\Users\chrisrytting1')
import discretenorm as dn
import tauchenhussey as th
nodes = 7
mu = 2
sigma = .5
shock, probability=dn.discretenorm(nodes,mu,sigma)
#print shock, probability

#17

zeromate = np.zeros((100,7,100))
zeromate1= np.zeros((100,7))
vt1e = np.zeros((100, 7, 100))
for i in range(100):
    for j in range(7):
        for k in range (100):
            if (i - k)>0:
                vt1e[i][j][k] = np.log(w[i]- w[k])*shock[j]
            else:
                vt1e[i][j][k] = -1000 
cmate = vt1e + B*zeromate
maxwmate = np.max(cmate, axis = 2)
policye = np.argmax(cmate, axis = 2)/100.0
#print vt1e

#18

def dTe(cont1, cont2):
    diff = cont1-cont2
    diffr = np.reshape(diff, (1,700))
    diffT = np.transpose(diffr)
    dTe = np.dot(diffr, diffT)
    return dTe

#19

probabilityt = np.transpose(probability)
def reshapemat(matrix):
    valueshock = np.dot(matrix, probability)
    wmatchstick = np.reshape(valueshock, (1,1,100))
    wmatchstickreshape = np.tile(wmatchstick, (100,7,1))
    return wmatchstickreshape
cmate2 = vt1e + B*reshapemat(maxwmate)
maxwmate2 = np.max(cmate2, axis = 2)
policye2 = np.argmax(cmate2, axis = 2)/100.0
d2 = dTe(maxwmate2, maxwmate)

#20
cmate3 = vt1e + B*reshapemat(maxwmate2)
maxwmate3 = np.max(cmate3, axis = 2)
policye3 = np.argmax(cmate3, axis = 2)/100.0
d3 = dTe(maxwmate3, maxwmate2)


#21
d = 5
y=0
maxwmatej = maxwmate
while d>10**-9:
    wmatchstickreshape = reshapemat(maxwmatej)
    cmatei = vt1e + B*wmatchstickreshape
    maxwmatei = np.max(cmatei, axis = 2)
    policeyei = np.argmax(cmatei, axis = 2)/100.0
    d = dTe(maxwmatej,maxwmatei)
    maxwmatej=maxwmatei
    y+=1
    #print d
#print y

#22

policeyei[0] = 0
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

epsilon2 = np.linspace(.5, 3.5, 7)
fig = plt.figure()
X, Y = np.meshgrid(epsilon2, w)
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,policeyei,rstride=5)
ax.set_xlabel("Taste Shock Today")
ax.set_ylabel("Size of Cake Today")
ax.set_zlabel("Policy Function")


#plt. show()

#23 - 24
sigma = 0.5
rho = 0.5
mu = 2
M = 7
base_sigma = (0.5 + (rho/4))*sigma + (0.5 - (rho/4))*(sigma/np.sqrt(1-rho**2))

a, ceps = th.tauchenhussey(M, mu, rho, sigma, base_sigma)

V0 = np.zeros((100, 1, 7))

ceps = np.transpose(ceps)

V0ceps = np.dot(V0, ceps)

V0ceps = np.reshape(V0ceps, (100, 7, 1))

V0ceps = np.reshape(V0ceps, (1, 7, 100))

V0ceps = np.tile(V0ceps, (100,1,1))

W = np.zeros((100,100))

for i,x in enumerate(w):
    for j,y in enumerate(w):
        if (x-y)>0:
            W[i][j] = x - y
        else:
            W[i][j] = 10**-10
W = np.log(W)
W = np.reshape(W, (100,1,100))
W = np.tile(W, (1,7,1))

e = np.linspace(0.5, 3.5, 7)

e = np.reshape(e, (1,7,1))

W = W*e

V0 = np.dot(V0, ceps)

V0 = np.transpose(V0)

V0 = np.reshape(V0, (100,7,1))
V0d = np.reshape(V0, (1,7,100))
V0mat = np.tile(V0d, (100, 1, 1))
totalmat = W + B*V0mat
nV = np.max(totalmat, axis = 2)
nP = np.argmax(totalmat, axis = 2)

#print nP
#print nV

#25

dVmatrix = nV
dVmatrix = np.reshape(dVmatrix, (1,700))
dVmatrixT = np.transpose(dVmatrix)
norm = np.dot(dVmatrix, dVmatrixT)
#print "delta =",  norm


#26 - 28
norm = 5
while norm > 10**-9:
    v0 = nV
    nV = np.dot(nV, ceps)
    nV = np.reshape(np.transpose(nV), (1,7,100))
    V0mat = np.tile(nV, (100, 1, 1))

    for i in range(0,100):
        for j in range(0, 100):
            if j>=i:
                V0mat [i,:,j]=-100
    
    totalmat = V0mat*B + W
    nV = np.max(totalmat, axis = 2)
    dVmatrix = v0-nV
    dVmatrix = np.reshape(dVmatrix, (1,700))
    dVmatrixT = np.transpose(dVmatrix)
    norm = np.dot(dVmatrix, dVmatrixT)
    #print "delta = ", norm

    #use while loop for #26 = #28
#29

newpolicy = np.argmax(totalmat, axis = 2)
#print newpolicy

epsilon = np.linspace(.5,3.5, 7)
fig = plt.figure()
X,Y = np.meshgrid(epsilon, w)
ax = fig.gca(projection = '3d')
ax.plot_surface(X, Y, newpolicy, rstride=5)
ax.set_xlabel("Taste Shock Today")
ax.set_ylabel("Size of Cake Today")
ax.set_zlabel("Policy Function")

plt.show()

#30

B = 0.9

w = np.linspace(0.01, 1, 100)

wdiffmat = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        if (i - j)>0:
            wdiffmat[i][j] = w[i]- w[j]
        else:
            wdiffmat[i][j] = 0.0000000001 
utilmat = 1.0/wdiffmat
print utilmat[0][1]
policy = np.zeros(100)
valuemat = np.tile(zerovec, (100,1))


#construct w' minus policy function matrix

wtoday = np.linspace(.01, 1.0, 100)

wmat = np.zeros((100,100))

for i, x in enumerate(policy):
    for j, y in enumerate(wtoday):
        if (i-j) > 0: 
            wmat[i][j] = (y - x)
        else:
            wmat[i][j] = -0.000000001

wmat = 1.0/wmat

#construct matrix to be argmined
cmat = np.abs(utilmat - B * wmat)

#policy function from argmin
newpolicy = np.argmin(cmat, axis = 1)
newpolicy = (newpolicy+1)/100.0

print dT(newpolicy, policy)

norm = 5

oldpolicy = policy 
x
while norm > (10**-9):
    wpol = wtoday - oldpolicy    
    wmat = np.tile(wpol, (100,1))
    for i in range(100):
        for j in range(100):
            if i-j <= 0:
                wmat[i][j]=100000000
            if wmat[i][j] == 0:
                wmat[i][j] = 100000000
                '''
    for i, x in enumerate(oldpolicy):
        for j, y in enumerate(wtoday):
            if (i-j) > 0:
                if (y-x) == 0:
                    wmat[i][j] = 100000000000
            else:
                wmat[i][j] = 1000000000001
                '''
    wmat1 = 1.0/wmat
    cmat = np.abs(utilmat - B*wmat1)
    newpolicy1 = np.argmin(cmat, axis = 1)
    newpolicy = (newpolicy1+1)/100.0
    norm = dT(newpolicy, oldpolicy)
    x+=1
    print norm
    oldpolicy = newpolicy
print x

plt.plot(wtoday, newpolicy)
plt.xlabel('Cake today')
plt.ylabel('Cake Tomorrow')
plt.show()










    


