import numpy as np 
from cython_cssor import cyssor 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
#Exercise 1
 
resolution = 501
U= np.zeros((resolution,resolution))
A=np.linspace(0,1,resolution)
U[0]=np.sin(2*np.pi*A)
U[-1]=-U[0]
print "First",U
cyssor(U,1.9) 
print "Second",U
 
 
X, Y = np.meshgrid(A,A)
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.set_xlabel('Cake Today')
ax.set_ylabel('Cake Tomorrow')
ax.set_zlabel('Taste Shock')
plt.title('Da graaaaaaph')
ax.plot_surface (X, Y, U, rstride=5)
plt.show()
 
