from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
 
print "10.3"
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
g = X**3 - X**2 - Y**2
plt.contour(X, Y, g, [0])
plt.xlim(0,3)
plt.show()
 
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
g = X**3 + X**2 - Y**2
plt.contour(X, Y, g, [0])
plt.show()
 
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
g = X**3 - Y**2
plt.contour(X, Y, g, [0])
plt.show()
 
x, y, z = np.ogrid[-1:1:100j, -1:1:100j, -1:1:100j]
g = z**2 - x**2 + y**2
mlab.contour3d(g, contours = [0]) 
mlab.show()
 
x, y, z = np.ogrid[-1:1:100j, -1:1:100j, -1:1:100j]
g = x**2*y - z**2
mlab.contour3d(g, contours = [0]) 
mlab.show()

import numpy as np
from matplotlib import pyplot as plt
'''
x = np.linspace(-1.5,1.5, 200)
y = np.linspace(-1.5,1.5, 200)
X, Y = np.meshgrid(x,y)
g = lambda x: x**3 - x**2 - y**2
plt.contour(X, Y, g, [0])
plt.show()
'''
