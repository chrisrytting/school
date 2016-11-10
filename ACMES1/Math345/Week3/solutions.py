# name this file 'solutions.py'
"""Volume I Lab 3: Plotting with matplotlib
Chris Rytting
9-15-15
"""

# Add your import statements here.
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab

# Problem 1
def curve():
    """Plot the curve 1/(x-1) on [-2,6]. Plot the two sides of the curve separately
    (still with a single call to plt.plot()) so that the graph looks discontinuous 
    at x = 1.
    """
    f = lambda x: 1/(x-1)
    x = np.linspace(-2,6,500)
    r = f(x)
    r[r > 100] = np.inf
    r[r < -100] = -np.inf
    plt.plot(x, r, 'm--', linewidth = 4)
    plt.axis([-2,6,-6,6])
    plt.show()
    pass

# Problem 2
def colormesh():
    """Plot the function f(x,y) = sin(x)sin(y)/(xy) on [-2*pi, 2*pi]x[-2*pi, 2*pi].
    Include the scale bar in your plot.
    """
    
    f = lambda x,y: (np.sin(x)*np.sin(y))/(x*y)
    x = np.linspace(-2*np.pi, 2* np.pi, 100)
    y = np.linspace(-2*np.pi, 2* np.pi, 100)
    X,Y = np.meshgrid(x,y)
    plt.pcolormesh(X,Y,f(X,Y), cmap = 'seismic')
    plt.colorbar()
    plt.axis([-2*np.pi,2*np.pi,-2*np.pi,2*np.pi])
    plt.gca().set_aspect('equal')
    plt.show()
    pass

# Problem 3
def histogram():
    """Plot a histogram and a scatter plot of 50 random numbers chosen in the
    interval [0,1)
    """
    randomshaha = np.random.rand(50)
    plt.subplot(121)
    plt.hist(randomshaha, bins =5)
    plt.subplot(122)
    x = np.linspace(1,50)
    plt.scatter(x,randomshaha)
    mean = np.mean(randomshaha)
    meanline = np.ones(50)*mean
    plt.xlim(1,50)
    plt.plot(x,meanline, 'r')
    plt.show()
    pass
    
# Problem 4
def ripple():
    """Plot z = sin(10(x^2 + y^2))/10 on [-1,1]x[-1,1] using Mayavi."""
    X,Y = np.mgrid[-1:1:0.01, -1:1:0.01]
    Z = np.sin(10*(X**2 + Y**2))/10
    mlab.surf(X, Y, Z, colormap='RdYlGn') 
    mlab.show()
    pass
    
