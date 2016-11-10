"""
Solutions file for Volume 1 Lab 13
Call your solutions file 'lab13.py'
"""

import numpy as np
from matplotlib import pyplot as plt


def cent_diff_quotients(f, pts, h=1e-5):
    """ 
    Calculate the centered difference quotient of a function at some given points.
    Inputs: 
        f   -- a callable function
        pts -- an array of points
        h   -- floating point
    Returns:
        array of centered difference quotients
    """
    return (f(pts + h) - f(pts - h))/(2.0 * h)


def jacobian(f, m, n, pt, h=1e-5):
    """
    Compute the approximate Jacobian matrix of a function at a point using the 
    centered coefficients difference quotient.
    Discuss part 2 with a partner.
    Inputs:
        f  -- a callable function
        m  -- dimension of the range of f
        n  -- dimension of the domain of f
        pt -- n-dimensional array
        h  -- floating point
    Returns:
        Jacobian matrix (numpy array)
    """
    iden = np.eye(n)
    return np.hstack([np.vstack((f(pt + h*iden[i, :]) - f(pt - h*iden[i,:]))/(2.0*h)) for i in xrange(n)])

################################################################## Filters ####
gaussian_blur = 1. / 159 * np.array([[2.,  4.,  5.,  4.,  2.],
                                   [4.,  9.,  12., 9.,  4.],
                                   [5.,  12., 15., 12., 5.],
                                   [4.,  9.,  12., 9.,  4.],
                                   [2.,  4.,  5.,  4.,  2.]])

S = 1. / 8 * np.array([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
###############################################################################

def Filter(image, filt):
    """
    Apply a filter to an image.
    Try question 2 and discuss with a partner.
    Inputs:
        image  -- an array of intensities representing an image
        filter -- an array representing the filter to apply
    Returns:
        array of the filtered image intensities
    """
    m, n = image.shape
    l, k = filt.shape
    
    image_pad = np.zeros((n+l-1, n+k-1))
    image_pad[l/2:-(l/2), k/2:-(k/2)] = image
    
    C = np.empty_like(image)
    for i in xrange(n):
        for j in xrange(m):
            C[i, j] = np.sum(filt * image_pad[i: i+l, j : j+k])
    return C

def sobel(image):
    """
    Apply the Sobel filter to an image.
    Calculate the cutoff gradient (M) by using four times the average value of 
    the Euclidean norm of the partial derivatives.
    Inputs:
        image -- an array of intensities representing an image
    Returns:
        array after applying the sobel filter
    """
    image_x = Filter(image, S)
    image_y = Filter(image, S.T)
    magGrad = np.sqrt(image_x**2 + image_y**2)
    M = np.mean(magGrad) * 4
    return magGrad > M

'''
f = lambda x: np.exp(x)
pts = np.array([-2, -1, 0, 1, 2, .5])
print cent_diff_quotients(f, pts)
print np.exp(pts)
'''
'''
f = lambda x: np.array([np.exp(x[0]), np.exp(x[1]), np.exp(x[0]*x[1])])
x1 = np.array([1,2], dtype = float)
'''

img = plt.imread('cameraman.png')
blur = Filter(img, gaussian_blur)
plt.imshow(blur, cmap='gray')
plt.figure()
plt.imshow(img, cmap='gray')
plt.figure()
plt.imshow(sobel(blur), cmap='gray')
plt.show()
