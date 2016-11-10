import numpy as np
from scipy.stats import norm
from scipy.stats import uniform

vals = norm.cdf([-6,-5,-4,-3,-2,-1,1,2,3,4,5,6])
val = []
val.append( vals[6] - vals[5])
val.append( vals[7] - vals[4])
val.append( vals[8] - vals[3])
val.append( vals[9] - vals[2])
val.append( vals[10] - vals[1])
val.append( vals[11] - vals[0])

#Problem 2
import matplotlib.pyplot as plt

print uniform.cdf(.333)

