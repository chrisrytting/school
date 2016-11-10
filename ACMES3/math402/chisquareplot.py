from scipy.stats import chi2
import numpy as np
import matplotlib.pyplot as plt
import seaborn 

dofs = np.arange(1,6)
for df in dofs:
    x = np.linspace(chi2.ppf(0.01, df), chi2.ppf(0.99, df), 100)
    plt.ylim(-.01,1)
    plt.plot(x, chi2.pdf(x, df), '-', label='df = {}'.format(df))
    plt.legend(loc='best')
plt.show()

