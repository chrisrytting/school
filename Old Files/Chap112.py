import numpy as np
from spectrum import speriodogram
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats

GDP = np.genfromtxt("MCLGDP.csv", delimiter=',', names=True)
GDP = GDP[1:]
n = len(GDP)
newGDP = np.zeros((n))
for i in xrange(n):
	x,y = GDP[i]
	newGDP[i] = y
print newGDP

p1 = speriodogram(newGDP)
plt.plot(p1[20:])
plt.show()

CPI = np.genfromtxt("MCLCPI.csv", delimiter=',', names=True)
CPI = CPI[1:]
n = len(CPI)
newCPI = np.zeros((n))
for i in xrange(n):
	x,y = CPI[i]
	newCPI[i] = y
print newCPI

p2 = speriodogram(newCPI)
plt.plot(p2[20:])
plt.show()

consump = np.genfromtxt("MCLconsumption.csv", delimiter=',', names=True)
consump = consump[1:]
n = len(consump)
newconsump = np.zeros((n))
for i in xrange(n):
	x,y = consump[i]
	newconsump[i] = y
print newconsump

p3 = speriodogram(newconsump)
plt.plot(p3[20:])
plt.show()

invest = np.genfromtxt("MCLinvestment.csv", delimiter=',', names=True)
invest = invest[1:]
n = len(invest)
newinvest = np.zeros((n))
for i in xrange(n):
	x,y = invest[i]
	newinvest[i] = y
print newinvest

p4 = speriodogram(newinvest)
plt.plot(p4[20:])
plt.show()


#11.4
c_cycle_1600, c_trend_1600 = sm.tsa.filters.hpfilter(newconsump, lamb=1600)
i_cycle_1600, i_trend_1600 = sm.tsa.filters.hpfilter(newinvest, lamb=1600)
gdp_cycle_1600, gdp_trend_1600 = sm.tsa.filters.hpfilter(newGDP, lamb=1600)
cpi_cycle_1600, cpi_trend_1600, = sm.tsa.filters.hpfilter(newCPI, lamb=1600)

plt.subplot(2,2,1)
plt.suptitle('HP Filter Cyclical')
plt.plot(gdp_cycle_1600)
plt.title('GDP')
plt.subplot(2,2,2)
plt.plot(c_cycle_1600)
plt.title('Consumption')
plt.subplot(2,2,3)
plt.plot(i_cycle_1600)
plt.title('Investment')
plt.subplot(2,2,4)
plt.plot(cpi_cycle_1600)
plt.title('CPI')
plt.show()

plt.subplot(2,2,1)
plt.suptitle('HP Filter Trend')
plt.plot(gdp_trend_1600)
plt.title('GDP')
plt.subplot(2,2,2)
plt.plot(c_trend_1600)
plt.title('Consumption')
plt.subplot(2,2,3)
plt.plot(i_trend_1600)
plt.title('Investment')
plt.subplot(2,2,4)
plt.plot(cpi_trend_1600)
plt.title('CPI')
plt.show()

#11.5
c_cycle_100, c_trend_100 = sm.tsa.filters.hpfilter(newconsump, lamb=100)
i_cycle_100, i_trend_100 = sm.tsa.filters.hpfilter(newinvest, lamb=100)
gdp_cycle_100, gdp_trend_100 = sm.tsa.filters.hpfilter(newGDP, lamb=100)
cpi_cycle_100, cpi_trend_100, = sm.tsa.filters.hpfilter(newCPI, lamb=100)

c_cycle_400, c_trend_400 = sm.tsa.filters.hpfilter(newconsump, lamb=400)
i_cycle_400, i_trend_400 = sm.tsa.filters.hpfilter(newinvest, lamb=400)
gdp_cycle_400, gdp_trend_400 = sm.tsa.filters.hpfilter(newGDP, lamb=400)
cpi_cycle_400, cpi_trend_400, = sm.tsa.filters.hpfilter(newCPI, lamb=400)

c_cycle_6400, c_trend_6400 = sm.tsa.filters.hpfilter(newconsump, lamb=6400)
i_cycle_6400, i_trend_6400 = sm.tsa.filters.hpfilter(newinvest, lamb=6400)
gdp_cycle_6400, gdp_trend_6400 = sm.tsa.filters.hpfilter(newGDP, lamb=6400)
cpi_cycle_6400, cpi_trend_6400, = sm.tsa.filters.hpfilter(newCPI, lamb=6400)

c_cycle_25600, c_trend_25600 = sm.tsa.filters.hpfilter(newconsump, lamb=25600)
i_cycle_25600, i_trend_25600 = sm.tsa.filters.hpfilter(newinvest, lamb=25600)
gdp_cycle_25600, gdp_trend_25600 = sm.tsa.filters.hpfilter(newGDP, lamb=25600)
cpi_cycle_25600, cpi_trend_25600, = sm.tsa.filters.hpfilter(newCPI, lamb=25600)

c_cycle_100_mean = np.mean(c_cycle_100)
c_cycle_100_std = np.std(c_cycle_100)
c_cycle_100_corr, p = stats.pearsonr(c_cycle_100, newGDP[len(newGDP)-len(c_cycle_100):])
c_cycle_100_ac, p =stats.pearsonr(c_cycle_100[:-1],c_cycle_100[1:])

c_trend_100_mean = np.mean(c_trend_100)
c_trend_100_std = np.std(c_trend_100)
c_trend_100_corr, p = stats.pearsonr(c_trend_100, newGDP[len(newGDP)-len(c_trend_100):])
c_trend_100_ac, p =stats.pearsonr(c_trend_100[:-1],c_trend_100[1:])

i_cycle_100_mean = np.mean(i_cycle_100)
i_cycle_100_std = np.std(i_cycle_100)
i_cycle_100_corr, p = stats.pearsonr(i_cycle_100, newGDP)
i_cycle_100_ac, p =stats.pearsonr(i_cycle_100[:-1],i_cycle_100[1:])

i_trend_100_mean = np.mean(i_trend_100)
i_trend_100_std = np.std(i_trend_100)
i_trend_100_corr, p = stats.pearsonr(i_trend_100, newGDP)
i_trend_100_ac, p =stats.pearsonr(i_trend_100[:-1],i_trend_100[1:])

gdp_cycle_100_mean = np.mean(gdp_cycle_100)
gdp_cycle_100_std = np.std(gdp_cycle_100)
gdp_cycle_100_ac, p =stats.pearsonr(gdp_cycle_100[:-1],gdp_cycle_100[1:])

gdp_trend_100_mean = np.mean(gdp_trend_100)
gdp_trend_100_std = np.std(gdp_trend_100)
gdp_trend_100_ac, p =stats.pearsonr(gdp_trend_100[:-1],gdp_trend_100[1:])

cpi_cycle_100_mean = np.mean(cpi_cycle_100)
cpi_cycle_100_std = np.std(cpi_cycle_100)
cpi_cycle_100_corr, p = stats.pearsonr(cpi_cycle_100, newGDP)
cpi_cycle_100_ac, p =stats.pearsonr(cpi_cycle_100[:-1],cpi_cycle_100[1:])

cpi_trend_100_mean = np.mean(cpi_trend_100)
cpi_trend_100_std = np.std(cpi_trend_100)
cpi_trend_100_corr, p = stats.pearsonr(cpi_trend_100, newGDP)
cpi_trend_100_ac, p =stats.pearsonr(cpi_trend_100[:-1],cpi_trend_100[1:])

c_cycle_400_mean = np.mean(c_cycle_400)
c_cycle_400_std = np.std(c_cycle_400)
c_cycle_400_corr, p = stats.pearsonr(c_cycle_400, newGDP[len(newGDP)-len(c_cycle_400):])
c_cycle_400_ac, p =stats.pearsonr(c_cycle_400[:-1],c_cycle_400[1:])

c_trend_400_mean = np.mean(c_trend_400)
c_trend_400_std = np.std(c_trend_400)
c_trend_400_corr, p = stats.pearsonr(c_trend_400, newGDP[len(newGDP)-len(c_trend_400):])
c_trend_400_ac, p =stats.pearsonr(c_trend_400[:-1],c_trend_400[1:])

i_cycle_400_mean = np.mean(i_cycle_400)
i_cycle_400_std = np.std(i_cycle_400)
i_cycle_400_corr, p = stats.pearsonr(i_cycle_400, newGDP)
i_cycle_400_ac, p =stats.pearsonr(i_cycle_400[:-1],i_cycle_400[1:])

i_trend_400_mean = np.mean(i_trend_400)
i_trend_400_std = np.std(i_trend_400)
i_trend_400_corr, p = stats.pearsonr(i_trend_400, newGDP)
i_trend_400_ac, p =stats.pearsonr(i_trend_400[:-1],i_trend_400[1:])

gdp_cycle_400_mean = np.mean(gdp_cycle_400)
gdp_cycle_400_std = np.std(gdp_cycle_400)
gdp_cycle_400_ac, p =stats.pearsonr(gdp_cycle_400[:-1],gdp_cycle_400[1:])

gdp_trend_400_mean = np.mean(gdp_trend_400)
gdp_trend_400_std = np.std(gdp_trend_400)
gdp_trend_400_ac, p =stats.pearsonr(gdp_trend_400[:-1],gdp_trend_400[1:])

cpi_cycle_400_mean = np.mean(cpi_cycle_400)
cpi_cycle_400_std = np.std(cpi_cycle_400)
cpi_cycle_400_corr, p = stats.pearsonr(cpi_cycle_400, newGDP)
cpi_cycle_400_ac, p =stats.pearsonr(cpi_cycle_400[:-1],cpi_cycle_400[1:])

cpi_trend_400_mean = np.mean(cpi_trend_400)
cpi_trend_400_std = np.std(cpi_trend_400)
cpi_trend_400_corr, p = stats.pearsonr(cpi_trend_400, newGDP)
cpi_trend_400_ac, p =stats.pearsonr(cpi_trend_400[:-1],cpi_trend_400[1:])

c_cycle_1600_mean = np.mean(c_cycle_1600)
c_cycle_1600_std = np.std(c_cycle_1600)
c_cycle_1600_corr, p = stats.pearsonr(c_cycle_1600, newGDP[len(newGDP)-len(c_cycle_1600):])
c_cycle_1600_ac, p =stats.pearsonr(c_cycle_1600[:-1],c_cycle_1600[1:])

c_trend_1600_mean = np.mean(c_trend_1600)
c_trend_1600_std = np.std(c_trend_1600)
c_trend_1600_corr, p = stats.pearsonr(c_trend_1600, newGDP[len(newGDP)-len(c_trend_1600):])
c_trend_1600_ac, p =stats.pearsonr(c_trend_1600[:-1],c_trend_1600[1:])

i_cycle_1600_mean = np.mean(i_cycle_1600)
i_cycle_1600_std = np.std(i_cycle_1600)
i_cycle_1600_corr, p = stats.pearsonr(i_cycle_1600, newGDP)
i_cycle_1600_ac, p =stats.pearsonr(i_cycle_1600[:-1],i_cycle_1600[1:])

i_trend_1600_mean = np.mean(i_trend_1600)
i_trend_1600_std = np.std(i_trend_1600)
i_trend_1600_corr, p = stats.pearsonr(i_trend_1600, newGDP)
i_trend_1600_ac, p =stats.pearsonr(i_trend_1600[:-1],i_trend_1600[1:])

gdp_cycle_1600_mean = np.mean(gdp_cycle_1600)
gdp_cycle_1600_std = np.std(gdp_cycle_1600)
gdp_cycle_1600_ac, p =stats.pearsonr(gdp_cycle_1600[:-1],gdp_cycle_1600[1:])

gdp_trend_1600_mean = np.mean(gdp_trend_1600)
gdp_trend_1600_std = np.std(gdp_trend_1600)
gdp_trend_1600_ac, p =stats.pearsonr(gdp_trend_1600[:-1],gdp_trend_1600[1:])

cpi_cycle_1600_mean = np.mean(cpi_cycle_1600)
cpi_cycle_1600_std = np.std(cpi_cycle_1600)
cpi_cycle_1600_corr, p = stats.pearsonr(cpi_cycle_1600, newGDP)
cpi_cycle_1600_ac, p =stats.pearsonr(cpi_cycle_1600[:-1],cpi_cycle_1600[1:])

cpi_trend_1600_mean = np.mean(cpi_trend_1600)
cpi_trend_1600_std = np.std(cpi_trend_1600)
cpi_trend_1600_corr, p = stats.pearsonr(cpi_trend_1600, newGDP)
cpi_trend_1600_ac, p =stats.pearsonr(cpi_trend_1600[:-1],cpi_trend_1600[1:])

c_cycle_6400_mean = np.mean(c_cycle_6400)
c_cycle_6400_std = np.std(c_cycle_6400)
c_cycle_6400_corr, p = stats.pearsonr(c_cycle_6400, newGDP[len(newGDP)-len(c_cycle_6400):])
c_cycle_6400_ac, p =stats.pearsonr(c_cycle_6400[:-1],c_cycle_6400[1:])

c_trend_6400_mean = np.mean(c_trend_6400)
c_trend_6400_std = np.std(c_trend_6400)
c_trend_6400_corr, p = stats.pearsonr(c_trend_6400, newGDP[len(newGDP)-len(c_trend_6400):])
c_trend_6400_ac, p =stats.pearsonr(c_trend_6400[:-1],c_trend_6400[1:])

i_cycle_6400_mean = np.mean(i_cycle_6400)
i_cycle_6400_std = np.std(i_cycle_6400)
i_cycle_6400_corr, p = stats.pearsonr(i_cycle_6400, newGDP)
i_cycle_6400_ac, p =stats.pearsonr(i_cycle_6400[:-1],i_cycle_6400[1:])

i_trend_6400_mean = np.mean(i_trend_6400)
i_trend_6400_std = np.std(i_trend_6400)
i_trend_6400_corr, p = stats.pearsonr(i_trend_6400, newGDP)
i_trend_6400_ac, p =stats.pearsonr(i_trend_6400[:-1],i_trend_6400[1:])

gdp_cycle_6400_mean = np.mean(gdp_cycle_6400)
gdp_cycle_6400_std = np.std(gdp_cycle_6400)
gdp_cycle_6400_ac, p =stats.pearsonr(gdp_cycle_6400[:-1],gdp_cycle_6400[1:])

gdp_trend_6400_mean = np.mean(gdp_trend_6400)
gdp_trend_6400_std = np.std(gdp_trend_6400)
gdp_trend_6400_ac, p =stats.pearsonr(gdp_trend_6400[:-1],gdp_trend_6400[1:])

cpi_cycle_6400_mean = np.mean(cpi_cycle_6400)
cpi_cycle_6400_std = np.std(cpi_cycle_6400)
cpi_cycle_6400_corr, p = stats.pearsonr(cpi_cycle_6400, newGDP)
cpi_cycle_6400_ac, p =stats.pearsonr(cpi_cycle_6400[:-1],cpi_cycle_6400[1:])

cpi_trend_6400_mean = np.mean(cpi_trend_6400)
cpi_trend_6400_std = np.std(cpi_trend_6400)
cpi_trend_6400_corr, p = stats.pearsonr(cpi_trend_6400, newGDP)
cpi_trend_6400_ac, p =stats.pearsonr(cpi_trend_6400[:-1],cpi_trend_6400[1:])

c_cycle_25600_mean = np.mean(c_cycle_25600)
c_cycle_25600_std = np.std(c_cycle_25600)
c_cycle_25600_corr, p = stats.pearsonr(c_cycle_25600, newGDP[len(newGDP)-len(c_cycle_25600):])
c_cycle_25600_ac, p =stats.pearsonr(c_cycle_25600[:-1],c_cycle_25600[1:])

c_trend_25600_mean = np.mean(c_trend_25600)
c_trend_25600_std = np.std(c_trend_25600)
c_trend_25600_corr, p = stats.pearsonr(c_trend_25600, newGDP[len(newGDP)-len(c_trend_25600):])
c_trend_25600_ac, p =stats.pearsonr(c_trend_25600[:-1],c_trend_25600[1:])

i_cycle_25600_mean = np.mean(i_cycle_25600)
i_cycle_25600_std = np.std(i_cycle_25600)
i_cycle_25600_corr, p = stats.pearsonr(i_cycle_25600, newGDP)
i_cycle_25600_ac, p =stats.pearsonr(i_cycle_25600[:-1],i_cycle_25600[1:])

i_trend_25600_mean = np.mean(i_trend_25600)
i_trend_25600_std = np.std(i_trend_25600)
i_trend_25600_corr, p = stats.pearsonr(i_trend_25600, newGDP)
i_trend_25600_ac, p =stats.pearsonr(i_trend_25600[:-1],i_trend_25600[1:])

gdp_cycle_25600_mean = np.mean(gdp_cycle_25600)
gdp_cycle_25600_std = np.std(gdp_cycle_25600)
gdp_cycle_25600_ac, p =stats.pearsonr(gdp_cycle_25600[:-1],gdp_cycle_25600[1:])

gdp_trend_25600_mean = np.mean(gdp_trend_25600)
gdp_trend_25600_std = np.std(gdp_trend_25600)
gdp_trend_25600_ac, p =stats.pearsonr(gdp_trend_25600[:-1],gdp_trend_25600[1:])

cpi_cycle_25600_mean = np.mean(cpi_cycle_25600)
cpi_cycle_25600_std = np.std(cpi_cycle_25600)
cpi_cycle_25600_corr, p = stats.pearsonr(cpi_cycle_25600, newGDP)
cpi_cycle_25600_ac, p =stats.pearsonr(cpi_cycle_25600[:-1],cpi_cycle_25600[1:])

cpi_trend_25600_mean = np.mean(cpi_trend_25600)
cpi_trend_25600_std = np.std(cpi_trend_25600)
cpi_trend_25600_corr, p = stats.pearsonr(cpi_trend_25600, newGDP)
cpi_trend_25600_ac, p =stats.pearsonr(cpi_trend_25600[:-1],cpi_trend_25600[1:])

print c_trend_25600_mean 
print c_trend_25600_std 

print i_cycle_25600_mean 
print i_cycle_25600_std 

print i_trend_25600_mean 
print i_trend_25600_std 

print gdp_cycle_25600_mean 
print gdp_cycle_25600_std 

print gdp_trend_25600_mean 
print gdp_trend_25600_std 

print cpi_cycle_25600_mean 
print cpi_cycle_25600_std 

print cpi_trend_25600_mean 
print cpi_trend_25600_std 

plt.subplot(2,3,1)
plt.suptitle('GDP Cycle')
plt.plot(gdp_cycle_100)
plt.title('Lambda 100')
plt.subplot(2,3,3)
plt.plot(gdp_cycle_400)
plt.title('Lambda 400')
plt.subplot(2,3,4)
plt.plot(gdp_cycle_1600)
plt.title('Lambda 1600')
plt.subplot(2,3,5)
plt.plot(gdp_cycle_6400)
plt.title('Lambda 6400')
plt.subplot(2,3,6)
plt.plot(gdp_cycle_25600)
plt.title('Lambda 25600')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('GDP Trend')
plt.plot(gdp_trend_100)
plt.title('Lambda 100')
plt.subplot(2,3,3)
plt.plot(gdp_trend_400)
plt.title('Lambda 400')
plt.subplot(2,3,4)
plt.plot(gdp_trend_1600)
plt.title('Lambda 1600')
plt.subplot(2,3,5)
plt.plot(gdp_trend_6400)
plt.title('Lambda 6400')
plt.subplot(2,3,6)
plt.plot(gdp_trend_25600)
plt.title('Lambda 25600')
plt.show()

#11.6
consump_diff = newconsump[1:]-newconsump[:-1]
gdp_diff = newGDP[1:]-newGDP[:-1]
invest_diff = newinvest[1:]-newinvest[:-1]
cpi_diff = newCPI[1:]-newCPI[:-1]

c_cycle_bp = sm.tsa.filters.bkfilter(newconsump, low=6, high=32, K=8)
i_cycle_bp = sm.tsa.filters.bkfilter(newinvest, low=6, high=32, K=8)
gdp_cycle_bp = sm.tsa.filters.bkfilter(newGDP, low=6, high=32, K=8)
cpi_cycle_bp = sm.tsa.filters.bkfilter(newCPI, low=6, high=32, K=8)

consump_diff_mean = np.mean(consump_diff)
c_cycle_bp_mean = np.mean(c_cycle_bp)
gdp_diff_mean = np.mean(gdp_diff)
gdp_cycle_bp_mean = np.mean(gdp_cycle_bp)
invest_diff_mean = np.mean(invest_diff)
i_cycle_bp_mean = np.mean(i_cycle_bp)
cpi_diff_mean = np.mean(cpi_diff)
cpi_cycle_bp_mean = np.mean(cpi_cycle_bp)

consump_diff_std = np.std(consump_diff)
c_cycle_bp_std = np.std(c_cycle_bp)
gdp_diff_std = np.std(gdp_diff)
gdp_cycle_bp_std = np.std(gdp_cycle_bp)
invest_diff_std = np.std(invest_diff)
i_cycle_bp_std = np.std(i_cycle_bp)
cpi_diff_std = np.std(cpi_diff)
cpi_cycle_bp_std = np.std(cpi_cycle_bp)
'''
consump_diff_corr, p = stats.pearsonr(consump_diff, newGDP[len(newGDP)-len(consump_diff):])
c_cycle_bp_corr, p = stats.pearsonr(c_cycle_bp, newGDP[len(newGDP)-len(c_cycle_bp):])
invest_diff_corr, p = stats.pearsonr(invest_diff, newGDP)
i_cycle_bp_corr, p = stats.pearsonr(i_cycle_bp, newGDP)
cpi_diff_corr, p = stats.pearsonr(cpi_diff, newGDP)
cpi_cycle_bp_corr, p = stats.pearsonr(cpi_cycle, newGDP)
'''
consump_diff_ac, p =stats.pearsonr(consump_diff[:-1],consump_diff[1:])
c_cycle_bp_ac, p =stats.pearsonr(c_cycle_bp[:-1],c_cycle_bp[1:])
gdp_diff_ac, p =stats.pearsonr(gdp_diff[:-1],gdp_diff[1:])
gdp_cycle_bp_ac, p =stats.pearsonr(gdp_cycle_bp[:-1],gdp_cycle_bp[1:])
invest_diff_ac, p =stats.pearsonr(invest_diff[:-1],invest_diff[1:])
i_cycle_bp_ac, p =stats.pearsonr(i_cycle_bp[:-1],i_cycle_bp[1:])
cpi_diff_ac, p =stats.pearsonr(cpi_diff[:-1],cpi_diff[1:])
cpi_cycle_bp_ac, p =stats.pearsonr(cpi_cycle_bp[:-1],cpi_cycle_bp[1:])

plt.subplot(2,3,1)
plt.suptitle('GDP')
plt.plot(gdp_cycle_1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(gdp_diff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(gdp_cycle_bp)
plt.title('BP')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('Consumption')
plt.plot(c_cycle_1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(consump_diff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(c_cycle_bp)
plt.title('BP')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('Investment')
plt.plot(i_cycle_1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(invest_diff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(i_cycle_bp)
plt.title('BP')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('CPI')
plt.plot(cpi_cycle_1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(cpi_diff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(cpi_cycle_bp)
plt.title('BP')
plt.show()

#11.7
SP = np.genfromtxt("MCLSP500.csv", delimiter=',', names=True)
SP = SP[1:]
n = len(SP)
newSP = np.zeros((n))
for i in xrange(n):
	x,y = SP[i]
	newSP[i] = y
newSP = np.log(newSP) 
newSP = newSP[:-1]
print newSP

sp_cycle_hp, sp_trend_hp = sm.tsa.filters.hpfilter(newSP, lamb=1600)
sp_cycle_bp = sm.tsa.filters.bkfilter(newSP, low=6, high=32, K=8)
sp_diff = newSP[1:]-newSP[:-1]

n = len(newSP)
ones = np.ones((n))
x1 = np.linspace(1,n,n)
x = np.array([ones, x1])
y = np.array([newSP])
x = x.T
xxi = np.linalg.inv(np.dot(x.T,x))
xy = np.dot(x.T, y.T)
b = np.dot(xxi, xy)
sp_ols = np.dot(x,b)

sp_cycle_hp_mean = np.mean(sp_cycle_hp)
sp_cycle_bp_mean = np.mean(sp_cycle_bp)
sp_diff_mean = np.mean(sp_diff)
sp_ols_mean = np.mean(sp_ols)

sp_cycle_hp_std = np.std(sp_cycle_hp)
sp_cycle_bp_std = np.std(sp_cycle_bp)
sp_diff_std = np.std(sp_diff)
sp_ols_std = np.std(sp_ols)

sp_cycle_hp_ac, p = stats.pearsonr(sp_cycle_hp[:-1],sp_cycle_hp[1:])
sp_cycle_bp_ac, p = stats.pearsonr(sp_cycle_bp[:-1],sp_cycle_bp[1:])
sp_diff_ac, p = stats.pearsonr(sp_diff[:-1],sp_diff[1:])
sp_ols_ac, p = stats.pearsonr(sp_ols[:-1],sp_ols[1:])

print 'sp_cycle_hp_mean =', sp_cycle_hp_mean
print 'sp_cycle_bp_mean =', sp_cycle_bp_mean
print 'sp_diff_mean =', sp_diff_mean
print 'sp_ols_mean =', sp_ols_mean 

print 'sp_cycle_hp_std =', sp_cycle_hp_std
print 'sp_cycle_bp_std =', sp_cycle_bp_std
print 'sp_diff_std =', sp_diff_std
print 'sp_ols_std =', sp_ols_std

print 'sp_cycle_hp_ac =', sp_cycle_hp_ac
print 'sp_cycle_bp_ac =', sp_cycle_bp_ac
print 'sp_diff_ac =', sp_diff_ac
print 'sp_ols_ac =', sp_ols_ac
