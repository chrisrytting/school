import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from spectrum import speriodogram
import statsmodels.api as sm

def printdatamats(matrix, title, ilist, jlist, klist = []):
	
	print title

	if klist == []:
		for j in range(0, len(jlist)):
			print jlist[j]
			for i in range(0, len(ilist)):
				print ilist[i], matrix[i, j]
	else: 
		for j in range(0, len(jlist)):
			print jlist[j]
			for k in range(0, len(klist)):
				print klist[k]
				for i in range(0, len(ilist)):
					print ilist[i], matrix[i, j, k]


print "EXERCISE 3"

def extract_csv(filename):
	raw = np.genfromtxt(filename, delimiter= ',', names = True)
	n = len(raw)
	data = np.zeros(n)
	for i in xrange(n):
		x, y = raw[i]
		data[i] = y
	return data

GDPdata = extract_csv('MCLGDP.csv')
Consumptiondata = extract_csv('MCLconsumption.csv')
Investmentdata = extract_csv('MCLinvestment.csv')
CPIdata = extract_csv('MCLconsumption.csv')


plt.suptitle('Raw data')
plt.subplot(2, 2, 1)
myplot = speriodogram(GDPdata)
plt.plot(myplot[20:])
plt.title("GDP")
plt.subplot(2, 2, 2)
myplot = speriodogram(Consumptiondata)
plt.plot(myplot[20:])
plt.title("Consumption")
plt.subplot(2, 2, 3)
myplot = speriodogram(Investmentdata)
plt.plot(myplot[20:])
plt.title("Investment")
plt.subplot(2, 2, 4)
myplot = speriodogram(CPIdata)
plt.plot(myplot[20:])
plt.title("CPI")
plt.show()

print "EXERCISE 4"

c_cycle_1600, c_trend_1600 = sm.tsa.filters.hpfilter(Consumptiondata, lamb=1600)
i_cycle_1600, i_trend_1600 = sm.tsa.filters.hpfilter(Investmentdata, lamb=1600)
gdp_cycle_1600, gdp_trend_1600 = sm.tsa.filters.hpfilter(GDPdata, lamb=1600)
cpi_cycle_1600, cpi_trend_1600, = sm.tsa.filters.hpfilter(CPIdata, lamb=1600)

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


print "EXERCISE 5"

lambdalist = [100, 400, 1600, 6400, 25600]

def getmoments5(data):

	mean = np.mean(data)
	std = np.std(data)
	auto , p = stats.pearsonr(data[:-1], data[1:])
	if data.size != GDPdata.size:
		corr , p = stats.pearsonr(data , GDPdata[len(GDPdata)-len(data):])
	else:
		corr , p = stats.pearsonr(data , GDPdata)

	statarray = np.array([mean, std, corr, auto])

	return statarray

def getstatsmat5(data):
	
	#data corresponds to either GDPdata, Consumptiondata, etc

	#Stat, cycle/trend, lambda
	statmat = np.zeros((4, 2, 5))

	cyclemat = np.zeros((len(data), 5))
	trendmat = np.zeros((len(data), 5))

	for l in range(0, 5): #For each lambda
		cycle, trend = sm.tsa.filters.hpfilter(data, lambdalist[l])
		"""
		plt.plot(cycle)
		plt.title('Cycletest')
		plt.show()
		"""
		cyclemat[:, l] = cycle
		trendmat[:, l] = trend
		"""
		plt.plot(cyclemat[:, l])
		plt.title('Cycletest2')
		plt.show()
		"""
		cyclemean, cyclestdv, cyclecorr, cycleauto = getmoments5(cycle)
		trendmean, trendstdv, trendcorr, trendauto = getmoments5(trend)

		statmat[:, 0, l] = [cyclemean, cyclestdv, cyclecorr, cycleauto]
		statmat[:, 1, l] = [trendmean, trendstdv, trendcorr, trendauto]

	
	return statmat, cyclemat, trendmat

def printgraphs5(cyclemat, trendmat):
	
	plt.subplot(2,3,1)
	plt.suptitle('GDP Cycle')
	plt.plot(cyclemat[:,0])
	plt.title('Lambda 100')
	plt.subplot(2,3,3)
	plt.plot(cyclemat[:,1])
	plt.title('Lambda 400')
	plt.subplot(2,3,4)
	plt.plot(cyclemat[:,2])
	plt.title('Lambda 1600')
	plt.subplot(2,3,5)
	plt.plot(cyclemat[:,3])
	plt.title('Lambda 6400')
	plt.subplot(2,3,6)
	plt.plot(cyclemat[:,4])
	plt.title('Lambda 25600')
	plt.show()

	plt.subplot(2,3,1)
	plt.suptitle('GDP Trend')
	plt.plot(trendmat[:,0])
	plt.title('Lambda 100')
	plt.subplot(2,3,3)
	plt.plot(trendmat[:,1])
	plt.title('Lambda 400')
	plt.subplot(2,3,4)
	plt.plot(trendmat[:,2])
	plt.title('Lambda 1600')
	plt.subplot(2,3,5)
	plt.plot(trendmat[:,3])
	plt.title('Lambda 6400')
	plt.subplot(2,3,6)
	plt.plot(trendmat[:,4])
	plt.title('Lambda 25600')
	plt.show()

#__HPs is a (4, 2, 5) matrix that has mean, std, corr, auto, for GDPcycle and GDPtrend, all for each lambda
#__cyclemat is a (225, 5) matrix that has the HP cycle for each lambda. __trendmat is similiar
GDPHPs, GDPcyclemat, GDPtrendmat = getstatsmat5(GDPdata)
consumpHPs, consumpcyclemat, consumptrendmat = getstatsmat5(Consumptiondata)
investHPs, investcyclemat, investtrendmat = getstatsmat5(Investmentdata)
CPIHPs, CPIcyclemat, CPItrendmat = getstatsmat5(CPIdata)


printgraphs5(GDPcyclemat, GDPtrendmat)

#MAKE SURE TO COMMENT ON THE DATA!!!!!

#printdatamats(GDPHPs, "GDP:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["CYCLICAL", "\nTREND"], ["\nLambda = 100", "\nLambda = 400", "\nLambda = 1600", "\nLambda = 6400", "\nLambda = 25600"])
#printdatamats(consumpHPs, "\nConsumption:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["CYCLICAL", "\nTREND"], ["\nLambda = 100", "\nLambda = 400", "\nLambda = 1600", "\nLambda = 6400", "\nLambda = 25600"])
#printdatamats(investHPs, "\nInvestment:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["CYCLICAL", "\nTREND"], ["\nLambda = 100", "\nLambda = 400", "\nLambda = 1600", "\nLambda = 6400", "\nLambda = 25600"])
#printdatamats(CPIHPs, "\nCPI:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["CYCLICAL", "\nTREND"], ["\nLambda = 100", "\nLambda = 400", "\nLambda = 1600", "\nLambda = 6400", "\nLambda = 25600"])

print "EXERCISE 6"

def get_diffandBP(data):
	diff = data[1:] - data[:-1]
	BP = sm.tsa.filters.bkfilter(data, low=6, high=32, K=8)
	return diff, BP

def getstatmat6(diff, HP, BP):
	statmat = np.zeros((4, 3))
	
	statmat[:, 0] = getmoments5(diff)
	statmat[:, 1] = HP #This is the list of moments for the cyclical with lamba=1600
	statmat[:, 2] = getmoments5(BP)

	return statmat

def plotfilter6(diff, HP, BP, title):
	plt.subplot(2,3,1)
	plt.suptitle(title)
	plt.plot(HP)
	plt.title('HP')
	plt.subplot(2,3,3)
	plt.plot(diff)
	plt.title('First-difference')
	plt.subplot(2,3,4)
	plt.plot(BP)
	plt.title('BP')
	plt.show()	


GDPdiff, GDPBP = get_diffandBP(GDPdata)
consumpdiff, consumpBP = get_diffandBP(Consumptiondata)
investdiff, investBP = get_diffandBP(Investmentdata)
CPIdiff, CPIBP = get_diffandBP(CPIdata)


#Each __statmat is a 4x3 matrix with the 4 moments for diff, HP, and BP filters
#__HPs[:, 0, 2] is the list of moments for the cyclical component with lamba=1600
GDPstatmat = getstatmat6(GDPdiff, GDPHPs[:, 0, 2], GDPBP)
consumpstatmat = getstatmat6(consumpdiff, investHPs[:, 0, 2], consumpBP)
investstatmat = getstatmat6(investdiff, consumpHPs[:, 0, 2], investBP)
CPIstatmat = getstatmat6(CPIdiff, CPIHPs[:, 0, 2], CPIBP)


plotfilter6(GDPdiff, GDPcyclemat[:,0], GDPBP, "GDP")
plotfilter6(consumpdiff, consumpcyclemat[:,0], consumpBP, "Consumption")
plotfilter6(investdiff, investcyclemat[:,0], investBP, "Investment")
plotfilter6(CPIdiff, CPIcyclemat[:,0], CPIBP, "CPI")


printdatamats(GDPstatmat, "GDP:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["First-difference", "\nHP(lambda = 1600)", "\nBP(6, 32, K = 8)"])
printdatamats(consumpstatmat, "\nConsumption:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["First-difference", "\nHP(lambda = 1600)", "\nBP(6, 32, K = 8)"])
printdatamats(investstatmat, "\nInvestment:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["First-difference", "\nHP(lambda = 1600)", "\nBP(6, 32, K = 8)"])
printdatamats(CPIstatmat, "\nCPI:\n", ["Mean =", "Stdv =", "Correlation with GDP =", "Autocorrelation ="], ["First-difference", "\nHP(lambda = 1600)", "\nBP(6, 32, K = 8)"])


print "EXERCISE 7"

def getmoments7(data):

	mean = np.mean(data)
	std = np.std(data)
	auto , p = stats.pearsonr(data[:-1], data[1:])

	statarray = np.array([mean, std, auto])

	return statarray

def getOLS(data):
	n = len(data)
	ones = np.ones((n))
	x1 = np.linspace(1,n,n)
	x = np.array([ones, x1])
	y = np.array([data])
	x = x.T
	xxi = np.linalg.inv(np.dot(x.T,x))
	xy = np.dot(x.T, y.T)
	b = np.dot(xxi, xy)
	OLS = np.dot(x,b)

	return OLS

def getstatmatSP(SPfilterlist):
	SPstatmat = np.zeros((3, 4))

	for f in range(len(SPfilterlist)):
		stats = getmoments7(SPfilterlist[f])
		SPstatmat[:, f] = stats

	return SPstatmat

SPdata = extract_csv("C:\Users\James\Documents\JamesMCL\Econ\MCLSP500.csv")
SPdata = np.log(SPdata) 
SPdata = SPdata[1: -1] #The first and last elements of the data set is a nan

SP_OLS =  getOLS(SPdata)
SP_cycle_hp, sp_trend_hp = sm.tsa.filters.hpfilter(SPdata, lamb=1600)
SP_cycle_bp = sm.tsa.filters.bkfilter(SPdata, low=6, high=32, K=8)
SP_diff = SPdata[1:]-SPdata[:-1]
filterlist = [SP_OLS, SP_cycle_hp, SP_cycle_bp, SP_diff]

#SPstatmat is a 3x4 matrix with the rows as mean, stdv, and autocorrelation, and columns as the filter types
SPstatmat = getstatmatSP(filterlist)

printdatamats(SPstatmat, "S&P Index:\n", ["Mean =", "Stdv =", "Autocorrelation ="], ["OLS", "\nHP(lambda = 1600)", "\nBP(6, 32, K = 8)", "\nFirst-difference"])

#ARE WE TOTALLY SURE WE WANT THE STATS FOR THE CYCLE OR DO WE ACTUALLY WANT THE TREND PART??? QUESTION 7 SAYS "STATIONARY"...
