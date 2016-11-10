import numpy as np
from spectrum import speriodogram
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats

#11.3

def loadplot(datadir):
    var = np.genfromtxt(datadir, delimiter =',', names = True)
    var = var[1:] 
    n = len(var)
    newvar = np.zeros((n))
    for i in xrange(n):
        x,y = var[i]
        newvar[i] = y
    return newvar  

def plotdata(data):
    p1 = speriodogram(data)
    plt.plot(p1[20:])
    plt.show()
#GDP
GDP = loadplot("/Users/chrisrytting1/Vim/MCLGDP.csv")
#plotdata(GDP)
#CPI
CPI = loadplot("/Users/chrisrytting1/Vim/MCLCPI.csv")
#plotdata(CPI)
#Consumption
consumption = loadplot("/Users/chrisrytting1/Vim/MCLconsumption.csv")
#plotdata(consumption)
#Investment
investment = loadplot("/Users/chrisrytting1/Vim/MCLinvestment.csv")
#plotdata(investment)

#11.4
def cycletrend(vec1, vec2, vec3, vec4, lambd):
    gdpcycle, gdptrend = sm.tsa.filters.hpfilter(vec1, lambd)
    cpicycle, cpitrend = sm.tsa.filters.hpfilter(vec2, lambd)
    ccycle, ctrend = sm.tsa.filters.hpfilter(vec3, lambd)
    icycle, itrend = sm.tsa.filters.hpfilter(vec4, lambd)

    plt.subplot(2,2,1)
    plt.suptitle('HP Filter Cyclical')
    plt.plot(gdpcycle)
    plt.title('GDP')
    plt.subplot(2,2,2)
    plt.plot(cpicycle)
    plt.title('CPI')
    plt.subplot(2,2,3)
    plt.plot(ccycle)
    plt.title('Consumption')
    plt.subplot(2,2,4)
    plt.plot(icycle)
    plt.title('Investment')
    plt.show()

    plt.subplot(2,2,1)
    plt.suptitle('HP Filter Trend')
    plt.plot(gdptrend)
    plt.title('GDP')
    plt.subplot(2,2,2)
    plt.plot(cpitrend)
    plt.title('CPI')
    plt.subplot(2,2,3)
    plt.plot(ctrend)
    plt.title('Consumption')
    plt.subplot(2,2,4)
    plt.plot(itrend)
    plt.title('Investment')
    plt.show()

#cycletrend(GDP, CPI, consumption, investment, 1600)

#11.4

def genstats(data,GDP):
    datamean = np.mean(data)
    datastd = np.std(data)
    datacorr, p = stats.pearsonr(data, GDP[len(GDP)-len(data):])
    dataac, p = stats.pearsonr(data[:-1],data[1:])
    print 'Mean = ', datamean
    print 'Standard deviation = ', datastd
    print 'Correlation with GDP = ', datacorr
    print 'Autocorrelation = ', dataac

def stat(vec1, vec2, vec3, vec4, lambd):
    gdpcycle, gdptrend = sm.tsa.filters.hpfilter(vec1, lambd)
    cpicycle, cpitrend = sm.tsa.filters.hpfilter(vec2, lambd)
    ccycle, ctrend = sm.tsa.filters.hpfilter(vec3, lambd)
    icycle, itrend = sm.tsa.filters.hpfilter(vec4, lambd)
    print 'For lambda = ', lambd
    print 'GDP cycle info: '
    genstats(gdpcycle, vec1)
    print 'CPI cycle info: '
    genstats(cpicycle, vec1)
    print 'Consumption cycle info: '
    genstats(ccycle, vec1)
    print 'Investment cycle info: '
    genstats(icycle, vec1)

    print 'GDP trend info: '
    genstats(gdptrend, vec1)
    print 'CPI trend info: '
    genstats(cpitrend, vec1)
    print 'Consumption trend info: '
    genstats(ctrend, vec1)
    print 'Investment trend info: '
    genstats(itrend, vec1)
    return gdpcycle, gdptrend

#11.5
'''
stat(GDP, CPI, consumption, investment, 100) 
stat(GDP, CPI, consumption, investment, 400) 
stat(GDP, CPI, consumption, investment, 1600) 
stat(GDP, CPI, consumption, investment, 6400)
stat(GDP, CPI, consumption, investment, 25600)
'''
GDPcycle100, GDPtrend100 = stat(GDP, CPI, consumption, investment, 100) 
GDPcycle400, GDPtrend400 = stat(GDP, CPI, consumption, investment, 400) 
GDPcycle1600, GDPtrend1600 = stat(GDP, CPI, consumption, investment, 1600) 
GDPcycle6400, GDPtrend6400 = stat(GDP, CPI, consumption, investment, 6400) 
GDPcycle25600, GDPtrend25600 = stat(GDP, CPI, consumption, investment, 25600) 
'''
plt.subplot(2,3,1)
plt.suptitle('GDP Cycle')
plt.plot(GDPcycle100)
plt.title('Lambda 100')
plt.subplot(2,3,3)
plt.plot(GDPcycle400)
plt.title('Lambda 400')
plt.subplot(2,3,4)
plt.plot(GDPcycle1600)
plt.title('Lambda 1600')
plt.subplot(2,3,5)
plt.plot(GDPcycle6400)
plt.title('Lambda 6400')
plt.subplot(2,3,6)
plt.plot(GDPcycle25600)
plt.title('Lambda 25600')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('GDP Trend')
plt.plot(GDPtrend100)
plt.title('Lambda 100')
plt.subplot(2,3,3)
plt.plot(GDPtrend400)
plt.title('Lambda 400')
plt.subplot(2,3,4)
plt.plot(GDPtrend1600)
plt.title('Lambda 1600')
plt.subplot(2,3,5)
plt.plot(GDPtrend6400)
plt.title('Lambda 6400')
plt.subplot(2,3,6)
plt.plot(GDPtrend25600)
plt.title('Lambda 25600')
plt.show()
'''
#11.6
'''
def stat2(vec1, vec2, vec3, vec4, lambd):
    gdpcycle, gdptrend = sm.tsa.filters.hpfilter(vec1, lambd)
    cpicycle, cpitrend = sm.tsa.filters.hpfilter(vec2, lambd)
    ccycle, ctrend = sm.tsa.filters.hpfilter(vec3, lambd)
    icycle, itrend = sm.tsa.filters.hpfilter(vec4, lambd)
    return gdpcycle, cpicycle, ccycle, icycle

GPDcycle1600, CPIcycle1600, ccycle1600, icycle1600 = stat2(GDP, CPI, consumption, investment, 1600) 

GDPdiff = GDP[1:] - GDP[:-1]
CPIdiff = CPI[1:] - CPI[:-1]
consumptiondiff = consumption[1:] - consumption[:-1]
investmentdiff = investment[1:] - investment[:-1]

gdpcyclebp = sm.tsa.filters.bkfilter(GDP, low=6, high=32, K=8)
cpicyclebp = sm.tsa.filters.bkfilter(CPI, low=6, high=32, K=8)
ccyclebp = sm.tsa.filters.bkfilter(consumption, low=6, high=32, K=8)
icyclebp = sm.tsa.filters.bkfilter(investment, low=6, high=32, K=8)

genstats(GDPdiff, GDP)
genstats(CPIdiff, GDP)
genstats(consumptiondiff, GDP)
genstats(investmentdiff, GDP)
genstats(gdpcyclebp, GDP)
genstats(cpicyclebp, GDP)
genstats(ccyclebp, GDP)
genstats(icyclebp, GDP)

plt.subplot(2,3,1)
plt.suptitle('GDP')
plt.plot(GDPcycle1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(GDPdiff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(gdpcyclebp)
plt.title('BP')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('CPI')
plt.plot(CPIcycle1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(CPIdiff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(cpicyclebp)
plt.title('BP')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('Consumption')
plt.plot(ccycle1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(consumptiondiff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(ccyclebp)
plt.title('BP')
plt.show()

plt.subplot(2,3,1)
plt.suptitle('Investment')
plt.plot(icycle1600)
plt.title('HP')
plt.subplot(2,3,3)
plt.plot(investmentdiff)
plt.title('First-difference')
plt.subplot(2,3,4)
plt.plot(icyclebp)
plt.title('BP')
plt.show()
'''
#11.7


sp500 = loadplot("MCLSP500.csv")


sp500cycle, sp500trend = sm.tsa.filters.hpfilter(vec1, 1600)
sp500diff = sp500[1:] - sp500[:-1]
sp500cyclebp = sm.tsa.filters.bkfilter(sp500, low=6, high=32, K=8)












