import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize
import LAMMPSTool as LT

def Proportional(xData, m,c):
    return m*xData+c

def IncreasingOnly(inArray, inTimes):
    arrMax = 0
    lstValues = []
    lstTimes = []
    for i in range(len(inArray)):
        if inArray[i] > arrMax:
            lstValues.append(inArray[i])
            if len(lstTimes) > 0:
                lstTimes.append(inTimes[i]-lstTimes[-1])
            else:
                lstTimes.append(inTimes[i])
            arrMax = inArray[i]
    return np.array(lstValues), np.array(lstTimes)

lstTemp = list(range(500,900,50))
lstITemp = list(map(lambda x: 1/x,lstTemp))
strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis001/Sigma5_5_25/Temp'
fltKeV = 8.617333262e-5

lstLogV = []
lstLogV2 = []
for i in lstTemp:
    arrValues = np.zeros(4)
    arrProjections = np.loadtxt(strRoot + str(i) + '/Projections.txt')
    arrTimes = np.loadtxt(strRoot + str(i) + '/Times.txt')
    arrFlattened = np.ravel(arrProjections)
    arr4Times = np.tile(arrTimes,4)
   # pop, popt = optimize.curve_fit(Proportional, np.sort(arr4Times), arrFlattened)
    lstIProjections = []
    lstITimes = []
    lstMValues = []
    for k in range(4):
        arrIProjections, arrITimes = IncreasingOnly(arrProjections[:,k],arrTimes)
        arrITimes = 1000*np.array((list(range(len(arrIProjections)))))
        lstIProjections.append(arrIProjections)
        lstITimes.append(arrITimes)
        pop,popt = optimize.curve_fit(Proportional, arrITimes, arrIProjections) 
        lstMValues.append(pop[0])
    arrAllIProjections = np.concatenate(lstIProjections)
    arrAllITimes = np.concatenate(lstITimes)
    #plt.scatter(arrAllITimes,arrAllIProjections)
    plt.show()
    pop, popt = optimize.curve_fit(Proportional, arrAllITimes, arrAllIProjections)  
    lstLogV2.append(np.median(lstMValues))
    #lstLogV2.append(np.log(pop[0]))
    for k in range(4):
        pop, popt = optimize.curve_fit(Proportional, arrTimes, arrProjections[:,k])    
        arrValues[k] = np.log(pop[0])
    lstLogV.append(arrValues)
arrLogV = np.vstack(lstLogV)
# for k in range(4):
#     arrIProjections = IncreasingOnly(arrProjections[:,k])
#     plt.scatter(np.array(list(range(len(arrIProjections))))*1000,arrIProjections)
# plt.show()
# plt.scatter(np.sort(arr4Times), arrFlattened)
# plt.show()
# plt.scatter(np.tile(lstITemp,4),np.ravel(arrLogV))
# plt.scatter(lstITemp, np.median(arrLogV,axis=1))
# plt.show()
# plt.scatter(lstITemp, lstLogV2)
# plt.legend(['Mean of 4', 'Single fit of all data'])
# plt.show()
#lstITemp = lstITemp[2:]
#lstLogV = lstLogV[2:]
#lstLogV2 = lstLogV2[2:]

print(-fltKeV*stats.linregress(lstITemp,np.log(np.mean(lstLogV,axis=1)))[0], -fltKeV*stats.linregress(lstITemp,lstLogV2)[0])


lstGrains = ['TJ.lst','1G.lst','2G.lst','12BV.lst', '13BV.lst', '32BH.lst']

dctPEValues = dict()
dctNValues = dict()

for k in lstGrains:
     objData = LT.LAMMPSData(strRoot[:-4]+'Min/' + str(k),1,4.05,LT.LAMMPSAnalysis3D)
     objLT = objData.GetTimeStepByIndex(-1)
     arrCol = objLT.GetColumnByName('c_pe1')
     arrCellVectors = objLT.GetCellVectors()
     dctPEValues[k] = np.sum(arrCol) 
     dctNValues[k] = len(arrCol)

nTJ = 0
fltTJPE = 0
for t in lstGrains[:3]:
    nTJ += dctNValues[t] 
    fltTJPE += dctPEValues[t]

nBC = 0
fltGBPE = 0

for s in lstGrains[3:]:
    nBC += dctNValues[s] 
    fltGBPE += dctPEValues[s]



arrProjections = np.loadtxt(strRoot + str(i) + '/Projections.txt')
arrTimes = np.loadtxt(strRoot + str(i) + '/Times.txt')
for k in range(4):
    plt.scatter(arrTimes ,arrProjections[:,k])
plt.legend(list(range(4)))
plt.show()

fltKeV = 8.617333262e-5
lstTemp = list(range(500,900,50))
strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis221/Sigma9_9_9/Temp'
lstColours = ['red', 'blue', 'green','black']
for i in lstTemp:
    arrLogV = np.loadtxt(strRoot + str(i) +  '/logV.txt')
    arrITemp = np.loadtxt(strRoot + str(i) + '/lstITemp.txt')
    #plt.scatter(arrITemp[0], arrLogVMean, c='b')
    #plt.scatter(arrITemp[:4], arrLogV[:4], c='r')
    for k in range(4):
        plt.scatter(arrITemp[0], arrLogV[k], c=lstColours[k])
    #plt.scatter([arrITemp[-1]], [arrLogV[-1]], c='g')
plt.legend(lstColours)
plt.show()

for i in lstTemp:
    arrLogV = np.loadtxt(strRoot + str(i) +  '/logV.txt')
    arrITemp = np.loadtxt(strRoot + str(i) + '/lstITemp.txt')
    arrLogVMean = np.mean(arrLogV)
    plt.scatter(arrITemp[0], arrLogVMean, c='b')
    plt.scatter(arrITemp[-1], arrLogV[-1], c='r')
plt.legend(['Mean of 4 velocities', 'Best fit of all the data'])
plt.show()

lstMean = []
lstAll = []
lstITemp = []

for i in lstTemp[:-3]:
    arrLogV = np.loadtxt(strRoot + str(i) +  '/logV.txt')
    arrITemp = np.loadtxt(strRoot + str(i) + '/lstITemp.txt')
    arrLogVMean = np.mean(arrLogV)
    lstMean.append(arrLogVMean)
    lstAll.append(arrLogV[-1])
    lstITemp.append(arrITemp[0])
    plt.scatter(arrITemp[0], arrLogVMean, c='b')
    plt.scatter(arrITemp[-1], arrLogV[-1], c='r')
plt.legend(['Mean of 4 velocities', 'Best fit of all the data'])
plt.show()

fltAll = stats.linregress(lstITemp,lstAll)[0]
fltMean = stats.linregress(lstITemp,lstMean)[0]
    
print(-fltAll*fltKeV, -fltMean*fltKeV)