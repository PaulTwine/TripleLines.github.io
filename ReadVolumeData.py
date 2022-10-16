#%%
from pydoc import stripid
import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp
from scipy import spatial
from scipy import stats 
from scipy.optimize import curve_fit


# arrValuesTJ  = np.loadtxt('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp600/u015/TJ/VolumeTJ.txt')
# arrValues12BV  = np.loadtxt('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp600/u015/12BV/Volume12BV.txt')
# arrValues13BV  = np.loadtxt('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp600/u015/13BV/Volume13BV.txt')
# plt.scatter(arrValuesTJ[0,:],arrValuesTJ[1,:])
# plt.scatter(arrValues12BV[0,:],arrValues12BV[1,:])
# plt.scatter(arrValues13BV[0,:],arrValues13BV[1,:])
# plt.xlabel('Time in fs')
# plt.ylabel('Mean distance between grain boundaries in Angstroms')
# plt.legend(['TJ', 'GB 1', 'GB 2'])
# plt.show()
#%%
def FitLine(x, a,b):
    return a*x + b
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp650/u01/'
objLog = LT.LAMMPSLog(strFilename + 'TJ/TJ.log')
arrValues = objLog.GetValues(1)
print(objLog.GetColumnNames(1))
plt.scatter(arrValues[50:,0],arrValues[50:,2])
popt, pop = curve_fit(FitLine,arrValues[50:,0], arrValues[50:,2])
plt.plot(arrValues[50:,0],FitLine(arrValues[50:,0],*popt),c='r')
plt.show()
print(np.corrcoef(arrValues[50:,0],arrValues[50:,2]),popt)
#%%
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp650' # 450/u'
lstFilenames = ['TJ', '12BV','13BV']
lstUValues = [0.005,0.01, 0.015,0.02,0.025,0.03]
strUValues = list(map(lambda s: str(s).split('.')[1], lstUValues))

for k in strUValues:
    lstLegend = []
    for j in lstFilenames:
        arrValues = np.loadtxt(strFilename  + '/u' + k + '/' + j +'/Volume' + j + '.txt')
        arrRows = np.where(arrValues[1,:] > 0)[0]
        arrTimes = arrValues[0,arrRows]
        arrVolumes = arrValues[1,arrRows]
        arrPE = arrValues[2,arrRows]
        arrCoeff = np.corrcoef(arrTimes,arrVolumes)
        #plt.scatter(arrTimes,arrVolumes)
        plt.scatter(arrTimes,arrPE)
        lstLegend.append(j)
    plt.legend(lstLegend)
    plt.show()
#%%

def FitCurve(inXValues, inParams):
    intLength = len(inParams)
    lstValues = []
    for k in range(intLength):
        lstValues.append(inXValues**k*inParams[k])
    arrValues = np.vstack(lstValues)
    return np.sum(arrValues, axis=0)
def FitThroughOrgin(x,a):
    return a*x
def FitLine(x, a,b):
    return a*x + b
def FitQuadratic(inX, a,b,c):
    return a*inX**2+b*inX + c
def DiffQuadratic(x, a,b):
    return 2*a*x + b 
#lstKelvin = [450,500,550,600,650]
lstKelvin = [650]
lstIKelvin = list(map(lambda x: 1/x, lstKelvin))
lstUValues = [0.005,0.01, 0.015,0.02,0.025,0.03]
strUValues = list(map(lambda s: str(s).split('.')[1], lstUValues))
lstUValues = np.array(lstUValues)*4/(4.05**3)
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp' # 450/u'
lstFilenames = ['TJ', '12BV','13BV']
lstTJM = []
lst12M = []
lst13M = []
lstTJI = []
lst12I = []
lst13I = []
for i in lstFilenames:
    for j in lstKelvin:
        lstVelocity = []
        for k in range(6):
            arrValues = np.loadtxt(strFilename + str(j) + '/u' + strUValues[k] + '/' + i +'/Volume' + i + '.txt')
            arrRows = np.where(arrValues[1,:] > 0)[0]
            arrTimes = arrValues[0,arrRows]
            arrVolumes = arrValues[1,arrRows]
            arrCoeff = np.corrcoef(arrTimes,arrVolumes)
            if i == 'TJ':
                #pop,popt = curve_fit(FitQuadratic,arrTimes, arrVolumes)
                #plt.plot(arrTimes,FitQuadratic(arrTimes,*pop))
                #lstVelocity.append(pop[1])
                pop,popt = curve_fit(FitLine,arrTimes, arrVolumes)
                plt.plot(arrTimes,FitLine(arrTimes,*pop))
                lstVelocity.append(np.abs(pop[0]))

            else:
                pop,popt = curve_fit(FitLine,arrTimes, arrVolumes)
                #plt.plot(arrTimes,FitLine(arrTimes,*pop))
                lstVelocity.append(np.abs(pop[0]))
            print(arrCoeff,pop)
            plt.title(i + ' with u=' + str(np.round(lstUValues[k],6)) + ' at temp ' + str(j))
            plt.scatter(arrTimes,arrVolumes)
            plt.show()
        #plt.scatter(lstUValues, lstVelocity)
        # pop,popt = curve_fit(FitThroughOrgin,lstUValues[:4], np.abs(lstVelocity[:4]))
        # plt.plot(lstUValues[:4], FitThroughOrgin(lstUValues[:4],pop[0]))
        # plt.scatter(lstUValues,lstVelocity)
        # plt.title(i + ' at ' + str(j))
        # plt.show()
    if i =='TJ':
        lstTJM.append(np.abs(pop[0]))
        lstTJI.append(1/j)
    elif i == '12BV':
        lst12M.append(np.abs(pop[0]))
        lst12I.append(1/j)
    elif i == '13BV':
        lst13M.append(np.abs(pop[0]))
        lst13I.append(1/j)

        #plt.show()

plt.scatter(lstTJI, np.log(lstTJM))
plt.show()
plt.scatter(lst12I, np.log(lst12M))
plt.show()
plt.scatter(lst13I, np.log(lst13M))
plt.show()
pop,popt = curve_fit(FitLine,lstTJI, np.log(lstTJM))
print(pop)
plt.show()
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*tuple(zip(*pts)))
# plt.show()