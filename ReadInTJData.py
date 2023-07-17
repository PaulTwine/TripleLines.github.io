import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy import optimize
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT 
import LatticeDefinitions as ld



# def TJCalculation(inArray: np.array,blnLocalOnly = False):
#     fltLength = inArray[0,-1]
#    # arrNMax = np.max([int(inArray[0,8]),int(inArray[0,9])])*np.ones(len(inArray)) 
#     arrTJMean = np.divide(inArray[:,5],inArray[:,9])
#     arrGBMean = np.divide(inArray[:,4],inArray[:,8])
#     arrTJAdjusted = inArray[:,3] + arrTJMean*(inArray[:,6]-inArray[:,7])
#     #arrGBExcess = inArray[:,2] + arrGBMean*(arrNMax-inArray[:,6])
#     arrTotalExcessEnergy = arrTJAdjusted - inArray[:,2]
#    # arrTotalExcessEnergy = inArray[:,3]-inArray[:,2] + arrTJMean*(inArray[:,6]-inArray[:,7])
#     return arrTotalExcessEnergy/fltLength
def TJCalculation(inArray: np.array,blnLocalOnly = False):
    fltLength = inArray[0,-1]
    arrNMax = np.max([int(inArray[0,8]),int(inArray[0,9])])*np.ones(len(inArray)) 
    arrTJMean = np.divide(inArray[:,5],inArray[:,9])
    arrGBMean = np.divide(inArray[:,4],inArray[:,8])
    #arrTJAdjusted = inArray[:,3] + arrTJMean*(inArray[:,6]-inArray[:,7])
    #arrGBAdjusted  = inArray[:,2]
    arrMu = inArray[0,4]/inArray[0,8]
    arrTJAdjusted = inArray[:,3] + arrMu*(arrNMax-inArray[:,7])
    arrGBAdjusted = inArray[:,2] + arrMu*(arrNMax-inArray[:,6])
    #arrTJAdjusted = inArray[:,3] + arrTJMean*(arrNMax-inArray[:,7])
    #arrGBAdjusted = inArray[:,2] + arrGBMean*(arrNMax-inArray[:,6])
    #arrTJAdjusted = inArray[:,3] - (-3.36*(arrNMax-inArray[:,7]))
    #arrGBAdjusted = inArray[:,2] - (-3.36*(arrNMax-inArray[:,6]))
    #arrGBExcess = inArray[:,2] + arrGBMean*(arrNMax-inArray[:,6])
    #arrTotalExcessEnergy = arrTJAdjusted - inArray[:,2]
    arrTotalExcessEnergy = arrTJAdjusted-arrGBAdjusted
   # arrTotalExcessEnergy = inArray[:,3]-inArray[:,2] + arrTJMean*(inArray[:,6]-inArray[:,7])
    return arrTotalExcessEnergy/fltLength


lstAxes = ['Axis001', 'Axis101', 'Axis111']

strBiDir = '/home/p17992pt/csf4_scratch/BiCrystal/'
strTJDir = '/home/p17992pt/csf4_scratch/TJCylinder/'
strAxis = 'Axis001'
intSigma = 17
strTJSigma = 'TJSigma' + str(intSigma) + '/'
arrValues = np.loadtxt(strTJDir + strAxis + '/' + strTJSigma + 'Values.txt')
arrSortedValues = arrValues[np.argsort(arrValues[:,1])]
arrTJValues = TJCalculation(arrSortedValues)/4
strKey = strAxis + ',' + str(intSigma)

lstMeanTJs = []
lstStdTJs = []
for j in range(0,100,10):
     lstMeanTJs.append(np.mean(arrTJValues[j:j+10]))
     lstStdTJs.append(np.std(arrTJValues[j:j+10]))

plt.scatter(np.array(list(range(0,9)))/10,np.array(lstMeanTJs)[:9])
plt.show()
plt.scatter(arrSortedValues[:,1]/10, arrTJValues)
plt.errorbar(arrSortedValues[:,1]/10, arrTJValues, yerr=0.1, c='black')
plt.show()


dctDMin = dict()

dctDMin['Axis001,5'] = [list(range(6,9))]
dctDMin['Axis001,13'] = [[6,7]]
dctDMin['Axis001,17'] = [list(range(0,8))]
dctDMin['Axis001,29'] = [list(range(2,8))]
#dctDMin['Axis001,53'] = [list(range(4,8))]
dctDMin['Axis001,37'] = [list(range(5,8))]

dctDMin['Axis101,3'] = [list(range(5,9))]
dctDMin['Axis101,9'] = [list(range(0,9))] #GB periodic unit rearrangement
dctDMin['Axis101,11'] = [list(range(5,9))]
dctDMin['Axis101,17'] = [list(range(7,8))]
dctDMin['Axis101,19'] = [list(range(0,7))]
dctDMin['Axis101,27'] = [list(range(6,9))] #disconnections nucleated in GB simulation cell

dctDMin['Axis111,3'] = [list(range(6,9))] #disconnections nucleated in GB simulation cell
dctDMin['Axis111,7'] = [list(range(7,9))] #disconnections nucleated in GB simulation cell
dctDMin['Axis111,13'] = [list(range(0,9))] #disconnections nucleated in GB simulatin cell
#dctDMin['Axis111,19'] = [list(range(6,9))] #discconections nucleated in GB simulation
dctDMin['Axis111,21'] = [list(range(6,9))] #distorted grain boundary disconnections in TJ simulation cell cylindrical grain disrupted
dctDMin['Axis111,31'] = [list(range(0,9))] #distorted grain boundary disconnections in TJ simulation cell cylindrical grain disrupted


arrRows = np.where(np.isin(arrSortedValues[:,1],dctDMin[strKey]))
print(np.mean(arrTJValues[arrRows]),np.std(arrTJValues[arrRows]))


arrRows2 = np.where(arrTJValues[arrRows] == np.max(arrTJValues[arrRows]))
#arrRows2 = np.where(arrTJValues[arrRows] > 0.05)
#arrRows2 = np.where(arrTJValues[arrRows] < -0.3 )
print(arrSortedValues[arrRows][arrRows2], arrTJValues[arrRows][arrRows2])


