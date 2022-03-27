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
import re
import sys 
import matplotlib.lines as mlines



## Assumes columns are 2: GBPE, 3: TJPE, 4: GB Lattice Atoms PE 5: TJLattice Atoms, 6: Number of GB atoms, 7 Number of TJ atoms, -1 is TJ Length
def TJCalculation(inArray: np.array,blnLocalOnly = False):
    fltLength = inArray[0,-1]
   # arrNMax = np.max([int(inArray[0,8]),int(inArray[0,9])])*np.ones(len(inArray)) 
    arrTJMean = np.divide(inArray[:,5],inArray[:,9])
    arrGBMean = np.divide(inArray[:,4],inArray[:,8])
    arrTJAdjusted = inArray[:,3] + arrTJMean*(inArray[:,6]-inArray[:,7])
    #arrGBExcess = inArray[:,2] + arrGBMean*(arrNMax-inArray[:,6])
    arrTotalExcessEnergy = arrTJAdjusted - inArray[:,2]
   # arrTotalExcessEnergy = inArray[:,3]-inArray[:,2] + arrTJMean*(inArray[:,6]-inArray[:,7])
    return arrTotalExcessEnergy/fltLength
    #return (inArray[:,3]-inArray[:,2] +inArray[:,5]*(inArray[:,6]-inArray[:,7]))/inArray[0,-1]
def GBCalculation(inSGBArray: np.array,inBiArray, intSigma: int, strAxis: str):
    fltGBArea, fltCylinderArea = FindTotalGBArea(intSigma,strAxis)
    fltExcessGB = inSGBArray[:,2] -3.36*(inSGBArray[0,6]*np.ones(len(inSGBArray))-inSGBArray[:,6]) -(-3.36*inSGBArray[0,6]*np.ones(len(inSGBArray))) #sGB
    #fltExcess = inArray[:,3] -3.36*(inArray[0,7]*np.ones(len(inArray))-inArray[:,7]) -(-3.36*inArray[0,7]*np.ones(len(inArray))) #sTJ
    fltExcessCSL = FindCSLExcess(inBiArray)
    return (fltExcessCSL/fltGBArea + (fltExcessGB-fltExcessCSL)/fltCylinderArea)/2
def FindCSLExcess(inCSLArray: np.array):
    return (inCSLArray[:,1]+(np.ones(len(inCSLArray))*inCSLArray[0,3]-inCSLArray[:,3])*(-3.36) -inCSLArray[0,3]*np.ones(len(inCSLArray))*(-3.36))/(2*inCSLArray[0,4]) 

def FindTotalGBArea(intSigma: int, strAxis: str):
    arrAxis = np.zeros(3)
    for j in range(3):
        arrAxis[j] = int(strAxis[j-3])
    objSigma = gl.SigmaCell(arrAxis.astype('int'), ld.FCCCell)
    objSigma.MakeCSLCell(intSigma)
    arrSigmaBasis = objSigma.GetBasisVectors()
    s0 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
    s1 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
    s2 = np.linalg.norm(arrSigmaBasis, axis=1)[2]    
    intHeight = 4
    intAtoms = 1*10**5
    intAtomsPerCell = 4 
    a = 4.05 ##lattice parameter
    h = a*np.round(intHeight/s2,0)
    i = np.sqrt((intAtoms/intAtomsPerCell)*a/(28*10*h*s0*s1))
    i = np.round(i,0).astype('int')
    r = 2*a*s1*i
    w = 28*a*i
    l = 10*a*i
    return 2*l*h, 4*r*np.pi*h
def SECalculation(inArray: np.array):
    fltLength = inArray[0,-1]
    arrTJMean = np.divide(inArray[:,5],inArray[:,7])
   # arrGBMean = np.divide(inArray[:,4],inArray[:,6])
    arrStrainDifference = inArray[:,5]-inArray[:,4] +(inArray[:,8]-inArray[:,9])*arrTJMean
    return arrStrainDifference/fltLength


lstAxes = ['Axis001', 'Axis101','Axis111']

lstAxis001 = [5,13,17,25,29]
lstAxis101 = [3,9,11,17,27]
lstAxis111 = [3,7,13,19,21]
dctAllTJ = dict()
dctCSLGB = dict()
dctAllGB = dict()
dctLists = dict()
dctDMIN = dict()
dctLists[lstAxes[0]] = lstAxis001
dctLists[lstAxes[1]] = lstAxis101
dctLists[lstAxes[2]] = lstAxis111
lstMetaResults = []
lstLowerResults = []
lstUpperResults = []
lstGBResults = []
lstMinResults = []
for strAxis in lstAxes:
    strBiDir = '/home/p17992pt/csf4_scratch/BiCrystal/'
    strTJDir = '/home/p17992pt/csf4_scratch/TJCylinder/'
    for intSigma in dctLists[strAxis]:
        strBiSigma = 'Sigma' + str(intSigma) + '/'
        strTJSigma = 'TJSigma' + str(intSigma) + '/'
        arrValues = np.loadtxt(strBiDir + strAxis + '/' + strBiSigma + 'Values.txt')
        #arrCSLGBExcessPerArea = (arrValues[:,1]+(np.ones(len(arrValues))*arrValues[0,3]-arrValues[:,3])*arrValues[:,2] -arrValues[0,3]*arrValues[:,2])/arrValues[0,4]
        # arrCSLGBExcessPerArea = (arrValues[:,1]+(np.ones(len(arrValues))*arrValues[0,3]-arrValues[:,3])*(-3.36) -arrValues[0,3]*np.ones(len(arrValues))*(-3.36))/(2*arrValues[0,4])
        #arrCSLGBExcessPerArea = FindCSLExcess(arrValues)
        arrValues = np.tile(arrValues, (10,1))
        arrCSLGBExcessPerArea = FindCSLExcess(arrValues)[:10] 
        arrUnique,index = np.unique(arrCSLGBExcessPerArea,return_index=True)
        arrUnique = arrUnique[np.argsort(index)]
        i = 0
        lstIndices = []
        lstGBValues = []
        lstGBValues.append(np.sort(arrUnique))
        lstGBValues.append(np.sort)
        arrSortedIndex = np.sort(index)
        lstRemove = []
        
        if len(arrUnique) > 1:
            while i < len(arrUnique):
                blnRemove = False
                f = 1
                if i == 0:
                    if arrUnique[i] > arrUnique[1]:
                        blnRemove= True
                elif i == len(arrUnique) -1: 
                    if arrUnique[i] > f*arrUnique[i-1]:
                        blnRemove= True
                elif (arrUnique[i] > f*arrUnique[i-1]) or (arrUnique[i] > f*arrUnique[i+1]):
                    blnRemove= True
                if blnRemove:
                    lstRemove.extend(np.where(arrCSLGBExcessPerArea == arrUnique[i])[0].tolist())
                i +=1
        lstIndices = []
        lstMetaStable = list(set(range(len(arrCSLGBExcessPerArea))).difference(lstRemove))
        intCounter = lstMetaStable[0] 
        lstMetaGrouped = []
        for j  in lstMetaStable:
            if j ==intCounter:
                lstIndices.append(j)
                intCounter +=1
            else:
                if len(lstIndices) > 0:
                    lstMetaGrouped.append(lstIndices)
                lstIndices = []
                lstIndices.append(j)
                intCounter = j+1
        if len(lstIndices) >0 and lstIndices not in lstMetaGrouped:
            lstMetaGrouped.append(lstIndices)
        arrTJValues = np.loadtxt(strTJDir + strAxis +'/'  + strTJSigma +'Values.txt')
        strDMINKey = strAxis + ',' + str(intSigma)
        dctDMIN[strDMINKey] = lstMetaGrouped
        for k in range(len(lstMetaGrouped)):
            if k ==2:
                strKey = strDMINKey + ',' + str((k+1)/2)
            else:
                strKey = strDMINKey + ',' + str(k+1)
            arrRows = np.isin(arrTJValues[:,1], lstMetaGrouped[k])
            intLength = len(np.where(arrRows)[0])
            dctAllTJ[strKey] =  TJCalculation(arrTJValues[arrRows],False)/4
            #dctCSLGB[strKey] = arrCSLGBExcessPerArea[np.array(lstMetaGrouped[k])][0]*np.ones(intLength)
            dctCSLGB[strKey] = FindCSLExcess(arrValues[arrRows])
            dctAllGB[strKey] = GBCalculation(arrTJValues[arrRows],arrValues[arrRows], intSigma, strAxis)
            dctDMIN[strKey] = lstMetaGrouped[k]

lstMarkers = ['o','v','s']
lstLegend = []
##excess grain boundary
for i in range(3):
    strCurrentAxis = lstAxes[i]
    strMarker = lstMarkers[i]
    lstX1 = []
    lstY1 = []
    lstX2 = []
    lstY2 = []
    lstX3 = []
    lstY3 = []
    for j in dctCSLGB.keys():
        lstGBKey = j.split(',')
        dList = dctDMIN[j]
        if lstGBKey[0] == strCurrentAxis:
            if lstGBKey[2] == '1':
                lstX1.append(int(lstGBKey[1]))
                lstY1.append(np.mean(dctCSLGB[j])) 
            elif lstGBKey[2] == '1.5':
                lstX3.append(int(lstGBKey[1]))
                lstY3.append(np.mean(dctCSLGB[j]))            
            elif lstGBKey[2] == '2':
                lstX2.append(int(lstGBKey[1]))
                lstY2.append(np.mean(dctCSLGB[j])) 
                
    plt.scatter(lstX1,lstY1, c='red',marker =strMarker,label='Small')
    if len(lstX3) > 0:
        plt.scatter(lstX3,lstY3, c='orange', marker =strMarker,label='Medium')
        plt.scatter(lstX2,lstY2, c='green', marker =strMarker,label='Large')
        lstLegend.extend([r'First $d_{\mathrm{min}}$',r'Middle $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
    elif len(lstX2) > 0:
        plt.scatter(lstX2,lstY2, c='green', marker =strMarker,label='Large')
        lstLegend.extend([r'First $d_{\mathrm{min}}$', r'Last $d_{\mathrm{min}}$'])
    else:
        lstLegend.extend([r'First $d_{\mathrm{min}}$'])
#plt.legend([r'First $d_{\mathrm{min}}$',r'Middle $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
plt.legend(lstLegend)
plt.ylabel('CSL grain boundary excess energy per unit area in eV $\AA^{-2}$')
plt.xlabel('CSL grain boundary $\Sigma$ value')
plt.xticks(list(range(3,31,2)))
plt.tight_layout()
plt.show()

def AppendListsUsingSigmaValues(dct1: dict(), strAxis: str, strKey:str):
    lstX = []
    lstY = []
    for j in dct1.keys():
        lstKey1 = j.split(',')
        if lstKey1[0] == strAxis:
            if lstKey1[2] == strKey:
                lstX.append(int(lstKey1[1])*np.ones(len(dct1[j])))
                lstY.append(dct1[j])
    return lstX,lstY


def CombineListsOfTwoDictionaries(dct1: dict(),dct2: dict(), strAxis = '', strKey = '', blnContatenate=True):
    lstX = []
    lstY = []
    for j in dct1.keys():
        lstKey1 = j.split(',')
        if lstKey1[0] == strAxis or strAxis == '':
            if lstKey1[2] == strKey or strKey == '':    
                lstX.append(dct1[j])
                lstY.append(dct2[j])
    if blnContatenate and len(lstX) > 1:
        lstX = np.concatenate(lstX, axis = 0)
        lstY = np.concatenate(lstY, axis = 0)
    return lstX,lstY

def AppendDictionaryValuesBySigmaValues(dct1: dict, lstOfSigmaValues: list, strAxis: str):
    lstReturn = []
    [lstReturn.append([]) for j in lstOfSigmaValues]
    for j in dct1.keys():
        lstKey = j.split(',')
        if strAxis == lstKey[0]:
            intPosition = lstOfSigmaValues.index(int(lstKey[1]))
            if len(lstReturn[intPosition]) == 0:
                lstReturn[intPosition] = dct1[j]
            else:
                lstReturn[intPosition] = np.append(lstReturn[intPosition],dct1[j], axis=0)
    return lstReturn

def AppendCSLDictionary(dctCSL: dict(), lstOfSigmaValues: list, strAxis: str):
    lstReturn = []
    [lstReturn.append([]) for j in lstOfSigmaValues]
    for j in dctCSL.keys():
        lstKey = j.split(',')
        if strAxis == lstKey[0]:
            intPosition = lstOfSigmaValues.index(int(lstKey[1]))
            if len(lstReturn[intPosition]) == 0:
                lstReturn[intPosition] = dctCSL[j]*np.ones(10)
            else:
                lstReturn[intPosition] = np.append(lstReturn[intPosition],dctCSL[j]*np.ones(10), axis=0)
    return lstReturn

def MapMeanAcrossList(inList: list):
    lstReturn = inList
    for j in lstReturn:
        if len(j) ==0:
            lstReturn.remove(j)
    return list(map(np.mean, lstReturn))

def ReturnMeanValuesInList(inList: list):
    return [np.mean(j[0]) for j in inList]

def ShiftValuesOfList(lstOfArrays: np.array, d: float):
    for j in range(len(lstOfArrays)):
        lstOfArrays[j] += d*np.ones(len(lstOfArrays[j])) 
    return lstOfArrays



lstMod = ['Axis001,13,1','Axis001,25,1','Axis001,29,1','Axis101,27,1','Axis111,7,2','Axis111,21,1']
#comment this section out for original values
for k in lstMod:
    dctAllTJ.pop(k)
    dctCSLGB.pop(k)
    dctDMIN.pop(k)
    dctAllGB.pop(k)
lstAxis101 = [3,9,11,17]
##############################################


intP = 2
strAxis = lstAxes[intP]
strMarker = lstMarkers[intP]
lstColours = ['red','orange','green']

lstSigma001F,lstTJ001F = AppendListsUsingSigmaValues(dctAllTJ, strAxis, '1')
lstSigma001M,lstTJ001M = AppendListsUsingSigmaValues(dctAllTJ, strAxis, '1.5')
lstSigma001L,lstTJ001L = AppendListsUsingSigmaValues(dctAllTJ, strAxis, '2')
lstSigma001F = ShiftValuesOfList(lstSigma001F, -0.2)
lstSigma001L = ShiftValuesOfList(lstSigma001L, 0.2)

plt.scatter(np.concatenate(lstSigma001F,axis=0),np.concatenate(lstTJ001F,axis=0),c=lstColours[0],marker=strMarker)
if len(lstSigma001M)> 0:
    plt.scatter(np.concatenate(lstSigma001M,axis=0),np.concatenate(lstTJ001M,axis=0),c=lstColours[1],marker=strMarker)
    plt.scatter(np.concatenate(lstSigma001L,axis=0),np.concatenate(lstTJ001L,axis=0),c=lstColours[2],marker=strMarker)
    plt.legend([r'First $d_{\mathrm{min}}$',r'Middle $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
elif len(lstSigma001L)>0:
    plt.scatter(np.concatenate(lstSigma001L,axis=0),np.concatenate(lstTJ001L,axis=0),c=lstColours[2],marker=strMarker)
    plt.legend([r'First $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
else:
    plt.legend([r'First $d_{\mathrm{min}}$'])
plt.ylabel('Mean triple line formation energy per unit length in eV $\AA^{-1}$')
plt.xlabel('CSL grain boundary $\Sigma$ value')
plt.xticks(list(range(3,31,2)))

plt.tight_layout()
plt.show()

lstAllGBF,lstAllTJF = CombineListsOfTwoDictionaries(dctCSLGB,dctAllTJ,strAxis,'1',True)
lstAllGBM,lstAllTJM = CombineListsOfTwoDictionaries(dctCSLGB,dctAllTJ,strAxis,'1.5',True)
lstAllGBL,lstAllTJL = CombineListsOfTwoDictionaries(dctCSLGB,dctAllTJ,strAxis,'2',True)

plt.scatter(lstAllGBF,lstAllTJF, c=lstColours[0],marker=strMarker)
if len(lstAllGBM) > 0:
    plt.scatter(lstAllGBM,lstAllTJM, c=lstColours[1],marker=strMarker)
    plt.scatter(lstAllGBL,lstAllTJL, c=lstColours[2],marker=strMarker)
    plt.legend([r'First $d_{\mathrm{min}}$',r'Middle $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
elif len(lstAllGBL) > 0:
    plt.scatter(lstAllGBL,lstAllTJL, c=lstColours[2],marker=strMarker)
    plt.legend([r'First $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
else:
    plt.legend([r'First $d_{\mathrm{min}}$'])
plt.xlabel('CSL grain boundary excess energy per unit area in eV $\AA^{-2}$ in $S_{\mathrm{GB}}$')
plt.ylabel('Mean triple line formation energy per unit length in eV $\AA^{-1}$')
plt.tight_layout()
plt.show()

lstAllGBF,lstAllTJF = CombineListsOfTwoDictionaries(dctAllGB,dctAllTJ,strAxis,'1',True)
lstAllGBM,lstAllTJM = CombineListsOfTwoDictionaries(dctAllGB,dctAllTJ,strAxis,'1.5',True)
lstAllGBL,lstAllTJL = CombineListsOfTwoDictionaries(dctAllGB,dctAllTJ,strAxis,'2',True)

plt.scatter(lstAllGBF,lstAllTJF, c=lstColours[0],marker=strMarker)
if len(lstAllGBM) > 0:
    plt.scatter(lstAllGBM,lstAllTJM, c=lstColours[1],marker=strMarker)
    plt.scatter(lstAllGBL,lstAllTJL, c=lstColours[2],marker=strMarker)
    plt.legend([r'First $d_{\mathrm{min}}$',r'Middle $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
elif len(lstAllGBL) > 0:
    plt.scatter(lstAllGBL,lstAllTJL, c=lstColours[2],marker=strMarker)
    plt.legend([r'First $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
else:
    plt.legend([r'First $d_{\mathrm{min}}$'])
plt.xlabel('Total grain boundary excess energy per unit area in eV $\AA^{-2}$ in $S_{\mathrm{GB}}$')
plt.ylabel('Mean triple line formation energy per unit length in eV $\AA^{-1}$')
plt.tight_layout()
plt.show()


lstAllTJValues = []
lstAllGBValues = []
lstAllTJValues.append(lstAllTJF)
if len(lstAllGBM) > 0:
    lstAllTJValues.append(lstAllTJM[0])
    lstAllGBValues.append(lstAllGBM[0])
lstAllTJValues.append(lstAllTJL)
lstAllGBValues.append(lstAllGBF)
lstAllGBValues.append(lstAllGBL)
lstAllTJValues = np.concatenate(lstAllTJValues,axis=0)
lstAllGBValues = np.concatenate(lstAllGBValues,axis=0)
plt.scatter(lstAllGBValues,lstAllTJValues)
plt.show()
print(np.corrcoef(lstAllGBValues,lstAllTJValues))
plt.hist(lstAllTJValues, bins =20)
plt.show()


lst100TJ = AppendDictionaryValuesBySigmaValues(dctAllTJ, lstAxis001,'Axis001')
lst101TJ = AppendDictionaryValuesBySigmaValues(dctAllTJ, lstAxis101,'Axis101')
lst111TJ = AppendDictionaryValuesBySigmaValues(dctAllTJ, lstAxis111,'Axis111')
lst100GB = AppendDictionaryValuesBySigmaValues(dctAllGB, lstAxis001,'Axis001')
lst101GB = AppendDictionaryValuesBySigmaValues(dctAllGB, lstAxis101,'Axis101')
lst111GB = AppendDictionaryValuesBySigmaValues(dctAllGB, lstAxis111,'Axis111')
arr100TJ = np.concatenate(lst100TJ,axis=0)
arr101TJ = np.concatenate(lst101TJ,axis=0)
arr111TJ = np.concatenate(lst111TJ,axis=0)

print(np.mean(arr100TJ),np.std(arr100TJ),np.mean(arr101TJ),np.std(arr101TJ),np.mean(arr111TJ),np.std(arr111TJ))

lstAll = []
lstAll.extend(lst100TJ)
lstAll.extend(lst101TJ)
lstAll.extend(lst111TJ)

arrAll = np.concatenate(lstAll, axis=0)

print(np.mean(arrAll),np.std(arrAll), len(arrAll), len(arrAll[arrAll < 0])/len(arrAll))


plt.scatter(lstAxis001,MapMeanAcrossList(lst100TJ),marker='o',c='black')
plt.scatter(lstAxis101, MapMeanAcrossList(lst101TJ),marker='v',c='blue')
plt.scatter(lstAxis111,MapMeanAcrossList(lst111TJ),marker='s',c='purple')
plt.xlabel('CSL grain boundary $\Sigma$ value')
plt.xticks(list(range(3,31,2)))
plt.ylabel('Mean triple line formation energy per unit length in eV $\AA^{-1}$')
plt.legend(['Axis 001', 'Axis 101', 'Axis 111'])
plt.tight_layout()
plt.show()




plt.scatter(MapMeanAcrossList(lst100GB),MapMeanAcrossList(lst100TJ),marker='o',c='black')
plt.scatter(MapMeanAcrossList(lst101GB),MapMeanAcrossList(lst101TJ),marker='v',c='blue')
plt.scatter(MapMeanAcrossList(lst111GB),MapMeanAcrossList(lst111TJ),marker='s',c='purple')
plt.xlabel('Total grain boundary excess energy per unit area in eV $\AA^{-2}$ in $S_{\mathrm{GB}}$')
plt.ylabel('Mean triple line formation energy per unit length in eV $\AA^{-1}$')
plt.legend(['Axis 001', 'Axis 101', 'Axis 111'])
plt.tight_layout()
plt.show()

lstGBs = []
lstGBs.extend(MapMeanAcrossList(lst100GB))
lstGBs.extend(MapMeanAcrossList(lst101GB))
lstGBs.extend(MapMeanAcrossList(lst111GB))

lstTJs = []

lstTJs.extend(MapMeanAcrossList(lst100TJ))
lstTJs.extend(MapMeanAcrossList(lst101TJ))
lstTJs.extend(MapMeanAcrossList(lst111TJ))

print(stats.pearsonr(lstGBs,lstTJs))#,nan_policy ='omit'))
# print(stats.pearsonr(np.concatenate(lst100GB,axis=0),np.concatenate(lst100TJ,axis=0)))#,nan_policy ='omit'))
# print(stats.pearsonr(np.concatenate(lst101GB,axis=0),np.concatenate(lst101TJ,axis=0)))#,nan_policy ='omit'))
# print(stats.pearsonr(np.concatenate(lst111GB,axis=0),np.concatenate(lst111TJ,axis=0)))#,nan_policy ='omit'))

print(stats.pearsonr(MapMeanAcrossList(lst100GB),MapMeanAcrossList(lst100TJ)))#,nan_policy ='omit'))
print(stats.pearsonr(MapMeanAcrossList(lst101GB),MapMeanAcrossList(lst101TJ)))#,nan_policy ='omit'))
print(stats.pearsonr(MapMeanAcrossList(lst111GB),MapMeanAcrossList(lst111TJ)))#,nan_policy ='omit'))



lstCSLExcess100 = AppendDictionaryValuesBySigmaValues(dctCSLGB, lstAxis001, 'Axis001')
lstCSLExcess101 = AppendDictionaryValuesBySigmaValues(dctCSLGB, lstAxis101, 'Axis101')
lstCSLExcess111 = AppendDictionaryValuesBySigmaValues(dctCSLGB, lstAxis111, 'Axis111')

plt.scatter(MapMeanAcrossList(lstCSLExcess100),MapMeanAcrossList(lst100TJ),marker='o',c='black')
plt.scatter(MapMeanAcrossList(lstCSLExcess101),MapMeanAcrossList(lst101TJ),marker='v',c='blue')
plt.scatter(MapMeanAcrossList(lstCSLExcess111),MapMeanAcrossList(lst111TJ),marker='s',c='purple')
plt.xlabel('CSL grain boundary excess energy per unit area in eV $\AA^{-2}$ in $S_{\mathrm{GB}}$')
plt.ylabel('Mean triple line formation energy per unit length in eV $\AA^{-1}$')
plt.legend(['Axis 001', 'Axis 101', 'Axis 111'])
plt.tight_layout()
plt.show()
plt.show()

plt.scatter(np.concatenate(lstCSLExcess100,axis=0),np.concatenate(lst100GB,axis=0),marker='o',c='black')
plt.scatter(np.concatenate(lstCSLExcess101,axis=0),np.concatenate(lst101GB,axis=0),marker='v',c='blue')
plt.scatter(np.concatenate(lstCSLExcess111,axis=0),np.concatenate(lst111GB,axis=0),marker='s',c='purple')
plt.xlabel('CSL grain boundary excess energy per unit area in eV $\AA^{-2}$')
plt.ylabel('Total grain boundary excess energy per unit area in eV $\AA^{-2}$ in $S_{\mathrm{GB}}$')
plt.legend(['Axis 001', 'Axis 101', 'Axis 111'])
plt.show()


plt.hist(arrAll, bins=25, density = True)
plt.xlabel('Mean triple line formation energy per unit length in eV $\AA^{-1}$')
plt.ylabel('Proportion')
plt.show()


# # arrOutlier = np.loadtxt(strTJDir + 'Axis001/TJSigma5/Values.txt')
# # arrFinalValues = TJCalculation(arrOutlier)
# # arrRows = np.where(arrFinalValues > 0)
# # print(arrFinalValues[arrRows],arrOutlier[arrRows])

