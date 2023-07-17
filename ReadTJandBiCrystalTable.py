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
from scipy import stats


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
        #arrGBExcessPerArea = (arrValues[:,1]+(np.ones(len(arrValues))*arrValues[0,3]-arrValues[:,3])*arrValues[:,2] -arrValues[0,3]*arrValues[:,2])/arrValues[0,4]
        arrGBExcessPerArea = (arrValues[:,1]+(np.ones(len(arrValues))*arrValues[0,3]-arrValues[:,3])*(-3.36) -arrValues[0,3]*np.ones(len(arrValues))*(-3.36))/arrValues[0,4] 
        arrUnique,index = np.unique(arrGBExcessPerArea,return_index=True)
        arrUnique = arrUnique[np.argsort(index)]
        i = 0
        lstIndices = []
        lstGBValues = []
        lstGBValues.append(np.sort(arrUnique))
        lstGBValues.append(np.sort)
        arrSortedIndex = np.sort(index)
        # while i < len(arrSortedIndex)+1:
        #     lstIndices.append(list(range(arrSortedIndex[i],arrSortedIndex[i+1])))
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
                    lstRemove.extend(np.where(arrGBExcessPerArea == arrUnique[i])[0].tolist())
                i +=1
        lstIndices = []
        lstMetaStable = list(set(range(len(arrValues))).difference(lstRemove))
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
        # if len(lstMetaGrouped) ==0:
        #     lstMetaGrouped.append(lstMetaStable)
        arrTJValues = np.loadtxt(strTJDir + strAxis +'/'  + strTJSigma +'Values.txt')
        strDMINKey = strAxis + ',' + str(intSigma)
        dctDMIN[strDMINKey] = lstMetaGrouped
        for k in range(len(lstMetaGrouped)):
            strKey = strDMINKey + ',' + str(k+1)
            arrRows = np.isin(arrTJValues[:,1], lstMetaGrouped[k])
            dctAllTJ[strKey] =  TJCalculation(arrTJValues[arrRows],False)/4
            dctAllGB[strKey] = arrGBExcessPerArea[np.array(lstMetaGrouped[k])]


        lstMetaStable = list(set(range(len(arrValues))).difference(lstRemove))
        arrMin = np.where(arrGBExcessPerArea == np.min(arrGBExcessPerArea))[0]
        intFirst = lstMetaStable[0]
        i = 1
        lstLowerValues = []
        lstLowerValues.append(intFirst)
        lstUpperValues = []
        blnStop = False
        while  i < len(lstMetaStable) and not(blnStop):
            if lstMetaStable[i] ==intFirst+1:
                lstLowerValues.append(lstMetaStable[i])
                intFirst = lstMetaStable[i]
            else:
                blnStop = True
            i +=1 
        if len(lstLowerValues) == len(lstMetaStable):
            lstUpperValues = lstLowerValues
            lstLowerValues = []
        else:
            lstUpperValues = list(set(lstMetaStable).difference(lstLowerValues))

        arrTJValues = np.loadtxt(strTJDir + strAxis +'/'  + strTJSigma +'Values.txt')
        arrGBExcess = np.tile(arrGBExcessPerArea,(10,1)).reshape(100,1)
        arrMetaRows = np.isin(arrTJValues[:,1],  lstMetaStable)
        arrLowerRows = np.isin(arrTJValues[:,1],  lstLowerValues)
        arrUpperRows = np.isin(arrTJValues[:,1],  lstUpperValues)
        lstMetaResults.append([strAxis[:-1], intSigma,TJCalculation(arrTJValues[arrMetaRows],False)/4,arrGBExcess[arrMetaRows]])
        lstMinResults.append([strAxis[:-1], intSigma,TJCalculation(arrTJValues[arrMin],False)/4,arrGBExcess[arrMin]])
        if np.any(arrLowerRows == True) :
            lstLowerResults.append([strAxis[:-1], intSigma,TJCalculation(arrTJValues[arrLowerRows],False)/4,arrGBExcess[arrLowerRows]])
        if np.any(arrUpperRows == True):
            lstUpperResults.append([strAxis[:-1], intSigma,TJCalculation(arrTJValues[arrUpperRows],False)/4,arrGBExcess[arrUpperRows]])
        
        
      

lstValues001 = []
lstValues101 = []
lstValues111 = []
lstSigma001 = []
lstSigma101 =[]
lstSigma111= []
lstValues001U = []
lstValues101U = []
lstValues111U = []
lstSigma001U = []
lstSigma101U =[]
lstSigma111U= []
lstValues001L = []
lstValues101L = []
lstValues111L = []
lstSigma001L = []
lstSigma101L =[]
lstSigma111L= []
lstGBExcess001 = []
lstGBExcess101 = []
lstGBExcess111 = []
lstGBExcess001L = []
lstGBExcess101L = []
lstGBExcess111L = []
lstGBExcess001U = []
lstGBExcess101U = []
lstGBExcess111U = []
lstMin001 = []
lstMin101 = []
lstMin111 = []
lstMinSigma001 = []
lstMinSigma101 = []
lstMinSigma111 = []
d= 0.2
for j in lstMetaResults:
    if j[0] == 'Axis001':
        lstValues001.append(j[2])
        lstSigma001.append(j[1]*np.ones(len(j[2])))
        lstGBExcess001.append(j[3])
    elif j[0] == 'Axis101': #and int(j[1]) !=11:
        lstValues101.append(j[2])
        lstSigma101.append(j[1]*np.ones(len(j[2])))
        lstGBExcess101.append(j[3])
    elif j[0] == 'Axis111':
        lstValues111.append(j[2])
        lstSigma111.append(j[1]*np.ones(len(j[2])))
        lstGBExcess111.append(j[3])
for j in lstUpperResults:
    if j[0] == 'Axis001':
        lstValues001U.append(j[2])
        lstSigma001U.append((j[1]-d)*np.ones(len(j[2])))
        lstGBExcess001U.append(j[3])
    elif j[0] == 'Axis101': #and int(j[1]) !=11:
        lstValues101U.append(j[2])
        lstSigma101U.append((j[1]-d)*np.ones(len(j[2])))
        lstGBExcess101U.append(j[3])
    elif j[0] == 'Axis111':
        lstValues111U.append(j[2])
        lstSigma111U.append((j[1]-d)*np.ones(len(j[2])))
        lstGBExcess111U.append(j[3])
for j in lstLowerResults:
    if j[0] == 'Axis001':
        lstValues001L.append(j[2])
        lstSigma001L.append((j[1]+d)*np.ones(len(j[2])))
        lstGBExcess001L.append(j[3])
    elif j[0] == 'Axis101': #and int(j[1]) !=11:
        lstValues101L.append(j[2])
        lstSigma101L.append((j[1]+d)*np.ones(len(j[2])))
        lstGBExcess101L.append(j[3])
    elif j[0] == 'Axis111':
        lstValues111L.append(j[2])
        lstSigma111L.append((j[1]+d)*np.ones(len(j[2])))
        lstGBExcess111L.append(j[3])
for j in lstMinResults:
    if j[0] == 'Axis001':
        lstMin001.append(j[2])
        lstMinSigma001.append(j[1]*np.ones(len(j[2])))
    elif j[0] == 'Axis101': #and int(j[1]) !=11:
        lstMin101.append(j[2])
        lstMinSigma101.append(j[1]*np.ones(len(j[2])))
    elif j[0] == 'Axis111':
        lstMin111.append(j[2])
        lstMinSigma111.append(j[1]*np.ones(len(j[2])))


plt.scatter(list(map(np.mean,lstGBExcess001)),list(map(np.mean,lstValues001)))
plt.show()
#arrUpperValues = np.append(np.append(np.concatenate(lstValues001L),np.concatenate(lstValues101L),axis=0),np.concatenate(lstValues111L),axis=0)
#arrUpperValues = np.append(np.concatenate(lstUpperResults,axis=0))
#plt.hist(arrUpperValues,bins=25)
#### All the 

#plt.scatter(list(map(np.mean,lstGBExcess111U)),list(map(np.mean,lstValues111U)))
#plt.show()

# plt.scatter(np.concatenate(lstMinSigma001),np.concatenate(lstMin001))
# plt.scatter(np.concatenate(lstMinSigma101,axis=0),np.concatenate(lstMin101,axis=0))
# plt.scatter(np.concatenate(lstMinSigma111,axis=0),np.concatenate(lstMin111,axis=0))
# plt.show()



plt.scatter(np.round(list(map(np.unique, lstSigma001L))),list(map(np.mean,(lstValues001L))),marker='o', c='r')
plt.scatter(np.round(list(map(np.unique, lstSigma101L))),list(map(np.mean,(lstValues101L))),marker='v', c='r')
plt.scatter(np.round(list(map(np.unique, lstSigma111L))),list(map(np.mean,(lstValues111L))),marker='s',c='r')
plt.scatter(np.round(list(map(np.unique, lstSigma001U))),list(map(np.mean,(lstValues001U))),marker='o', c='g')
plt.scatter(np.round(list(map(np.unique, lstSigma101U))),list(map(np.mean,(lstValues101U))),marker='v', c='g')
plt.scatter(np.round(list(map(np.unique, lstSigma111U))),list(map(np.mean,(lstValues111U))),marker='s',c='g')
plt.scatter(lstAxis001, list(map(np.mean,lstValues001)),marker="o",c='y')
plt.scatter(lstAxis101, list(map(np.mean,lstValues101)),marker='v',c='y')
plt.scatter(lstAxis111, list(map(np.mean,lstValues111)),marker='s',c='y')
plt.xlim([1,31])
plt.ylim([-0.7,0.9])
plt.xticks(list(range(3,31,2)))
plt.legend([r'[001] Small $d_{\mathrm{min}}$',r'[101] Small $d_{\mathrm{min}}$',r'[111] Small $d_{\mathrm{min}}$',r'[001] Large $d_{\mathrm{min}}$',r'[101] Large $d_{\mathrm{min}}$',r'[111] Large $d_{\mathrm{min}}$',r'[001] Both $d_{\mathrm{min}}$',r'[101] Both $d_{\mathrm{min}}$',r'[111] Both $d_{\mathrm{min}}$'],loc = 'upper center', ncol=3)
plt.ylabel('Triple line excess energy per unit length in eV $\AA^{-1}$')
plt.xlabel('CSL grain boundary $\Sigma$ value')
# plt.scatter(lstAxis001, list(map(np.mean,lstValues001U)))
# plt.scatter(lstAxis101, list(map(np.mean,lstValues101U)))
# plt.scatter(lstAxis111, list(map(np.mean,lstValues111U)))
plt.show()

# plt.scatter(lstAxis001, list(map(np.mean,lstValues001U)))
# plt.scatter(lstAxis101, list(map(np.mean,lstValues101U)))
# plt.scatter(lstAxis111, list(map(np.mean,lstValues111U)))
# plt.xlim([2,31])
# plt.xticks(list(range(3,31,2)))
# plt.legend(['[001]','[101]','[111]'])
# # plt.scatter(lstAxis001, list(map(np.mean,lstValues001U)))
# # plt.scatter(lstAxis101, list(map(np.mean,lstValues101U)))
# # plt.scatter(lstAxis111, list(map(np.mean,lstValues111U)))
# plt.show()



# plt.scatter(np.round(list(map(np.unique, lstSigma001U))),list(map(np.mean,(lstValues001U))))
# plt.scatter(np.round(list(map(np.unique, lstSigma101U))),list(map(np.mean,(lstValues101U))))
# plt.scatter(np.round(list(map(np.unique, lstSigma111U))),list(map(np.mean,(lstValues111U))))
# plt.ylim([-0.2,0])
# #plt.scatter(lstAxis001,list(map(np.mean,(lstValues001U))))
# #plt.scatter(lstAxis101,list(map(np.mean,(lstValues101U))))
# #plt.scatter(lstAxis111,list(map(np.mean,(lstValues111U))))
# plt.legend(['[001]','[101]','[111]'])
# plt.show()
plt.scatter(np.concatenate(lstSigma001),np.concatenate(lstValues001),c='g',marker='s')
plt.scatter(np.concatenate(lstSigma001),np.concatenate(lstValues001),c='r',marker='s')

plt.ylabel('Triple line excess energy per unit length in eV $\AA^{-1}$')
plt.xlabel('CSL grain boundary $\Sigma$ value')
plt.xlim([2,31])
plt.legend(['Large $d_{min}$ values','Small $d_{min}$ values'])
# #plt.ylim([-0.9,.25])
plt.xticks(list(range(3,31,2)))
plt.show()

# lstFinalValues = []
# lstFinalValues.append(np.concatenate(lstValues001U,axis=0))
# lstFinalValues.append(lstValues001L[0])
# for i in [0,2,3,4]:
#     lstFinalValues.append(lstValues101U[i])
# lstFinalValues.append(lstValues101L[1])
# for k in [0,2,3,4]:
#     lstFinalValues.append(lstValues111U[k])
# lstFinalValues.append(np.concatenate(lstValues111L,axis=0))
# arrFinalValues = np.concatenate(lstFinalValues,axis=0)
# arrRows = np.where(np.abs(arrFinalValues) > 0.55)[0]
# arrFinalValues = np.delete(arrFinalValues,arrRows, axis=0)
# ae, loce, scalee = stats.skewnorm.fit(arrFinalValues)
# xrange = np.linspace(min(arrFinalValues),max(arrFinalValues),100)
# pdf = stats.skewnorm.pdf(xrange,ae, loce, scalee)
# plt.plot(xrange,pdf,'k',linewidth=2)
# plt.hist(arrFinalValues,bins=20,density=True)
# plt.xlabel('Triple line excess energy per unit length in eV $\AA^{-1}$')
# plt.ylabel('Proportion')
# plt.legend(['Skew-normal fitted curve','Triple line data'])
# plt.show()
# print(np.mean(arrFinalValues),np.std(arrFinalValues),len(arrFinalValues))



lstFinal001 = []
lstFinal101 = []
lstFinal111 = []
lstGB001 = []
lstGB101 = []
lstGB111 = []
lstFinal001S = []
lstFinal101S = []
lstFinal111S = []


lstAllValues = []
lstAllGBExcess=[]

for i in [0,1,2]: #Axis001
    lstFinal001.append(np.mean(lstValues001[i]))
    lstFinal001S.append(lstAxis001[i])
    lstAllValues.append(lstValues001[i])
    lstAllGBExcess.append(lstGBExcess001[i])
for j in [3,4]:
    lstFinal001.append(np.mean(lstValues001U[j]))
    lstFinal001S.append(lstAxis001[j])
    lstAllValues.append(lstValues001U[j])
    lstAllGBExcess.append(lstGBExcess001U[j])
for i in [0,1,3,4]: #Axis101
    arrValues101 = lstValues101[i]
    arrGBExcess101 = lstGBExcess101[i]
    arrRows = np.where((arrValues101 > -0.6) & (arrValues101 < 0.2))[0]
    arrValues101 = arrValues101[arrRows]
    lstFinal101.append(np.mean(arrValues101))
    lstFinal101S.append(lstAxis101[i])
    lstAllValues.append(arrValues101)
    lstAllGBExcess.append(arrGBExcess101[arrRows])
for j in [2]:
    lstFinal101.append(np.mean(lstValues101U[j]))
    lstFinal101S.append(lstAxis101[j])
    lstAllValues.append(lstValues101U[j])
    lstAllGBExcess.append(lstGBExcess101U[j])
for i in [0,2,3,4]: #Axis111
    arrValues111 = lstValues111[i]
    arrGBExess111 = lstGBExcess111[i]
    arrRows = np.where(arrValues111 < 0.5)[0]
    arrValues111 = arrValues111[arrRows]
    lstFinal111.append(np.mean(arrValues111))
    lstFinal111S.append(lstAxis111[i])
    lstAllValues.append(arrValues111)
    lstAllGBExcess.append(arrGBExess111[arrRows])

lstFinal111.append(np.mean(lstValues111L[0]))
lstFinal111S.append(lstAxis111[1])
lstAllValues.append(lstValues111L[0])
lstAllGBExcess.append(lstGBExcess111L[0])
#plt.scatter(list(map(np.mean,lstAllGBExcess)),list(map(np.mean,lstAllValues)))
#plt.show()
arrAllValues = np.concatenate(lstAllValues,axis=0) 
#arrRows = np.where(np.abs(arrAllValues) > 0.55)[0]
#arrAllValues = np.delete(arrAllValues,arrRows, axis=0)
ae, loce, scalee = stats.skewnorm.fit(arrAllValues)
xrange = np.linspace(min(arrAllValues),max(arrAllValues),100)
pdf = stats.skewnorm.pdf(xrange,ae, loce, scalee)
plt.plot(xrange,pdf,'k',linewidth=2)
plt.hist(arrAllValues,bins=20,density=True)
plt.show()
plt.scatter(lstFinal001S,lstFinal001,c='black', marker='o')
plt.scatter(lstFinal101S,lstFinal101,c='blue', marker='v')
plt.scatter(lstFinal111S,lstFinal111,c='purple',marker = 's')
# plt.scatter(np.round(list(np.unique(lstSigma001L[0]))),[np.mean(lstValues001L[0])],marker='o', c='r')
# plt.scatter(np.round(list(map(np.unique, lstSigma101L))),list(map(np.mean,(lstValues101L))),marker='v', c='r')
# plt.scatter(np.round(list(map(np.unique, lstSigma111L))),list(map(np.mean,(lstValues111L))),marker='s',c='r')
# plt.scatter(np.round(list(map(np.unique, lstSigma001U))),list(map(np.mean,(lstValues001U))),marker='o', c='g')
# plt.scatter(np.round(list(map(np.unique, lstSigma101U))),list(map(np.mean,(lstValues101U))),marker='v', c='g')
# plt.scatter(np.round(list(map(np.unique, lstSigma111U))),list(map(np.mean,(lstValues111U))),marker='s',c='g')
# plt.scatter(lstAxis001, list(map(np.mean,lstValues001)),marker="o",c='y')
# plt.scatter(lstAxis101, list(map(np.mean,lstValues101)),marker='v',c='y')
# plt.scatter(lstAxis111, list(map(np.mean,lstValues111)),marker='s',c='y')
plt.xlim([1,31])
#plt.ylim([-0.2,0.0])
plt.xticks(list(range(3,31,2)))
plt.legend(['Axis 001', 'Axis 101', 'Axis 111'])
#plt.legend([r'[001] Small $d_{\mathrm{min}}$',r'[101] Small $d_{\mathrm{min}}$',r'[111] Small $d_{\mathrm{min}}$',r'[001] Large $d_{\mathrm{min}}$',r'[101] Large $d_{\mathrm{min}}$',r'[111] Large $d_{\mathrm{min}}$',r'[001] Both $d_{\mathrm{min}}$',r'[101] Both $d_{\mathrm{min}}$',r'[111] Both $d_{\mathrm{min}}$'],loc = 'upper center', ncol=3)
plt.ylabel('Triple line excess energy per unit length in eV $\AA^{-1}$')
plt.xlabel('CSL grain boundary $\Sigma$ value')
# plt.scatter(lstAxis001, list(map(np.mean,lstValues001U)))
# plt.scatter(lstAxis101, list(map(np.mean,lstValues101U)))
# plt.scatter(lstAxis111, list(map(np.mean,lstValues111U)))
plt.show()
print(np.mean(arrAllValues),np.std(arrAllValues), len(arrAllValues))



# arrOutlier = np.loadtxt(strTJDir + 'Axis001/TJSigma5/Values.txt')
# arrFinalValues = TJCalculation(arrOutlier)
# arrRows = np.where(arrFinalValues > 0)
# print(arrFinalValues[arrRows],arrOutlier[arrRows])

