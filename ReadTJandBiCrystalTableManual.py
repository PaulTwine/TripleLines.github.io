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
from sklearn.ensemble import GradientBoostingRegressor
import MiscFunctions as mf
from pickle import FALSE

## Assumes columns are 2: GBPE, 3: TJPE, 4: GB Lattice Atoms PE 5: TJLattice Atoms, 6: Number of GB atoms, 7 Number of TJ atoms,, 8 Number of PTM GB Atoms, 9 Number of PTM TJ Atoms, 10 is TJ Length
def AdjustEnergyByMassBalance(inArray: np.array):
    arrNMax = np.max([int(inArray[0,8]),int(inArray[0,9])])*np.ones(len(inArray)) 
    arrTJMean = np.divide(inArray[:,5],inArray[:,9])
    arrGBMean = np.divide(inArray[:,4],inArray[:,8])
    arrMu = inArray[0,4]/inArray[0,8]
    arrTJAdjusted = inArray[:,3] + arrMu*(arrNMax-inArray[:,7])
    arrGBAdjusted = inArray[:,2] + arrMu*(arrNMax-inArray[:,6])
    return arrTJAdjusted, arrGBAdjusted, arrNMax  
def TJCalculation(inArray: np.array,blnLocalOnly = False):
    fltLength = inArray[0,-1]
    # arrNMax = np.max([int(inArray[0,8]),int(inArray[0,9])])*np.ones(len(inArray)) 
    # arrTJMean = np.divide(inArray[:,5],inArray[:,9])
    # arrGBMean = np.divide(inArray[:,4],inArray[:,8])
    # #arrTJAdjusted = inArray[:,3] + arrTJMean*(inArray[:,6]-inArray[:,7])
    #arrGBAdjusted  = inArray[:,2]
    # arrMu = inArray[0,4]/inArray[0,8]
    arrTJAdjusted, arrGBAdjusted = AdjustEnergyByMassBalance(inArray)[0:2]
    #arrTJAdjusted = inArray[:,3] + arrMu*(arrNMax-inArray[:,7])
    #arrGBAdjusted = inArray[:,2] + arrMu*(arrNMax-inArray[:,6])
    #arrTJAdjusted = inArray[:,3] + arrTJMean*(arrNMax-inArray[:,7])
    #arrGBAdjusted = inArray[:,2] + arrGBMean*(arrNMax-inArray[:,6])
    #arrTJAdjusted = inArray[:,3] - (-3.36*(arrNMax-inArray[:,7]))
    #arrGBAdjusted = inArray[:,2] - (-3.36*(arrNMax-inArray[:,6]))
    #arrGBExcess = inArray[:,2] + arrGBMean*(arrNMax-inArray[:,6])
    #arrTotalExcessEnergy = arrTJAdjusted - inArray[:,2]
    arrTotalExcessEnergy = arrTJAdjusted-arrGBAdjusted
   # arrTotalExcessEnergy = inArray[:,3]-inArray[:,2] + arrTJMean*(inArray[:,6]-inArray[:,7])
    return arrTotalExcessEnergy/fltLength
    #return (inArray[:,3]-inArray[:,2] +inArray[:,5]*(inArray[:,6]-inArray[:,7]))/inArray[0,-1]
def WeightedGBCalculation(inSGBArray: np.array,inBiArray, intSigma: int, strAxis: str): #returns weighted excess by area, non-weighted by area
    fltGBArea, fltCylinderArea = FindTotalGBArea(intSigma,strAxis)
    #fltExcessGB = inSGBArray[:,2] -3.36*(inSGBArray[0,6]*np.ones(len(inSGBArray))-inSGBArray[:,6]) -(-3.36*inSGBArray[0,6]*np.ones(len(inSGBArray))) #sGB
    arrAtoms = inBiArray[:,-2] - inSGBArray[:,6]
    fltMu = inSGBArray[0,4]/inSGBArray[0,8]
    fltGBExcess = inSGBArray[:,2]-(fltMu*inSGBArray[:,6])
    fltGB = inSGBArray[:,2] + (fltMu*arrAtoms)
    #fltExcess = inArray[:,3] -3.36*(inArray[0,7]*np.ones(len(inArray))-inArray[:,7]) -(-3.36*inArray[0,7]*np.ones(len(inArray))) #sTJ
    fltExcessCSL = FindCSLExcess(inBiArray)
    return (fltGBExcess/(fltGBArea+fltCylinderArea)),(fltExcessCSL/fltGBArea + 2*(fltGB-inBiArray[:,1])/fltCylinderArea)/3
def FindCSLExcess(inCSLArray: np.array):
    #return (inCSLArray[:,1]+(np.ones(len(inCSLArray))*inCSLArray[0,3]-inCSLArray[:,3])*(-3.36) -inCSLArray[0,3]*np.ones(len(inCSLArray))*(-3.36))/(2*inCSLArray[0,4])
    return inCSLArray[:,1] -(-3.36*inCSLArray[:,3]) 
def FindGBExcess(inGBArray: np.array):
    fltGBArea, fltCylinderArea = FindTotalGBArea(intSigma, strAxis)
    return (inGBArray[:,2] -(-3.36*inGBArray[:,6]))/(fltGBArea+fltCylinderArea)
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
    intAtoms = 1.5*10**5
    intAtomsPerCell = 4
    a = 4.05 ##lattice parameter
    h = a*np.round(intHeight/s2,0)
    i = np.sqrt(intAtoms*a/(32*12*intAtomsPerCell*h*np.linalg.det(arrSigmaBasis)))
    i = np.round(i,0).astype('int')   
    r = 2*a*s1*i
    w = 32*a*i*s0
    l = 12*a*i*s1
    return 2*l*h*s2, 4*r*np.pi*h*s2 #there are two planar grain boundaries and two cylindrical grain boundaries
def SECalculation(inArray: np.array):
    fltLength = inArray[0,-1]
    arrTJMean = np.divide(inArray[:,5],inArray[:,7])
   # arrGBMean = np.divide(inArray[:,4],inArray[:,6])
    arrStrainDifference = inArray[:,5]-inArray[:,4] +(inArray[:,8]-inArray[:,9])*arrTJMean
    return arrStrainDifference/fltLength


def PCAAnalysis(xData, yData):
    xCentred = xData -np.mean(xData)
    yCentred = yData -np.mean(yData)
    xCentred = xCentred/np.max(xCentred)
    yCentred = yCentred/np.max(yCentred)
    arrData = np.array([xCentred, yCentred])
    arrCov = np.cov(arrData)
    arrValues,arrVectors = np.linalg.eig(arrCov)
    arrSorted = np.argsort(arrValues)[::-1]
    return arrValues[arrSorted], np.transpose(arrVectors)[arrSorted]

def PCAVectors(xData, yData, fltScale = 1.96):
    arrMean = np.array([np.mean(xData),np.mean(yData)])
    arrValues, arrVectors = PCAAnalysis(xData,yData)
    arrValues = fltScale*np.sqrt(arrValues)
    for i in range(len(arrVectors)):
        arrVectors[i] = np.sqrt(arrValues[i])*arrVectors[i]
    return arrMean,arrVectors
    

def InterceptError(x: np.array, a = 1.96)->np.array:
    fltLower = -(x.intercept-a*x.intercept_stderr)/(x.slope + a*x.stderr)
    fltUpper = -(x.intercept + a*x.intercept_stderr)/(x.slope-a*x.stderr)
    return np.sort(np.array([fltLower,fltUpper]))


# print(gf.CubicCSLGenerator(np.array([1,1,1]),6))


#strCSLAxis = 'CSL grain boundary excess energy \n per unit area in eV $\AA^{-2}$'
strCSLAxis = r'$\gamma_{\mathrm{CSL}}$ in eV $\AA^{-2}$'
strSigmaAxis = 'CSL grain boundary $\Sigma$ value' 
#strTJAxis  =  'Mean triple line formation energy \n per unit length in eV $\AA^{-1}$'
strTJAxis  =  r'$\bar{\lambda}_{\mathrm{TJ}}$ in eV $\AA^{-1}$'
strMeanTJAxis =  r'Mean of $\bar{\lambda}_{\mathrm{TJ}}$ in eV $\AA^{-1}$'
#strNWGBAxis = 'Non-weighted mean grain boundary excess energy \n per unit area in eV $\AA^{-2}$ in $S_{\mathrm{GB}}$'
strNWGBAxis  = r'$\gamma_{\mathrm{NW}}$ in eV $\AA^{-2}$'
strGBAxis  = '$\gamma$ in eV $\AA^{-2}$'
strDMinAxis= '$d_{\mathrm{min}}/r_0$'

lstAxes = ['Axis001', 'Axis101','Axis111']
lstLegendAxes = ['Axis [001]', 'Axis [101]', 'Axis [111]']
lstAxis001 = [5,13,17,29,37]
#lstAxis001 = [5,29]
lstAxis101 = [3,9,11,19,27]
#lstAxis101 = [3,9,11,17]
lstAxis111 = [3,7,13,21,31]
#lstAxis111 = [3,7]
lstAllSigma = []


dctAllTJ = dict()
dctCSLGB = dict()
dctAllGB = dict()
dctCSLMeta = dict()
dctSGBExcess = dict()
dctLists = dict()
dctLists[lstAxes[0]] = lstAxis001
dctLists[lstAxes[1]] = lstAxis101
dctLists[lstAxes[2]] = lstAxis111
for i in lstAxes:
    lstAllSigma.extend(dctLists[i])
lstAllSigma = np.unique(lstAllSigma).tolist()
lstAllSigmaPositions = []



lstMetaResults = []
lstLowerResults = []
lstUpperResults = []
lstGBResults = []
lstMinResults = []

dctDMin = dict()

dctDMin['Axis001,5'] = [list(range(6,9))]
dctDMin['Axis001,13'] = [[6,7]]
dctDMin['Axis001,17'] = [list(range(0,8))]
dctDMin['Axis001,29'] = [list(range(2,8))]
dctDMin['Axis001,37'] = [list(range(5,8))]

dctDMin['Axis101,3'] = [list(range(0,9))]
dctDMin['Axis101,9'] = [list(range(5,9))] #GB periodic unit rearrangement
dctDMin['Axis101,11'] = [list(range(0,8))]
#dctDMin['Axis101,17'] = [list(range(7,8))]
dctDMin['Axis101,19'] = [list(range(0,8))]
dctDMin['Axis101,27'] = [list(range(6,9))] #disconnections nucleated in GB simulation cell

dctDMin['Axis111,3'] = [list(range(6,9))] #disconnections nucleated in GB simulation cell
dctDMin['Axis111,7'] = [list(range(7,9))] #disconnections nucleated in GB simulation cell
dctDMin['Axis111,13'] = [list(range(0,9))] #disconnections nucleated in GB simulatin cell
#dctDMin['Axis111,19'] = [list(range(6,9))] #discconections nucleated in GB simulation
dctDMin['Axis111,21'] = [list(range(6,9))] #distorted grain boundary disconnections in TJ simulation cell cylindrical grain disrupted
dctDMin['Axis111,31'] = [list(range(0,9))]
lstColours = ['darkblue','purple','peru']


for strAxis in lstAxes:
    strBiDir = '/home/p17992pt/csf4_scratch/BiCrystal/'
    strTJDir = '/home/p17992pt/csf4_scratch/TJCylinder/'
    for intSigma in dctLists[strAxis]:
        strBiSigma = 'Sigma' + str(intSigma) + '/'
        strTJSigma = 'TJSigma' + str(intSigma) + '/'
        arrValues = np.loadtxt(strBiDir + strAxis + '/' + strBiSigma + 'Values.txt')
        arrValues = np.tile(arrValues, (10,1))
        fltGBArea = FindTotalGBArea(intSigma, strAxis)[0]
        arrCSLGBExcessPerArea = FindCSLExcess(arrValues)[:10]/fltGBArea 
        arrUnique,index = np.unique(arrCSLGBExcessPerArea,return_index=True)
        arrUnique = np.sort(arrUnique)
        i = 0
        lstIndices = []
        lstGBValues = []
        lstGBValues.append(np.sort(arrUnique))
        #lstGBValues.append(np.sort)
        arrTJValues = np.loadtxt(strTJDir + strAxis +'/'  + strTJSigma +'Values.txt')
        strDMINKey = strAxis + ',' + str(intSigma)
        lstMetaGrouped = dctDMin[strDMINKey] 
        arrRows = np.isin(arrTJValues[:,1], lstMetaGrouped[0])
        dctAllTJ[strDMINKey] = TJCalculation(arrTJValues[arrRows],False)/4
        dctCSLGB[strDMINKey] = FindCSLExcess(arrValues[arrRows])/fltGBArea
        WeightedExcess, nonWeightedExcess = WeightedGBCalculation(arrTJValues[arrRows],arrValues[arrRows], intSigma, strAxis)
        dctAllGB[strDMINKey] = nonWeightedExcess
        dctSGBExcess[strDMINKey] = WeightedExcess
        dctCSLMeta[strDMINKey] = arrUnique
lstMarkers = ['o','v','s']
##excess grain boundary
for i in range(3):
    strCurrentAxis = lstAxes[i]
    lstX1 = []
    lstY1 = []
    for j in dctCSLGB.keys():
        lstGBKey = j.split(',')
        dList = dctDMin[j]
        if lstGBKey[0] == strCurrentAxis:
            #arrX1 = (int(lstGBKey[1])-0.2*(1-i))*np.ones(len(dctCSLMeta[j]))
            lstX1.extend([lstAllSigma.index(int(lstGBKey[1])) for k in dctCSLMeta[j]])
            lstY1.append(dctCSLMeta[j])
    arrY = np.concatenate(lstY1,axis=0)
    plt.scatter(lstX1 + (i-1)*np.ones(len(lstX1))/10, arrY, c=lstColours[i],marker =lstMarkers[i],label='Small')
    
#plt.legend([r'First $d_{\mathrm{min}}$',r'Middle $d_{\mathrm{min}}$',r'Last $d_{\mathrm{min}}$'])
plt.legend(lstLegendAxes)
plt.ylabel(strCSLAxis)
plt.xlabel(strSigmaAxis)
plt.xticks(list(range(len(lstAllSigma))),lstAllSigma)
plt.tight_layout()
plt.show()

def AppendListsUsingSigmaValues(dct1: dict(), strAxis: str):
    lstX = []
    lstY = []
    lstSigma = dctLists[strAxis]
    for j in dct1.keys():
        lstKey1 = j.split(',')
        if lstKey1[0] == strAxis:
            lstX.append(lstSigma.index(int(lstKey1[1]))*np.ones(len(dct1[j])))
            lstY.append(dct1[j])
    return lstX,lstY


def CreateDminAndOtherList(dctDmin: dict(),dct2: dict(), strAxis = ''):
    lstX = []
    lstY = []
    for j in dctDmin.keys():
        lstKey1 = j.split(',')
        if lstKey1[0] == strAxis or strAxis == '': 
            for k in range(10): 
                lstX.extend(dctDmin[j])
            lstY.extend(dct2[j])
    return mf.FlattenList(lstX),lstY



def CombineListsOfTwoDictionaries(dct1: dict(),dct2: dict(), strAxis = '', blnContatenate=True):
    lstX = []
    lstY = []
    for j in dct1.keys():
        lstKey1 = j.split(',')
        if lstKey1[0] == strAxis or strAxis == '':  
            lstX.append(dct1[j])
            lstY.append(dct2[j])
    if blnContatenate and len(lstX) > 1:
        lstX = np.concatenate(lstX, axis = 0)
        lstY = np.concatenate(lstY, axis = 0)
    return lstX,lstY

def AppendDictionaryValuesBySigmaValues(dct1: dict, lstOfSigmaValues: list, strAxis: str):
    lstReturn = []
    lstSigmaPositions = []
    [lstReturn.append([]) for j in lstOfSigmaValues]
    [lstSigmaPositions.append([]) for j in lstOfSigmaValues]
    for j in dct1.keys():
        lstKey = j.split(',')
        intSigmaPosition = lstAllSigma.index(int(lstKey[1]))
        if strAxis == lstKey[0]:
            intPosition = lstOfSigmaValues.index(int(lstKey[1]))
            if len(lstReturn[intPosition]) == 0:
                lstReturn[intPosition] = dct1[j]
                lstSigmaPositions[intPosition] = intSigmaPosition*np.ones(len(dct1[j])) 
            else:
                lstReturn[intPosition] = np.append(lstReturn[intPosition],dct1[j], axis=0)
                lstSigmaPositions[intPosition] = np.append(lstSigmaPositions[intPosition],intSigmaPosition*np.ones(len(dct1[j])), axis=0)
    return lstReturn, lstSigmaPositions

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
def MapStdAcrossList(inList: list):
    lstReturn = inList
    for j in lstReturn:
        if len(j) ==0:
            lstReturn.remove(j)
    return list(map(lambda x : np.std(x,ddof=1), lstReturn))
def MapMinAcrossList(inList: list):
    lstReturn = inList
    for j in lstReturn:
        if len(j) ==0:
            lstReturn.remove(j)
    return list(map(lambda x : np.min(x), lstReturn))

def MapMedianAcrossList(inList: list):
    lstReturn = inList
    for j in lstReturn:
        if len(j) ==0:
            lstReturn.remove(j)
    return list(map(lambda x : np.median(x), lstReturn))



def ReturnMeanValuesInList(inList: list):
    return [np.mean(j[0]) for j in inList]

def ShiftValuesOfList(lstOfArrays: np.array, d: float):
    for j in range(len(lstOfArrays)):
        lstOfArrays[j] += d*np.ones(len(lstOfArrays[j])) 
    return lstOfArrays
def FitLine(xData: np.array, arrPar: np.array):
    return arrPar[0]*xData + arrPar[1]

def ReturnDminMeanAndStd(arrDMin: list , arrTJ: list):
    lstMean = []
    lstStd = []
    arrUniqueD = np.unique(arrDMin)
    for i in arrUniqueD:
        arrRows = np.where(arrDMin == i)[0]
        lstMean.append(np.mean(arrTJ[arrRows]))
        lstStd.append(np.std(arrTJ[arrRows],ddof=1))
    return arrUniqueD, np.array(lstMean), np.array(lstStd)



lstD001D, lstTJ001D = CreateDminAndOtherList(dctDMin,dctAllTJ,'Axis001')
lstD101D, lstTJ101D = CreateDminAndOtherList(dctDMin,dctAllTJ,'Axis101')
lstD111D, lstTJ111D = CreateDminAndOtherList(dctDMin,dctAllTJ,'Axis111')

#lstAllTJ = mf.FlattenList(lstAllTJ)

plt.scatter((np.array(lstD001D)-0.1*np.ones(len(lstD001D)))/10, lstTJ001D,marker=lstMarkers[0],c=lstColours[0])
plt.scatter(np.array(lstD101D)/10, lstTJ101D,marker=lstMarkers[1],c=lstColours[1])
plt.scatter((np.array(lstD111D)+0.1*np.ones(len(lstD111D)))/10, lstTJ111D,marker=lstMarkers[2],c=lstColours[2])
plt.legend(lstLegendAxes)
plt.axhline(y=0,c='black',linestyle='--')
plt.ylabel(strTJAxis)
plt.xlabel(strDMinAxis)
plt.tight_layout()
plt.show()

arrDMin, arrTJMean001, arrTJStd001 = ReturnDminMeanAndStd(np.array(lstD001D), np.array(lstTJ001D))
arrDMin, arrTJMean101, arrTJStd101 = ReturnDminMeanAndStd(np.array(lstD101D), np.array(lstTJ101D))
arrDMin, arrTJMean111, arrTJStd111 = ReturnDminMeanAndStd(np.array(lstD111D), np.array(lstTJ111D))

plt.scatter((arrDMin-0.1*np.ones(len(arrDMin)))/10, arrTJMean001,marker=lstMarkers[0],c=lstColours[0])
plt.scatter(arrDMin/10, arrTJMean101,marker=lstMarkers[1],c=lstColours[1])
plt.scatter((arrDMin+0.1*np.ones(len(arrDMin)))/10, arrTJMean111,marker=lstMarkers[2],c=lstColours[2])
plt.legend(lstLegendAxes)
plt.errorbar((arrDMin-0.1*np.ones(len(arrDMin)))/10, arrTJMean001,arrTJStd001, c=lstColours[0],linestyle ='',capsize=5)
plt.errorbar(arrDMin/10, arrTJMean101,arrTJStd101,c=lstColours[1],linestyle ='',capsize=5)
plt.errorbar((arrDMin+0.1*np.ones(len(arrDMin)))/10, arrTJMean111,arrTJStd111, c=lstColours[2],linestyle ='',capsize=5)
plt.axhline(y=0,c='black',linestyle='--')
plt.ylabel(strMeanTJAxis)
plt.xlabel(strDMinAxis)
plt.tight_layout()
plt.show()



# plt.scatter(MapMeanAcrossList(lst100SP),MapMeanAcrossList(lst100TJ),marker='o',c='black')
# plt.scatter(MapMeanAcrossList(lst101SP), MapMeanAcrossList(lst101TJ),marker='v',c='blue')
# plt.scatter(MapMeanAcrossList(lst111SP),MapMeanAcrossList(lst111TJ),marker='s',c='purple')
# plt.axhline(y=0,c='black',linestyle='--',label='_nolegend_')
# plt.errorbar(MapMeanAcrossList(lst100SP),MapMeanAcrossList(lst100TJ),MapStdAcrossList(lst100TJ), c='black',linestyle ='',capsize=5)
# plt.errorbar(MapMeanAcrossList(lst101SP), MapMeanAcrossList(lst101TJ),MapStdAcrossList(lst101TJ), c='blue',linestyle ='',capsize=5)
# plt.errorbar(MapMeanAcrossList(lst111SP),MapMeanAcrossList(lst111TJ),MapStdAcrossList(lst111TJ),c='purple',linestyle ='',capsize=5)
# plt.xlabel(strSigmaAxis)
# plt.xticks(list(range(len(lstAllSigma))),lstAllSigma)
# plt.ylabel(strMeanTJAxis)
# plt.legend(lstLegendAxes)
# plt.tight_layout()
# plt.show()





intP = 0
strAxis = lstAxes[intP]
strMarker = lstMarkers[intP]


lstSigma001F,lstTJ001F = AppendListsUsingSigmaValues(dctAllTJ, strAxis)

plt.scatter(np.concatenate(lstSigma001F,axis=0),np.concatenate(lstTJ001F,axis=0),c=lstColours[intP],marker=strMarker)
plt.legend([lstLegendAxes[intP]])
plt.ylabel(strTJAxis)
plt.xlabel(strSigmaAxis)
plt.xticks(list(range(len(lstSigma001F))),dctLists[strAxis])
plt.axhline(y=0,c='black',linestyle='--')
plt.tight_layout()
plt.show()
print('Proportion of negative TJS for ' +strAxis,len(np.concatenate(lstTJ001F,axis=0)),len(np.where(np.concatenate(lstTJ001F,axis=0) <0)[0])/len(np.concatenate(lstTJ001F,axis=0)))


lstAllGBF,lstAllTJF = CombineListsOfTwoDictionaries(dctCSLGB,dctAllTJ,strAxis,True)





plt.scatter(lstAllGBF,lstAllTJF, c=lstColours[intP],marker=strMarker)
plt.legend([lstLegendAxes[intP]])
plt.xlabel(strCSLAxis)
plt.ylabel(strTJAxis)
plt.axhline(y=0,c='black',linestyle='--')
plt.tight_layout()
plt.show()

lstAllGBF,lstAllTJF = CombineListsOfTwoDictionaries(dctAllGB,dctAllTJ,strAxis,True)
arrOut = stats.linregress(lstAllGBF,lstAllTJF)
print(arrOut)

x,y, ci,pi = gf.ConfidenceAndPredictionBands(lstAllGBF,lstAllTJF,0.95)


arrResiduals = lstAllTJF -FitLine(lstAllGBF,arrOut)
stats.probplot(arrResiduals, dist="norm", plot=plt)
plt.show()

plt.title('Residuals for ' + strAxis)
plt.hist(arrResiduals,bins =25)
plt.show()


#arrMean, arrVectors = PCAVectors(lstAllGBF,lstAllTJF,1.96)
#arrQuiver = ([0,0],[1,1],[-1,-2],[1,1])
#plt.plot(lstAllGBF,FitLine(lstAllGBF, arrOut),c='green')
#plt.quiver(*arrMean,*arrVectors[0], scale_units='xy',scale=1,color='red')
#plt.quiver(*arrMean,*arrVectors[1],scale_units='xy',scale=1,color='orange')
#plt.quiver(*arrMean,*(-arrVectors[0]), scale_units='xy',scale=1,color='red')
#plt.quiver(*arrMean,*(-arrVectors[1]),scale_units='xy',scale=1,color='orange')
plt.plot(x,y,c='green',label='_nolegend_',linestyle='--')
if intP !=1:
    plt.plot(x,y+pi,c='red',label='_nolegend_',linestyle='--')
    plt.plot(x,y-pi,c='red',label='_nolegend_',linestyle='--')
    plt.fill_between(x,y-ci,y+ci,color='gray',label='_nolegend_')
plt.scatter(lstAllGBF,lstAllTJF, c=lstColours[intP],marker=strMarker)
plt.legend([lstLegendAxes[intP]])
plt.xlabel(strNWGBAxis)
plt.ylabel(strTJAxis)
plt.axhline(y=0,c='black',linestyle='--')
plt.tight_layout()
plt.show()



lstAllTJValues = []
lstAllGBValues = []
lstAllTJValues.append(lstAllTJF)
lstAllGBValues.append(lstAllGBF)
lstAllTJValues = np.concatenate(lstAllTJValues,axis=0)
lstAllGBValues = np.concatenate(lstAllGBValues,axis=0)
plt.scatter(lstAllGBValues,lstAllTJValues)
plt.tight_layout()
plt.show()
print('Individual results for Pearons ' +strAxis,stats.pearsonr(lstAllGBValues,lstAllTJValues))
print('Individual results for Spearman ' +strAxis,stats.spearmanr(lstAllGBValues,lstAllTJValues))
plt.hist(lstAllTJValues, bins =20)
plt.show()
plt.scatter(lstAllGBValues,lstAllTJValues)
plt.show()


lst100TJ,lst100SP = AppendDictionaryValuesBySigmaValues(dctAllTJ, lstAxis001,'Axis001')
lst101TJ,lst101SP = AppendDictionaryValuesBySigmaValues(dctAllTJ, lstAxis101,'Axis101')
lst111TJ, lst111SP = AppendDictionaryValuesBySigmaValues(dctAllTJ, lstAxis111,'Axis111')
lst100GB, lst100SP = AppendDictionaryValuesBySigmaValues(dctAllGB, lstAxis001,'Axis001')
lst101GB, lst101SP = AppendDictionaryValuesBySigmaValues(dctAllGB, lstAxis101,'Axis101')
lst111GB, lst111SP = AppendDictionaryValuesBySigmaValues(dctAllGB, lstAxis111,'Axis111')
# lst100GB, lst100SP = AppendDictionaryValuesBySigmaValues(dctSGBExcess, lstAxis001,'Axis001')
# lst101GB, lst101SP = AppendDictionaryValuesBySigmaValues(dctSGBExcess, lstAxis101,'Axis101')
# lst111GB, lst111SP = AppendDictionaryValuesBySigmaValues(dctSGBExcess, lstAxis111,'Axis111')
arr100TJ = np.concatenate(lst100TJ,axis=0)
arr101TJ = np.concatenate(lst101TJ,axis=0)
arr111TJ = np.concatenate(lst111TJ,axis=0)

plt.scatter(np.concatenate(ShiftValuesOfList(np.array(lst100SP),-0.1),axis=0), arr100TJ,marker=lstMarkers[0], c=lstColours[0])
plt.scatter(np.concatenate(lst101SP,axis=0), arr101TJ,marker=lstMarkers[1], c=lstColours[1])
plt.scatter(np.concatenate(ShiftValuesOfList(np.array(lst111SP),+0.1),axis=0), arr111TJ,marker=lstMarkers[2], c=lstColours[2])
plt.xticks(list(range(len(lstAllSigma))),lstAllSigma)
plt.xlabel(strSigmaAxis)
plt.ylabel(strTJAxis)
plt.legend(lstLegendAxes)
plt.axhline(y=0,c='black',linestyle='--')
plt.tight_layout()
plt.show()


print('Axis001',np.mean(arr100TJ),np.std(arr100TJ),'Axis101',np.mean(arr101TJ),np.std(arr101TJ),'Axis111',np.mean(arr111TJ),np.std(arr111TJ))

lstAllTJ = []
lstAllTJ.extend(lst100TJ)
lstAllTJ.extend(lst101TJ)
lstAllTJ.extend(lst111TJ)

lstAllGB = []
lstAllGB.extend(lst100GB)
lstAllGB.extend(lst101GB)
lstAllGB.extend(lst111GB)

arrAllTJ = np.concatenate(lstAllTJ, axis=0)
arrAllGB = np.concatenate(lstAllGB, axis=0)



plt.scatter(arrAllGB,arrAllTJ)
plt.show()
arrOut = stats.linregress(arrAllGB,arrAllTJ)
print(arrOut)
print(InterceptError(arrOut))

#arrMean, arrVectors = PCAVectors(arrAllGB,arrAllTJ,1.96)


x,y, ci,pi = gf.ConfidenceAndPredictionBands(arrAllGB,arrAllTJ,0.95)

plt.plot(x,y,c='green',label='_nolegend_',linestyle='--')
plt.plot(x,y+pi,c='red',label='_nolegend_',linestyle='--')
plt.plot(x,y-pi,c='red',label='_nolegend_',linestyle='--')
plt.fill_between(x,y-ci,y+ci,color='gray',label='_nolegend_')
plt.scatter(np.concatenate(lst100GB,axis=0),np.concatenate(lst100TJ,axis=0),marker='o',c=lstColours[0])
plt.scatter(np.concatenate(lst101GB,axis=0),np.concatenate(lst101TJ,axis=0),marker='v',c=lstColours[1])
plt.scatter(np.concatenate(lst111GB,axis=0),np.concatenate(lst111TJ,axis=0),marker='s',c=lstColours[2])
plt.legend(lstLegendAxes)
plt.axhline(y=0,c='black',linestyle='--')
# plt.quiver(*arrMean,*arrVectors[0], scale_units='xy',scale=1,color='red')
# plt.quiver(*arrMean,*arrVectors[1],scale_units='xy',scale=1,color='orange')
# plt.plot(arrAllGB,FitLine(arrAllGB, arrOut),c='green')
plt.xlabel(strNWGBAxis)
plt.ylabel(strTJAxis)
plt.tight_layout()
plt.show()
print("all data",stats.pearsonr(arrAllGB,arrAllTJ))

np.savetxt('/home/p17992pt/LAMMPSData/AllTJ.txt',arrAllTJ,delimiter=',')
np.savetxt('/home/p17992pt/LAMMPSData/AllGB.txt',arrAllGB,delimiter=',')


arrResiduals = arrAllTJ-FitLine(arrAllGB,arrOut)
stats.probplot(arrResiduals, dist="norm", plot=plt)
plt.show()
print(stats.shapiro(arrResiduals),np.mean(arrResiduals))
plt.hist(arrResiduals,bins =25)
plt.show()


print(np.mean(arrAllTJ),np.std(arrAllTJ), len(arrAllTJ), len(arrAllTJ[arrAllTJ < 0])/len(arrAllTJ))


plt.scatter(MapMeanAcrossList(lst100SP),MapMeanAcrossList(lst100TJ),marker='o',c=lstColours[0])
plt.scatter(MapMeanAcrossList(lst101SP), MapMeanAcrossList(lst101TJ),marker='v',c=lstColours[1])
plt.scatter(MapMeanAcrossList(lst111SP),MapMeanAcrossList(lst111TJ),marker='s',c=lstColours[2])
plt.axhline(y=0,c='black',linestyle='--',label='_nolegend_')
plt.errorbar(MapMeanAcrossList(lst100SP),MapMeanAcrossList(lst100TJ),MapStdAcrossList(lst100TJ), c=lstColours[0],linestyle ='',capsize=5)
plt.errorbar(MapMeanAcrossList(lst101SP), MapMeanAcrossList(lst101TJ),MapStdAcrossList(lst101TJ), c=lstColours[1],linestyle ='',capsize=5)
plt.errorbar(MapMeanAcrossList(lst111SP),MapMeanAcrossList(lst111TJ),MapStdAcrossList(lst111TJ),c=lstColours[2],linestyle ='',capsize=5)
plt.xlabel(strSigmaAxis)
plt.xticks(list(range(len(lstAllSigma))),lstAllSigma)
plt.ylabel(strMeanTJAxis)
plt.legend(lstLegendAxes)
plt.tight_layout()
plt.show()




# plt.scatter(MapMedianAcrossList(lst100GB),MapMedianAcrossList(lst100TJ),marker='o',c='black')
# plt.scatter(MapMedianAcrossList(lst101GB), MapMedianAcrossList(lst101TJ),marker='v',c='blue')
# plt.scatter(MapMedianAcrossList(lst111GB),MapMedianAcrossList(lst111TJ),marker='s',c='purple')
# plt.errorbar(MapMeanAcrossList(lst100SP),MapMeanAcrossList(lst100TJ),MapStdAcrossList(lst100TJ), c='black')
# plt.errorbar(MapMeanAcrossList(lst101SP), MapMeanAcrossList(lst101TJ),MapStdAcrossList(lst101TJ), c='blue')
# plt.errorbar(MapMeanAcrossList(lst111SP),MapMeanAcrossList(lst111TJ),MapStdAcrossList(lst111TJ),c='purple')
plt.xlabel(strSigmaAxis)
#plt.xticks(list(range(len(lstAllSigma))),lstAllSigma)
plt.ylabel(strMeanTJAxis)
plt.legend(lstLegendAxes)
plt.tight_layout()
plt.show()


plt.scatter(MapMeanAcrossList(lst100GB),MapMeanAcrossList(lst100TJ),marker='o',c=lstColours[0])
plt.scatter(MapMeanAcrossList(lst101GB),MapMeanAcrossList(lst101TJ),marker='v',c=lstColours[1])
plt.scatter(MapMeanAcrossList(lst111GB),MapMeanAcrossList(lst111TJ),marker='s',c=lstColours[2])
plt.axhline(y=0,c='black',linestyle='--',label='_nolegend_')
plt.errorbar(MapMeanAcrossList(lst100GB),MapMeanAcrossList(lst100TJ),MapStdAcrossList(lst100TJ),MapStdAcrossList(lst100GB), c=lstColours[0],linestyle ='',capsize=5)
plt.errorbar(MapMeanAcrossList(lst101GB), MapMeanAcrossList(lst101TJ),MapStdAcrossList(lst101TJ),MapStdAcrossList(lst101GB), c=lstColours[1],linestyle ='',capsize=5)
plt.errorbar(MapMeanAcrossList(lst111GB),MapMeanAcrossList(lst111TJ),MapStdAcrossList(lst111TJ),MapStdAcrossList(lst111GB),c=lstColours[2],linestyle ='',capsize=5)
plt.xlabel(strNWGBAxis)
#plt.xticks(list(range(len(lstAllSigma))),lstAllSigma)
plt.ylabel(strMeanTJAxis)
plt.legend(lstLegendAxes)
plt.tight_layout()
plt.show()


plt.xlabel(strNWGBAxis)
plt.ylabel(strMeanTJAxis)
plt.legend(lstLegendAxes)
plt.tight_layout()
plt.show()

# lstGBs = []
# lstGBs.extend(MapMeanAcrossList(lst100GB))
# lstGBs.extend(MapMeanAcrossList(lst101GB))
# lstGBs.extend(MapMeanAcrossList(lst111GB))

# lstTJs = []

# lstTJs.extend(MapMeanAcrossList(lst100TJ))
# lstTJs.extend(MapMeanAcrossList(lst101TJ))
# lstTJs.extend(MapMeanAcrossList(lst111TJ))

# print(stats.pearsonr(lstGBs,lstTJs))#,nan_policy ='omit'))
# plt.scatter(lstGBs,lstTJs)
# plt.show()

# print(stats.pearsonr(MapMeanAcrossList(lst100GB),MapMeanAcrossList(lst100TJ)))#,nan_policy ='omit'))
# print(stats.pearsonr(MapMeanAcrossList(lst101GB),MapMeanAcrossList(lst101TJ)))#,nan_policy ='omit'))
# print(stats.pearsonr(MapMeanAcrossList(lst111GB),MapMeanAcrossList(lst111TJ)))#,nan_policy ='omit'))



lstWeightedExcess100 = AppendDictionaryValuesBySigmaValues(dctSGBExcess, lstAxis001, 'Axis001')[0]
lstWeightedExcess101 = AppendDictionaryValuesBySigmaValues(dctSGBExcess, lstAxis101, 'Axis101')[0]
lstWeightedExcess111 = AppendDictionaryValuesBySigmaValues(dctSGBExcess, lstAxis111, 'Axis111')[0]

# plt.scatter(MapMeanAcrossList(lstCSLExcess100),MapMeanAcrossList(lst100TJ),marker='o',c='black')
# plt.scatter(MapMeanAcrossList(lstCSLExcess101),MapMeanAcrossList(lst101TJ),marker='v',c='blue')
# plt.scatter(MapMeanAcrossList(lstCSLExcess111),MapMeanAcrossList(lst111TJ),marker='s',c='purple')
# plt.xlabel(strCSLAxis)
# plt.ylabel(strMeanTJAxis)
# plt.legend(lstLegendAxes)
# plt.axhline(y=0,c='black',linestyle='--')
# plt.tight_layout()
# plt.show()


plt.scatter(np.concatenate(lstWeightedExcess100,axis=0),np.concatenate(lst100GB,axis=0),marker='o',c=lstColours[0])
plt.scatter(np.concatenate(lstWeightedExcess101,axis=0),np.concatenate(lst101GB,axis=0),marker='v',c=lstColours[1])
plt.scatter(np.concatenate(lstWeightedExcess111,axis=0),np.concatenate(lst111GB,axis=0),marker='s',c=lstColours[2])
plt.xlabel(str)
plt.ylabel(strNWGBAxis)
plt.legend(lstLegendAxes)
plt.plot(plt.ylim(),plt.ylim(), c='black',scalex=FALSE,scaley=FALSE, linestyle='--')
plt.tight_layout()
plt.show()


plt.hist(arrAllTJ,bins=25, density = True) 
plt.xlabel(strTJAxis)
plt.ylabel('Proportion')
plt.axvline(x=0,c='black',linestyle='--',label='_nolegend_')
plt.tight_layout()
plt.show()


# # arrOutlier = np.loadtxt(strTJDir + 'Axis001/TJSigma5/Values.txt')
# # arrFinalValues = TJCalculation(arrOutlier)
# # arrRows = np.where(arrFinalValues > 0)
# # print(arrFinalValues[arrRows],arrOutlier[arrRows])

