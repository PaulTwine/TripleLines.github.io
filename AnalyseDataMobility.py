# %%
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
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from matplotlib import animation
import MiscFunctions as mf
from sklearn.cluster import DBSCAN


# %%
def FitCurve(x, a,b,c):
    return a*x +b*np.sqrt(x)+c 
#%%
def DiffFitCurve(x,a,b):
    return a -1/2*b*x**(-1/2)
#%%
def FitProportional(x,a):
    return a*x
#%%
def FitLine(x, a, b):
    return a*x + b 
#%%
arrVolume = np.loadtxt('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp750/u01/TJ/VolumeTJ.txt')
objLog = LT.LAMMPSLog('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp750/u01/TJ/TJ.log')
#%%
intEnd = 80
arrPE = objLog.GetValues(1)[100:10*intEnd:10,2]
print(len(arrPE),len(arrVolume[1,10:intEnd]))
popt,pop = optimize.curve_fit(FitLine, arrVolume[1,10:intEnd],arrPE)
plt.plot(arrVolume[1,10:intEnd],FitLine(arrVolume[1,10:intEnd],*popt))
plt.scatter(arrVolume[1,10:intEnd],arrPE)
#plt.scatter(objLog.GetValues(1)[100:,0],objLog.GetValues(1)[100:,2])
print(popt)
#%%
plt.scatter(arrVolume[0,:],arrVolume[1,:])
plt.show()
# %%
#strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis221/Sigma9_9_9/Temp'
strRoot = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
#strRoot = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
lstTemp = [450,550, 650]
lstTemp = [450,550,650]
lstU = [0.02,0.03,0.04,0.05,0.06,0.07,0.08]
#lstU = [0.02, 0.03,0.04,0.05,0.06,0.07,0.08]
dctTJ = dict()
strType = 'TJ'
for T in lstTemp:
    for u in lstU:
        strU = str(u).split('.')[1]
        strDir = strRoot + str(T) + '/u' + strU + '/' + strType + '/'
        strLogFile = strDir + strType + '.log'
        objLog = LT.LAMMPSLog(strLogFile)
        print(strU, T, objLog.GetColumnNames(1))

# %%
def PopulateTJDictionary(strRoot: str, lstTemp: list, lstU: list, strType: str) -> dict():
    dctReturn = dict()
    for T in lstTemp:
        for u in lstU:
            strU = str(u).split('.')[1]
            strDir = strRoot + str(T) + '/u' + strU + '/' + strType + '/'
            print(strDir)
            strLogFile = strDir + strType + '.log'
            strVolumeFile = strDir + 'Volume' + strType + '.txt'
            print(strLogFile)
            objData = LT.LAMMPSData(
                strDir + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
            objAnalysis = objData.GetTimeStepByIndex(-1)
            arrCellVectors = objAnalysis.GetCellVectors()
            arrLog = LT.LAMMPSLog(strLogFile)
            print(strVolumeFile)
            arrVolume = np.loadtxt(strVolumeFile)
            objCSLMobility = gf.CSLMobility(
                arrCellVectors, arrLog.GetValues(1), arrVolume, strType, T, u)
            #objCSLMobility.FindMobilities(1, 4*4.05)
            dctReturn[str(T) + ',' + strU] = objCSLMobility
            #objLog = objCSLMobility.GetLogObject()
            # print(objLog.GetColumnNames(1),strU,str(T))
    return dctReturn
#%%
def PopulateGBDictionary(strRoot: str, lstTemp: list, lstU: list, strType1: str, strType2: str,arrTJCellVectors: np.array) -> dict():
    dctReturn = dict()
    for T in lstTemp:
        for u in lstU:
            strU = str(u).split('.')[1]
            strDir1 = strRoot + str(T) + '/u' + strU + '/' + strType1 + '/'
            strDir2 = strRoot + str(T) + '/u' + strU + '/' + strType2 + '/'
            strLogFile1 = strDir1 + strType1 + '.log'
            strLogFile2 = strDir2 + strType2 + '.log'
            strVolumeFile1 = strDir1 + 'Volume' + strType1 + '.txt'
            strVolumeFile2 = strDir2 + 'Volume' + strType2 + '.txt'
            fltArea = np.linalg.norm(np.cross(arrTJCellVectors[0],arrTJCellVectors[2]))
            arrLog1 = LT.LAMMPSLog(strLogFile1).GetValues(1)
            arrVolume1 = np.loadtxt(strVolumeFile1)
            arrLog2 = LT.LAMMPSLog(strLogFile2).GetValues(1)
            arrVolume2 = np.loadtxt(strVolumeFile2)
            tupShape = np.shape(arrLog1)
            arrLog = np.zeros(tupShape)
            arrLog[:,0] = arrLog1[:,0]
            arrLog[:,1] = (arrLog1[:,1] + arrLog2[:,1])/2
            arrLog[:,2] = arrLog1[:,2] + arrLog2[:,2]
            arrLog[:,3] = arrLog1[:,3] + arrLog2[:,3]
            arrLog[:,4] = (arrLog1[:,4] + arrLog2[:,4])/2
            intLength = np.min([len(arrVolume1[0,:]),len(arrVolume2[0,:])])
            arrVolume = arrVolume1[:,:intLength] + arrVolume2[:,:intLength]
            arrVolume[2,:] = arrVolume[1,:intLength]/fltArea
            arrVolume[0,:] = arrVolume1[0,:intLength]
            objCSLMobility = gf.CSLMobility(
                arrTJCellVectors, arrLog, arrVolume, strType1, T, u)
            #objCSLMobility.FindMobilities(1, 4*4.05)
            dctReturn[str(T) + ',' + strU] = objCSLMobility
            #objLog = objCSLMobility.GetLogObject()
            # print(objLog.GetColumnNames(1),strU,str(T))
    return dctReturn

# %%
arrPoints1 = np.loadtxt(strRoot + '450/u005/TJ/Mesh23TJ0.txt')
plt.scatter(*tuple(zip(*arrPoints1)))
plt.show()
plt.clf()
plt.cla()
plt.close()
# %%
strRoot7_7_49 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strRoot7_7_49 = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'

strRoot21_21_49 = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp'

strRootR = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp'

strRoot9_9_9 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis221/Sigma9_9_9/Temp'

# lstTemp = [450,475, 500,525, 550,575,600,625, 650]

lstU = [0.005,0.0075, 0.01,0.0125, 0.015,0.0175, 0.02]
strType = 'TJ'
strRoot7_7_49 = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
#dctTJR = PopulateTJDictionary(strRootR, lstTemp, lstU, 'TJ')
#%%
def MakeTJandBCDictionaries(strDir, lstinTemp, lstinU):
    dctTJ = PopulateTJDictionary(strDir,lstinTemp,lstinU, 'TJ')
    dct12BV = PopulateTJDictionary(strDir,lstinTemp,lstinU, '12BV')
    dct13BV = PopulateTJDictionary(strDir,lstinTemp,lstinU, 'TJ')
    return dctTJ,dct12BV,dct13BV
#%%
strRoot9_9_9 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp'
dctTJ, dctBV12,dctBV13 = MakeTJandBCDictionaries(strRoot9_9_9,lstTemp,lstU)


#%%
strRoot7_7_49 = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
dctTJ7 = PopulateTJDictionary(strRoot7_7_49, lstTemp, lstU, 'TJ')
dct12BV7 = PopulateTJDictionary(strRoot7_7_49, lstTemp, lstU, '12BV') 
dct13BV7 = PopulateTJDictionary(strRoot7_7_49, lstTemp, lstU, '13BV') 
#%%

dctTJ21 = PopulateTJDictionary(strRoot21_21_49, lstTemp, lstU, 'TJ')
dct12BV21 = PopulateTJDictionary(strRoot21_21_49, lstTemp, lstU, '12BV') 
dct13BV21 = PopulateTJDictionary(strRoot21_21_49, lstTemp, lstU, '13BV') 
#%%
strRoot9_9_9 = '/home/paul/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp'
dctTJ9 = PopulateTJDictionary(strRoot9_9_9, lstTemp, lstU, 'TJ')
#%%
dct12BV9 = PopulateTJDictionary(strRoot9_9_9, lstTemp, lstU, '12BV') 
#%%
dct13BV9 = PopulateTJDictionary(strRoot9_9_9, lstTemp, lstU, '13BV')
#%%
arrLog = LT.LAMMPSLog(strRoot9_9_9 + '550/u01/13BV/13BV.log')

#dctGB7 = PopulateGBDictionary(strRoot, lstTemp, lstU, '12BV','13BV',dctTJ7['450,005'].GetCellVectors())
#%%
lstU = [0.005, 0.01, 0.015, 0.02]
dctGBR = PopulateGBDictionary(strRootR, lstTemp, lstU, '12BV','13BV', dctTJ['450,005'].GetCellVectors())
dctTJR = PopulateTJDictionary(strRootR, lstTemp, lstU, 'TJ')
#%%

def QuickMobilityEstimate(strRoot: str, lstTemp: list, lstU: list, strType1: str) -> dict():
    dctReturn = dict()
    for T in lstTemp:
        for u in lstU:
            strU = str(u).split('.')[1]
            strDir1 = strRoot + str(T) + '/u' + strU + '/' + strType1 + '/'
            strLogFile1 = strDir1 + strType1 + '.log'
            strVolumeFile1 = strDir1 + 'Volume' + strType1 + '.txt'
            arrLog1 = LT.LAMMPSLog(strLogFile1).GetValues(1)
            arrVolume1 = np.loadtxt(strVolumeFile1)
            fltArea = arrVolume[0,1]/arrVolume[0,2]
            objRange = slice(100,200,1)
            lstVnOut,lstdUBydTOut = DoubleBootstrapEstimate(arrVolumeSpeed[0,objRange],-arrVolumeSpeed[2,objRange],arrLogValues[objRange,0],-arrLogValues[objRange,2],1000)
            #objLog = objCSLMobility.GetLogObject()
            # print(objLog.GetColumnNames(1),strU,str(T))
    return dctReturn

#%%
def PartitionByTemperature(dctAny: dict(),intTemp, uLower: float, uUpper: float):
    lstVn = []
    lstU = []
    for a in dctAny.keys():
        if (dctAny[a].GetTemp() == intTemp) and (dctAny[a].GetPEParameter() <= uUpper) and (dctAny[a].GetPEParameter() >= uLower):
           # intFinish = dctAny[a].GetLowVolumeCutOff(1,4*4.05)
           # intStart = int(intFinish/2)
           # dctAny[a].SetLinearRange(intStart,intFinish)
            objRange = dctAny[a].GetLinearRange()
            arrVolumeSpeed = dctAny[a].GetVolumeSpeed()
            arrLogValues =  dctAny[a].GetLogValues()
            #arrLogValues = arrLogValues[intStart:intFinish]
            # lstValuesU = BootstrapEstimate(arrLogValues[:,0],-arrLogValues[:,2], 1000)
            # lstValuesVn = BootstrapEstimate(arrVolumeSpeed[0,objRange],arrVolumeSpeed[2,objRange],1000)
            # lstVn.append(np.mean(lstValuesVn))
            # lstU.append(np.mean(lstValuesU)/np.mean(lstValuesVn))
            # #arrRows = dctAny[a].GetOverlapRows(1)
            #arrLogValues = arrLogValues[arrRows]
            lstVnOut,lstUOut = DoubleBootstrapEstimate(arrVolumeSpeed[0,objRange],-arrVolumeSpeed[2,objRange],
            arrVolumeSpeed[1,objRange],arrLogValues[objRange,2],10**4)
            #intFinish - intStart)
            popt,pop = optimize.curve_fit(FitLine,arrVolumeSpeed[0,objRange],arrVolumeSpeed[2,objRange])
            plt.scatter(arrVolumeSpeed[0,objRange],arrVolumeSpeed[2,objRange])
            plt.plot(arrVolumeSpeed[0,objRange], FitLine(arrVolumeSpeed[0,objRange],*popt),c='black')
            plt.title('Distance vs Time' + str(intTemp)+ str(dctAny[a].GetPEParameter()))
            plt.show()
            popt2,pop2 = optimize.curve_fit(FitLine,arrVolumeSpeed[1,objRange],arrLogValues[objRange,2])
            plt.scatter(arrVolumeSpeed[1,objRange],arrLogValues[objRange,2])
            plt.title('PE vs Volume' + str(intTemp)+str(dctAny[a].GetPEParameter()))
            plt.show()
          #  lstU.append(popt2[0])
            #lstU.append(lstUOut)
            lstU.append(lstUOut)
            lstVn.append(lstVnOut)
    return lstU,lstVn
#%%
def GetVolumeOrLAMMPSLog(inCSL: gf.CSLMobility, lstVolume: list, lstLAMMPS: list,intStart =100):
    arrLogValues = inCSL.GetLogValues()
    arrVolumeSpeed = inCSL.GetVolumeSpeed()
    intMax = np.min([len(arrLogValues[:,0]),len(arrVolumeSpeed[0])])
    if len(lstLAMMPS) == 0:
        x = arrVolumeSpeed[lstVolume[0],100:intMax]
        y = arrVolumeSpeed[lstVolume[1],100:intMax]
    elif len(lstVolume) == 0:
        x = arrLogValues[100:intMax,lstLAMMPS[0]]
        y = arrLogValues[100:intMax,lstLAMMPS[1]]
    else:
        x = arrVolumeSpeed[lstVolume[0],100:intMax]
        y = arrLogValues[100:intMax,lstLAMMPS[0]]
    return x,y
#%%
lstLegend = []
lstpopt = []
lstpopt2 = []
for k in lstU:
    strKey = '450,' + str(k).split('.')[1]
    x,y = PlotDistanceTime(dctTJ7[strKey], [0,2],[])
    popt,pop = optimize.curve_fit(FitLine,x,y)
    #plt.scatter(x,y)
    plt.plot(x,FitLine(x, *popt))
    lstLegend.append(k)
    lstpopt.append(popt[0]*4.05**3/4)
    x2,y2 = PlotDistanceTime(dctTJ7[strKey], [1],[2])
    popt2,pop2 = optimize.curve_fit(FitLine,x2,y2)
    lstpopt2.append(popt2[0])    
plt.legend(lstLegend)
plt.show()
plt.scatter(np.array(lstU)*4/(4.05**3),-np.array(lstpopt))
plt.scatter(np.array(lstpopt2),-np.array(lstpopt))
plt.legend(['Synthetic','Calculated'])
plt.show()


#%%
def WriteMobilityValues(lstInTemp, dctAny: dict,uLower: float, uUpper: float):
    lstMobility = []
    lstMobilityStd = []
    for j in lstInTemp:
        tupValues = PartitionByTemperature(dctAny,j,uLower,uUpper)
        for i in range(len(tupValues[0])):
                plt.title(str(j))
                plt.scatter(tupValues[0][i], tupValues[1][i])
        #plt.scatter(tupValues[0],tupValues[1])
        plt.ylim([0,0.005])
        plt.xlim([0,0.08*4*4.05**(-3)])
        plt.show()

        #popt,pop = optimize.curve_fit(FitLine,tupValues[0],tupValues[1])
        #plt.plot(np.array(tupValues[0]),FitLine(np.array(tupValues[0]),popt[0],popt[1]))
        #lstMobility.append(popt[0])
        lstValues = BlockBootstrapEstimate(tupValues[0],tupValues[1])
        #lstValues = BootstrapEstimate(tupValues[0],tupValues[1],100)
        lstMobility.append(np.mean(lstValues[0]))
        lstMobilityStd.append(1.96*np.std(lstValues[0]))
    return lstMobility, lstMobilityStd
#%%
lstNewTemp = [450,475,500,525,550,575,600,625,650]
lstMobTJ7,lstErrorTJ7 = WriteMobilityValues(lstNewTemp, dctTJ7,lstU[0],lstU[3])
#lstMobTJ21 = WriteMobilityValues(lstNewTemp, dctTJ21)
#%%
lstMob12BV,lstError12BV = WriteMobilityValues(lstNewTemp, dct12BV7,lstU[0],lstU[3])
lstMob13BV,lstError13BV = WriteMobilityValues(lstNewTemp, dct13BV7,lstU[0],lstU[3])
#%%
#lstMobGB = WriteMobilityValues(lstNewTemp, dctGB7)
lstMobBVs = []
lstMobBVs.append(lstMob12BV)
lstMobBVs.append(lstMob13BV)
arrBV = np.vstack(lstMobBVs)
arrMins = np.min(arrBV, axis=0) 
#%%
lstMobTJ,lstMobErrorTJ = WriteMobilityValues(lstTemp,dctTJ7, lstU[0],lstU[3])
lstMob12BV,lstMobError12BV = WriteMobilityValues(lstTemp,dct12BV7, lstU[0],lstU[3])
lstMob13BV,lstMobError13BV = WriteMobilityValues(lstTemp,dct13BV7, lstU[0],lstU[3])
#%%
PlotMobilities(lstTemp, lstMobTJ,lstMob12BV,lstMob13BV,lstMobErrorTJ,lstMobError12BV,lstMobError13BV)
#%%
lstMobTJ,lstMobErrorTJ = WriteMobilityValues(lstTemp,dctTJ21, lstU[0],lstU[2])
#%%
lstMob12BV,lstMobError12BV = WriteMobilityValues(lstTemp,dct12BV21, lstU[0],lstU[2])
#%%
lstMob13BV,lstMobError13BV = WriteMobilityValues(lstTemp,dct13BV21, lstU[0],lstU[2])
#%%
PlotMobilities(lstTemp, lstMobTJ7,[],[],lstErrorTJ7,[],[])
#%%
PlotMobilities(lstTemp, lstMobTJ7,lstMob12BV7,lstMob13BV,lstMobErrorTJ,lstError12BV7,lstError13BV)
#%%
#%%
def PlotMobilities(lstTemp,lstTJ,lst12BV, lst13BV,lstTJE,lst12BVE,lst13BVE):
    lstLegend = []
    if len(lstTJ) > 0:
        plt.scatter(lstTemp, lstTJ)
        plt.errorbar(lstTemp,lstTJ,lstTJE)
        lstLegend.append('TJ')
#plt.scatter(lstNewTemp,lstMobGB)
#plt.scatter(lstNewTemp,lstMobTJ21)
    if len(lst12BV) > 0:
        plt.scatter(lstTemp, lst12BV)
        plt.errorbar(lstTemp,lst12BV,lst12BVE)
        lstLegend.append('1,2 BC')
    if len(lst13BV) > 0:
        plt.scatter(lstTemp, lst13BV)
        plt.errorbar(lstTemp,lst13BV,lst13BVE)
        lstLegend.append('1,3 BC')
#plt.scatter(lstNewTemp,arrMins)
    plt.legend(lstLegend)
    plt.xlabel('Temperature in K')
    plt.ylabel('Mobility in $\AA^4$ eV$^{-1}$ fs$^{-1}$')
#plt.legend(['TJ 7-7-49', 'TJ 21-21-49'])
#plt.legend(['TJ','Min of 12BV 13BV'])
#plt.ylim([0.1,0.5])
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

# plt.scatter(arrMins,lstMobTJ)
# plt.show()
# print(np.corrcoef(arrMins[1:-1],lstMobTJ[1:-1]))
#%%
np.savetxt('/home/p17992pt/S999UAllBV12.txt',np.array(lstMob12BV))
#%%
lstTemp = [450,475,500,525,550,575,600,625,650]
lstTJ = np.loadtxt('/home/p17992pt/S999U015TJ.txt')
lstTJE = np.loadtxt('/home/p17992pt/S999U015TJE.txt')
lst12BV = np.loadtxt('/home/p17992pt/S999U015BV12.txt')
lst12BVE = np.loadtxt('/home/p17992pt/S999U015BV12E.txt')
lst13BV = np.loadtxt('/home/p17992pt/S999U015BV13.txt')
lst13BVE = np.loadtxt('/home/p17992pt/S999U015BV13E.txt')
#%%
PlotArrhenius(lstNewTemp,lstMobTJ9)
#%%
def PlotArrhenius(inlstTemp, inlstMob):
    arrITemp = 1/np.array(inlstTemp)
    arrLogMob = np.array(np.log(inlstMob))
    plt.scatter(arrITemp, arrLogMob)
    popt,pop = optimize.curve_fit(FitLine,arrITemp,arrLogMob)
    plt.xlabel('Inverse temperature K$^{-1}$')
    plt.ylabel('$\ln(M)$')
    plt.plot(arrITemp, FitLine(arrITemp,*popt))
    plt.xlim([np.min(arrITemp)-0.0001,np.max(arrITemp)+0.0001])
    plt.tight_layout()
    plt.show()
    print(popt)
#%%
#%%
def BlockBootstrapEstimate(lstX,lstY):
    lstValues = []
    lstAllX = []
    lstAllY = []
    intN =  min(list(map(lambda x: len(x),lstX)))
    for i in range(len(lstX)):
        inX = lstX[i]
        inY = lstY[i]
        arrPositions = mf.BootStrapRows(intN,1)[0]
        arrX = np.array(inX)[arrPositions]
        arrY = np.array(inY)[arrPositions]
        lstAllX.append(arrX)
        lstAllY.append(arrY)
    arrAllX = np.vstack(lstAllX)
    arrAllY = np.vstack(lstAllY)
    lstValues.append(list(map(lambda k:optimize.curve_fit(FitLine,arrAllX[:,k],arrAllY[:,k])[0][0],list(range(intN)))))
    return lstValues
#%%
def BootstrapEstimate(inX,inY, intN):
    lstValues = []
    arrPositions = mf.BootStrapRows(len(inX),intN)
    lstValues = list(map(lambda k:optimize.curve_fit(FitLine,np.array(inX)[k],np.array(inY)[k])[0][0],arrPositions))
    # for k in arrPositions:
        # popt,pop = optimize.curve_fit(FitLine,np.array(inX)[k],np.array(inY)[k])
        # lstValues.append(popt[0])
    return lstValues
#%%
def DoubleBootstrapEstimate(inX1,inY1,inX2,inY2, intN):
    arrPositions = mf.BootStrapRows(len(inX1),intN)
    lstValues1 = list(map(lambda k:optimize.curve_fit(FitLine,np.array(inX1)[k],np.array(inY1)[k])[0][0],arrPositions))
    lstValues2 = list(map(lambda k:optimize.curve_fit(FitLine,np.array(inX2)[k],np.array(inY2)[k])[0][0],arrPositions))
    return lstValues1,lstValues2
#%
#%%

#%%
class AnimateGBs(object):
    def __init__(self, strDir: str, arrCellVectors: np.array):
        self.__strRoot = strDir
        self.__CellVectors = arrCellVectors
        self.__blnTJ = True
        self.__bln12 = True
        self.__bln13 = True
        self.__ScatterSize = 0.5
    def Animate(self,i):
        self.__ax.clear()
        intStep = 500*i
        objPeriodicTree1 = gf.PeriodicWrapperKDTree(arrPoints12, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors),4*4.05,['p','p','p'])
        if self.__blnTJ:
            strDir = self.__strRoot + 'TJ/'
            arrPoints12 = np.loadtxt(strDir + 'Mesh12TJ' + str(intStep) + '.txt')
            objPeriodicTree12 = gf.PeriodicWrapperKDTree(arrPoints12, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors),4*4.05,['p','p','p'])
            arrPoints13 = np.loadtxt(strDir + 'Mesh13TJ' + str(intStep) + '.txt')
            objPeriodicTree13 = gf.PeriodicWrapperKDTree(arrPoints13, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors),4*4.05,['p','p','p'])
            arrPoints23 = np.loadtxt(strDir + 'Mesh23TJ' + str(intStep) + '.txt')
            objPeriodicTree23 = gf.PeriodicWrapperKDTree(arrPoints23, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors),4*4.05,['p','p','p'])
            # self.__ax.scatter(*tuple(zip(*arrPoints12)),c='b')
            # self.__ax.scatter(*tuple(zip(*arrPoints13)),c='b')
            # self.__ax.scatter(*tuple(zip(*arrPoints23)),c='b')
            self.__ax.scatter(*tuple(zip(*objPeriodicTree12.GetExtendedPoints())),c='b')
            self.__ax.scatter(*tuple(zip(*objPeriodicTree13.GetExtendedPoints())),c='b')
            self.__ax.scatter(*tuple(zip(*objPeriodicTree23.GetExtendedPoints())),c='b')
        if self.__bln12:
            strDir = self.__strRoot + '12BV/'
            arrPoints12 = np.loadtxt(strDir + 'Mesh1212BV' + str(intStep) + '.txt')
            arrPoints12 = arrPoints12 + self.__CellVectors[0]/2
            self.__ax.scatter(*tuple(zip(*arrPoints12)),c='r')
        if self.__bln13:
            strDir = self.__strRoot + '13BV/'
            arrPoints13 = np.loadtxt(strDir + 'Mesh1213BV' + str(intStep) + '.txt')
            self.__ax.scatter(*tuple(zip(*arrPoints13)),c='g')
    def WriteFile(self,strFilename: str, blnTJ: bool, bln12: bool, bln13: bool,intFrames):
        self.__blnTJ = blnTJ
        self.__bln12 = bln12
        self.__bln13 = bln13
        fig,ax = plt.subplots()
        self.__ax = ax
        ani = animation.FuncAnimation(fig, self.Animate,interval=500, frames=intFrames) 
        writergif = animation.PillowWriter(fps=10)
        ani.save(strFilename,writer=writergif)

#%%
class AnimateTJs(object):
    def __init__(self, strDir: str, arrCellVectors: np.array):
        self.__strRoot = strDir
        self.__CellVectors = arrCellVectors
        self.__blnTJ = True
        self.__bln12 = True
        self.__bln13 = True
        self.__ScatterSize = 0.5
        self.__bln3d = False
        self.__TimeStep = 500
    def SetTimeStep(self, inTimeStep):
        self.__TimeStep = inTimeStep
    def GetTimeStep(self):
        return self.__TimeStep
    def FindTripleLines(self, intStep):
        strDir = self.__strRoot + 'TJ/'
        arrPoints12 = np.loadtxt(strDir + 'Mesh12TJ' + str(intStep) + '.txt')
        arrPoints13 = np.loadtxt(strDir + 'Mesh13TJ' + str(intStep) + '.txt')
        arrPoints23 = np.loadtxt(strDir + 'Mesh23TJ' + str(intStep) + '.txt')
        lstAllMeshPoints = []
        lstAllMeshPoints.append(arrPoints12)
        lstAllMeshPoints.append(arrPoints13)
        lstAllMeshPoints.append(arrPoints23)
        intPos = 0
        for k in lstAllMeshPoints:
            clustering = DBSCAN(2*4.05).fit(k)
            arrLabels = clustering.labels_
            arrUniqueLabels,arrCounts = np.unique(arrLabels,return_counts=True)
            arrRows1 = np.where(arrCounts > 10)[0]
            arrRows2 = np.where(np.isin(arrLabels, arrUniqueLabels[arrRows1]))[0]
            lstAllMeshPoints[intPos] = k[arrRows2]
            intPos +=1
        intTJs = len(lstAllMeshPoints)
        lstAllTJMesh = []
        for i in range(intTJs):
            lstOverlapIndices = []
            intCount = 0
            objTreei = gf.PeriodicWrapperKDTree(lstAllMeshPoints[i],self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors),2*4.05,['p','p','p'])                    
            for j in range(intTJs):
                if i !=j:
                    objTreej = gf.PeriodicWrapperKDTree(lstAllMeshPoints[j],self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors),2*4.05,['p','p','p'])
                    arrIndices,arrDistances= objTreei.Pquery_radius(objTreej.GetExtendedPoints(),4*4.05)
                    lstIndices = mf.FlattenList(arrIndices)
                    if len(lstIndices) > 0:
                        if len(lstOverlapIndices) > 0:
                            lstOverlapIndices =list(set(lstOverlapIndices).intersection(set(lstIndices)))
                            intCount += 1
                        else:
                            lstOverlapIndices = lstIndices
                    else:
                        print("Missing mesh points")
            if intCount == 1:
                arrIndices = objTreei.GetPeriodicIndices(lstOverlapIndices)
                #arrIndices = mf.FlattenList(arrIndices)
                #arrPoints = objTreei.GetExtendedPoints()[lstOverlapIndices,:]
                arrPoints = objTreei.GetOriginalPoints()[arrIndices,:]
                arrPoints = np.unique(arrPoints, axis=0)
                lstAllTJMesh.append(arrPoints)
        lstTripleLines = self.GroupTripleLines(np.vstack(lstAllTJMesh))
        return lstTripleLines
    def GroupTripleLines(self, arrPoints: np.array):
        lstTripleLines = []
        objTreePositions = gf.PeriodicWrapperKDTree(self.__OriginalPositions,self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors),4*4.05,['p','p','p'])
        arrDistances,arrIndices = objTreePositions.Pquery(arrPoints, k =1)
        arrIndices = np.array(mf.FlattenList(arrIndices))
        lstIndices = objTreePositions.GetPeriodicIndices(arrIndices)
        arrPeriodicIndices = np.array(lstIndices)
        arrExtendedPoints = objTreePositions.GetExtendedPoints()
        arrOriginalPoints = objTreePositions.GetOriginalPoints() 
        arrOriginalIndices = objTreePositions.Pquery(arrOriginalPoints)
        arrOriginalIndices = np.unique(mf.FlattenList(arrOriginalIndices))
        #for j in range(len(self.__OriginalPositions)):
        for j in arrOriginalIndices.astype('int'):
            arrRows = np.where(arrPeriodicIndices == j)[0]
            if len(arrRows):
                arrNewIndices = arrIndices[arrRows]
                arrNewPoints = arrPoints[arrRows]
                arrNewExtendedPoints = arrExtendedPoints[arrNewIndices,:]
                #lstTripleLines.append(arrNewExtendedPoints)
                arrTranslations = arrNewExtendedPoints - arrOriginalPoints[j] #from the original point to the extended point
                arrTranslations[:,2] = np.zeros(len(arrTranslations))
                arrReturn = np.unique(arrNewPoints-arrTranslations,axis=0)
                lstTripleLines.append(arrReturn) #move points back to the position closest to the original triple line positions
            else:
                print('error frame ' + str(self.__strRoot))
        return lstTripleLines
    def FindMeanTripleLinePositions(self, intSteps: int, intStepSize: int):
        lstReturnPoints = []
        lstMissingSteps = []
        for i in range(intSteps):
            lstTripleLines = self.FindTripleLines(intStepSize*i)
            if len(lstTripleLines) == 4:
                lstNewPositions = list(map(lambda x: np.mean(x,axis=0),lstTripleLines))
                lstReturnPoints.append(lstNewPositions)
            else:
                lstMissingSteps.append(i)
        return lstReturnPoints,lstMissingSteps 
    def Animate(self,i):
        self.__ax.clear()
        intStep = 500*i
        lstAllPoints = self.FindTripleLines(intStep)
        lstColours = ['b','g','r','k']
        k = 0
        if len(lstAllPoints) == 4:
            lstNewPositions = list(map(lambda x: np.mean(x,axis=0),lstAllPoints))
           # self.__OriginalPositions = np.array(lstNewPositions)
            for j in lstAllPoints:
                if len(j) > 1:
                    self.__ax.set_aspect('auto')
                    if self.__bln3d:
                        self.__ax.scatter(*tuple(zip(*j)),c=lstColours[k])
                    else:
                        self.__ax.scatter(*tuple(zip(*j[:,:2])),c=lstColours[k])
                    self.__ax.set_xbound([-100,self.__CellVectors[0][0]+100])
                    self.__ax.set_ybound([-100,self.__CellVectors[1][1]+100])
                k +=1
        else:
            print("error missing junction lines only " + str(len(lstAllPoints)) + ' frame ' + str(i))
        print(str(i))
    def SetOriginalPositions(self, inPositions):
        self.__OriginalPositions = inPositions
    def WriteFile(self,strFilename: str,arrOriginalPositions: np.array, intFrames: int, bln3d= False):
        self.__OriginalPositions = arrOriginalPositions
        if bln3d:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            self.__bln3d = True
        else:
            fig,ax = plt.subplots()
        self.__ax = ax
        ani = animation.FuncAnimation(fig, self.Animate,interval=2000, frames=intFrames) 
        writergif = animation.PillowWriter(fps=10)
        ani.save(strFilename,writer=writergif) 
#%%
arrCellVectors = dctTJ7['550,015'].GetCellVectors()
objTJAnimate = AnimateTJs(strRoot7_7_49 +'550/u015/',arrCellVectors)
objTJAnimate.SetOriginalPositions(np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[0]+arrCellVectors[2]),0.5*(arrCellVectors[1]+arrCellVectors[2]),0.5*(arrCellVectors[0]+arrCellVectors[1]+arrCellVectors[2])]))
lstReturnPoints,lstMissingSteps = objTJAnimate.FindMeanTripleLinePositions(1000,100)
#%%
lstTime = list(range(0,100000,100))
intTJ = 3
arrPoints = np.vstack(lstReturnPoints)
#for j in range(len(lstReturnPoints)):sbatch 
#    plt.scatter(lstReturnPoints[j][intTJ][0],lstReturnPoints[j][intTJ][1], c='black')
plt.plot(arrPoints[intTJ:1000:4,0],arrPoints[intTJ:1000:4,1])
plt.axis('scaled')
plt.show()
plt.clf()
plt.cla()
plt.close()
#%%
print(np.where(arrPoints[2:800:4,0] < 1)[0])
objTJAnimate.FindMeanTripleLinePositions(1000,100)
#%%
def WriteTJAnimations(indctTJ: dict(), indct12BV: dict(), indct13BV: dict(), inRootDir: str, inSaveDir: str):
    for a in indctTJ:
        lstVolumeCutOff = []
        lstVolumeCutOff.append(indctTJ[a].GetLowVolumeCutOff(1,4*4.05))
        lstVolumeCutOff.append(indct12BV[a].GetLowVolumeCutOff(1,4*4.05))
        lstVolumeCutOff.append(indct13BV[a].GetLowVolumeCutOff(1,4*4.05))
        intFrames = np.min(lstVolumeCutOff)
        strTemp = str(indctTJ[a].GetTemp())
        strU = str(indctTJ[a].GetPEString())    
        strDirAnim = inRootDir + strTemp + '/u' + strU +'/'
        arrCellVectors = indctTJ[a].GetCellVectors()        
        objTJ = AnimateTJs(strDirAnim,arrCellVectors)
        objTJ.WriteFile(inSaveDir + strTemp + 'u' + strU + '.gif',np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[0]+arrCellVectors[2]),0.5*(arrCellVectors[1]+arrCellVectors[2]),0.5*(arrCellVectors[0]+arrCellVectors[1]+arrCellVectors[2])]),intFrames,False)
#%%
strRootDir = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strFileDir = r'/home/p17992pt/MobilityImages/Sigma7_7_49/TJAll'
WriteTJAnimations(dctTJ7,dct12BV7,dct13BV7,strRootDir,strFileDir)
#%%
def WriteGBAnimations(indctTJ: dict(), indct12BV: dict(), indct13BV: dict(), inRootDir: str, inSaveDir: str):
    for a in indctTJ:
        lstVolumeCutOff = []
        lstVolumeCutOff.append(indctTJ[a].GetLowVolumeCutOff(1,4*4.05))
        lstVolumeCutOff.append(indct12BV[a].GetLowVolumeCutOff(1,4*4.05))
        lstVolumeCutOff.append(indct13BV[a].GetLowVolumeCutOff(1,4*4.05))
        intFrames = np.min(lstVolumeCutOff)
        strTemp = str(indctTJ[a].GetTemp())
        strU = str(indctTJ[a].GetPEString())    
        strDirAnim = inRootDir + strTemp + '/u' + strU +'/'
        arrCellVectors = indctTJ[a].GetCellVectors()        
        objGB = AnimateGBs(strDirAnim,arrCellVectors)
        objGB.WriteFile(inSaveDir + strTemp + 'u' + strU + '.gif',True,True,True,intFrames) 
#%%
strRootDir = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strFileDir = r'/home/p17992pt/MobilityImages/Sigma7_7_49/GBAll'
WriteGBAnimations(dctTJ7,dct12BV7,dct13BV7,strRootDir,strFileDir)
#%%

#%%
###Checks mesh points 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
strRoot7_7_49 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strType = 'TJ'
objData = LT.LAMMPSData(strRoot7_7_49 + '450/u005/TJ/1Min.lst',1,4.05,LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
arrCellVectors = objAnalysis.GetCellVectors()
arrPoints12 = np.loadtxt(strRoot7_7_49 + '450/u005/TJ/Mesh12TJ46500.txt')
objTree12 = gf.PeriodicWrapperKDTree(arrPoints12,arrCellVectors,gf.FindConstraintsFromBasisVectors(arrCellVectors),20,['p','p','p'])
ax.scatter(*tuple(zip(*objTree12.GetExtendedPoints())))
arrPoints13 = np.loadtxt(strRoot7_7_49 + '450/u005/TJ/Mesh13TJ46500.txt')
objTree13 = gf.PeriodicWrapperKDTree(arrPoints13,arrCellVectors,gf.FindConstraintsFromBasisVectors(arrCellVectors),20,['p','p','p'])
ax.scatter(*tuple(zip(*objTree13.GetExtendedPoints())))
arrPoints23 = np.loadtxt(strRoot7_7_49 + '450/u005/TJ/Mesh23TJ46500.txt')
objTree23 = gf.PeriodicWrapperKDTree(arrPoints23,arrCellVectors,gf.FindConstraintsFromBasisVectors(arrCellVectors),20,['p','p','p'])
ax.scatter(*tuple(zip(*objTree23.GetExtendedPoints())))
plt.show()
#%%P
##Checking linear trends
dctAny = dct12BV
for a in dctAny.keys():
    if dctAny[a].GetTemp() == 550 and dctAny[a].GetPEParameter() <= 0.02:
        plt.title(str(a))
        arrValues = dctAny[a].GetLogValues()
        arrVolumeSpeed = dctAny[a].GetVolumeSpeed()
        intFinish = dctAny[a].GetLowVolumeCutOff(1,4*4.05)
        dctAny[a].SetLinearRange(10,intFinish)
        objSlice = dctAny[a].GetLinearRange()
        spl = UnivariateSpline(arrVolumeSpeed[0,objSlice],arrVolumeSpeed[1,objSlice])
        #spl.set_smoothing_factor(10)
        plt.plot(arrVolumeSpeed[0,5:intFinish], spl(arrVolumeSpeed[0,5:intFinish]), 'g', lw=3)
        plt.scatter(arrVolumeSpeed[0,5:intFinish],arrVolumeSpeed[2,5:intFinish],c='black')
        plt.show()
        popt2, pop = optimize.curve_fit(FitCurve, arrValues[5*5:5*intFinish,0],arrValues[5*5:5*intFinish,2])
        plt.scatter(arrValues[5*5:5*intFinish,0],arrValues[5*5:5*intFinish,2])
        plt.plot(arrValues[5*5:5*intFinish,0], FitCurve(arrValues[5*5:5*intFinish,0],*popt2),c='black')
        plt.show()
#%%
dctAny = dctTJ7
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() >= 0.005 and dctAny[a].GetTemp() == 550:
        arrRows = dctAny[a].GetOverlapRows(1)
        arrLogValues = dctAny[a].GetLogValues()
        arrVolumeSpeed = dctAny[a].GetVolumeSpeed()
        arrTime = arrLogValues[arrRows,0]
        arrPE = arrLogValues[arrRows,2]
        arrVolume = arrVolumeSpeed[1,:]
        intFinish = dctAny[a].GetLowVolumeCutOff(1,4*4.05)
        plt.title(str(dctAny[a].GetPEParameter()) + ' V against t')
        popt1C,pop1C = optimize.curve_fit(FitCurve,arrTime[5:intFinish], arrVolume[5:intFinish])
        plt.plot(arrTime[5:intFinish],FitCurve(arrTime[5:intFinish],*popt1C),c='black')
        plt.scatter(arrTime[10:intFinish],arrVolume[10:intFinish])
        plt.show()
        plt.title('PE against V')
        popt2C,pop2C = optimize.curve_fit(FitLine,arrVolume[5:intFinish], arrPE[5:intFinish])
        plt.plot(arrVolume[5:intFinish],FitLine(arrVolume[5:intFinish],*popt2C),c='black')
        plt.scatter(arrVolume[5:intFinish],arrPE[5:intFinish])
        plt.show()
#%%
### comparison of Triple line cell and bicrystal cell
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() <= 0.02:
        plt.scatter(dctGB[a].GetTemp(), dctGB[a].GetMobility(), c='red')
        plt.scatter(dctTJ[a].GetTemp(), dctTJ[a].GetMobility(), c='black')
plt.legend(['12BV', 'TJ'])
plt.show()
#%%
for a in dctTJ.keys():
    if dctTJ[a].GetTemp() == 600:
        plt.scatter(dctTJ[a].GetPEPerVolume(1, 4*4.05), -dctTJ[a].GetNormalSpeed(1, 4*4.05), c='red')
        #
#plt.legend(['12BV', 'TJ'])
plt.show()
# %%
##arrhenius plot of mobility
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.05:
        plt.scatter(1/dct12BV[a].GetTemp(), np.log(dct12BV[a].GetMobility()), c='red')
        plt.scatter(1/dctTJ[a].GetTemp(), np.log(dctTJ[a].GetMobility()), c='black')
plt.legend(['12BV', 'TJ'])
plt.show()
#%%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.05:
        plt.scatter(dctTJR[a].GetTemp(), dctTJR[a].GetMobility(), c='red')
        plt.scatter(dctTJ[a].GetTemp(), dctTJ[a].GetMobility(), c='black')
plt.legend(['TJR', 'TJ'])
plt.show()
#%%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.05:
        plt.scatter(1/dctTJR[a].GetTemp(), np.log(dctTJR[a].GetMobility()), c='red')
        plt.scatter(1/dctTJ[a].GetTemp(), np.log(dctTJ[a].GetMobility()), c='black')
plt.legend(['TJR', 'TJ'])
plt.show()
#%%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetTemp() == 450:
        plt.scatter(dctTJR[a].GetPEPerVolume(), np.log(dctTJR[a].GetMobility()), c='red')
        plt.scatter(1/dctTJ[a].GetTemp(), np.log(dctTJ[a].GetMobility()), c='black')
plt.legend(['TJR', 'TJ'])
plt.show()
# %%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.045:
        plt.scatter(dct12BV[a].GetTemp(), +4*dct12BV[a].GetPEParameter()*4.05**(-3) - dct12BV[a].GetPEPerVolume() , c='red')
        plt.scatter(dctTJ[a].GetTemp(), +4*dctTJ[a].GetPEParameter()*4.05**(-3) - dctTJ[a].GetPEPerVolume() , c='black')
plt.legend(['12BV', 'TJ'])
plt.show()
# %%
#%matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lstPEPerV = []
lstTemp = []
lstMob = []
for a in dctAny:
    if dctAny[a].GetPEParameter() != 0.01:
        lstPEPerV.append(1000*dctAny[a].GetPEPerVolume())
        lstTemp.append(dctAny[a].GetTemp())
        lstMob.append(dctAny[a].GetMobility())
        #ax.scatter(dctAny[a].GetPEPerVolume(),dctAny[a].GetTemp(), dctAny[a].GetMobility())
ax.scatter(lstPEPerV, lstTemp, lstMob)
ax.plot(lstPEPerV, lstTemp, lstMob)
surf = ax.plot_trisurf(lstPEPerV, lstTemp, lstMob)
fig.colorbar(surf)

fig.tight_layout()
plt.show()  # or:
