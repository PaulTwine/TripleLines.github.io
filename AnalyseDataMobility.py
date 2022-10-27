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
# %%


def FitLine(x, a, b):
    return a*x + b 
#%%
class CSLMobility(object):
    def __init__(self, arrCellVectors: np.array, arrLogValues: np.array, arrVolumeFile: np.array, strType: str, fltTemp: float, fltUPerVolume: float):
        self.__LogValues = arrLogValues
        self.__CellVectors = arrCellVectors
        self.__VolumeSpeed = arrVolumeFile
        self.__Temp = fltTemp
        self.__UValue = fltUPerVolume
        self.__Type = strType
        self.__Volume = np.abs(np.linalg.det(arrCellVectors))
        self.__Area = np.linalg.norm(
            np.cross(arrCellVectors[0], arrCellVectors[2]))
        self.__Mobility = 0
        self.__PEPerVolume = 0
    def FitLine(self, x, a, b):
        return a*x + b
    def GetLogValues(self):
        return self.__LogValues
    def GetCellVectors(self):
        return self.__CellVectors
    def GetNormalSpeed(self,intStage: int,fltNormalDistance: float):
        intFinish = self.GetLowVolumeCutOff(intStage, fltNormalDistance)
        intStart = np.max([intFinish -50,10]).astype('int')
        popt, pop = optimize.curve_fit(
            self.FitLine, self.__VolumeSpeed[0, intStart:intFinish], self.__VolumeSpeed[2,intStart:intFinish])
        return popt[0]
    def GetPEPerVolume(self,intStage: int,fltNormalDistance: float):
        arrRows = self.GetOverlapRows(intStage)
        arrPEValues = self.__LogValues[arrRows]
        intFinish = self.GetLowVolumeCutOff(1, fltNormalDistance)
       # intStart = np.round(intFinish/2, 0).astype('int')
        intStart = np.max([intFinish -50,10]).astype('int')
        popt, pop = optimize.curve_fit(
            self.FitLine, self.__VolumeSpeed[1, intStart:intFinish], arrPEValues[intStart:intFinish, 2])
        return popt[0]
    def GetVolumeSpeed(self, intColumn=None):
        if intColumn is None:
            return self.__VolumeSpeed
        else:
            return self.__VolumeSpeed[:, intColumn]
    def GetType(self):
        return self.__Type
    def GetTemp(self):
        return self.__Temp
    def GetPEParameter(self):
        return self.__UValue
    def GetPEString(self):
        return str(self.__UValue).split('.')[1]
    def GetOverlapRows(self, intStage: int):
        arrValues = self.__LogValues
        arrRows = np.where(np.isin(arrValues[:, 0], self.__VolumeSpeed[0, :]))[0]
        return arrRows
    def GetLowVolumeCutOff(self, intStage: int, fltDistance: float):
        arrCellVectors = self.GetCellVectors()
        fltArea = np.linalg.norm(
            np.cross(arrCellVectors[0], arrCellVectors[2]))
        arrValues = self.__VolumeSpeed[1, :]
        arrRows = np.where(arrValues < fltDistance*fltArea)[0]
        if len(arrRows) > 0:
            intReturn = np.min(arrRows)
        else:
            intReturn = len(arrValues)
        return intReturn
    def SetMobility(self, inMobility):
        self.__Mobility = inMobility
    def GetMobility(self):
        return self.__Mobility
    def SetPEPerVolume(self, inPE):
        self.__PEPerVolume = inPE
    def FindMobilities(self, intStage: int, fltNormalDistance: float):
        arrPEValues = self.__LogValues
        arrVolumeSpeed = self.GetVolumeSpeed()
        arrRows = self.GetOverlapRows(intStage)
        arrPEValues = arrPEValues[arrRows]
        intFinish = self.GetLowVolumeCutOff(1, fltNormalDistance)
       # intStart = np.round(intFinish/2, 0).astype('int')
        intStart = np.max([intFinish -25,10]).astype('int')
        popt, pop = optimize.curve_fit(
            self.FitLine, arrVolumeSpeed[1, intStart:intFinish], arrPEValues[intStart:intFinish, 2])
        popt2, pop2 = optimize.curve_fit(
            self.FitLine, arrVolumeSpeed[0, intStart:intFinish], arrVolumeSpeed[2, intStart:intFinish])
        arrCorr1= np.corrcoef(arrVolumeSpeed[1, intStart:intFinish], arrPEValues[intStart:intFinish, 2])[1,0]
        arrCorr2 = np.corrcoef(arrVolumeSpeed[0, intStart:intFinish], arrVolumeSpeed[2, intStart:intFinish])[1,0]
        if (np.abs(arrCorr1) < 0.95) or (np.abs(arrCorr2) < 0.95):
            print('Correlation warning rho1 = ' + str(arrCorr1) + ', rho2 = ' + str(arrCorr2) + ' temp ' + str(self.GetTemp()) + ' u parameter ' + str(self.GetPEParameter()))
        self.SetPEPerVolume(popt[0])
        self.SetMobility(-popt2[0]/popt[0])
        return intStart, intFinish


# %%
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
lstTemp = [450, 500, 550, 600, 650]
lstU = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
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
            strLogFile = strDir + strType + '.log'
            strVolumeFile = strDir + 'Volume' + strType + '.txt'
            objData = LT.LAMMPSData(
                strDir + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
            objAnalysis = objData.GetTimeStepByIndex(-1)
            arrCellVectors = objAnalysis.GetCellVectors()
            arrLog = LT.LAMMPSLog(strLogFile)
            arrVolume = np.loadtxt(strVolumeFile)
            objCSLMobility = CSLMobility(
                arrCellVectors, arrLog.GetValues(1), arrVolume, strType, T, u)
            objCSLMobility.FindMobilities(1, 4*4.05)
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
            arrVolume = arrVolume1 + arrVolume2
            arrVolume[2,:] = arrVolume[1,:]/fltArea
            arrVolume[0,:] = arrVolume1[0,:]
            objCSLMobility = CSLMobility(
                arrTJCellVectors, arrLog, arrVolume, strType1, T, u)
            objCSLMobility.FindMobilities(1, 4*4.05)
            dctReturn[str(T) + ',' + strU] = objCSLMobility
            #objLog = objCSLMobility.GetLogObject()
            # print(objLog.GetColumnNames(1),strU,str(T))
    return dctReturn
# %%


def FindMobilities(dctCSLMobility: dict()):
    for a in dctCSLMobility:
        objCSLMobility = dctCSLMobility[a]
        arrLog = objCSLMobility.GetLogObject()
        arrPEValues = arrLog.GetValues(1)
        arrVolumeSpeed = objCSLMobility.GetVolumeSpeed()
        arrRows = objCSLMobility.GetOverlapRows(1)
        arrPEValues = arrPEValues[arrRows]
        intFinish = objCSLMobility.GetLowVolumeCutOff(1, 4*4.05)
        intStart = np.round(intFinish/2, 0).astype('int')
        popt, pop = optimize.curve_fit(
            FitLine, arrVolumeSpeed[1, intStart:intFinish], arrPEValues[intStart:intFinish, 2])
        popt2, pop2 = optimize.curve_fit(
            FitLine, arrVolumeSpeed[0, intStart:intFinish], arrVolumeSpeed[2, intStart:intFinish])
        dctCSLMobility[a].SetPEPerVolume(popt[0])
        dctCSLMobility[a].SetMobility(-popt2[0]/popt[0])
        plt.title(str(a) + 'dU/dV')
        plt.plot(arrVolumeSpeed[1, intStart:intFinish], FitLine(
            arrVolumeSpeed[1, intStart:intFinish], popt[0], popt[1]), c='black')
        plt.scatter(arrVolumeSpeed[1, intStart:intFinish],
                    arrPEValues[intStart:intFinish, 2])
        plt.show()
        plt.title(str(a) + 'vn')
        plt.plot(arrVolumeSpeed[0, intStart:intFinish], FitLine(
            arrVolumeSpeed[0, intStart:intFinish], popt2[0], popt2[1]), c='black')
        plt.scatter(arrVolumeSpeed[0, intStart:intFinish],
                    arrVolumeSpeed[2, intStart:intFinish])
        print(np.corrcoef(
            arrVolumeSpeed[1, intStart:intFinish], arrPEValues[intStart:intFinish, 2]))
        plt.show()

# %%
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strRootR = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp'

lstTemp = [450, 500, 550, 600, 650]
lstU = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03,0.035,0.04,0.045,0.05]
strType = 'TJ'
dctTJR = PopulateTJDictionary(strRootR, lstTemp, lstU, 'TJ')
dctTJ = PopulateTJDictionary(strRoot, lstTemp, lstU, 'TJ')
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strType = 'TJ'
dct12BVR = PopulateGBDictionary(strRootR, lstTemp, lstU, '12BV','13BV', dctTJ['450,005'].GetCellVectors())
dct12BV = PopulateGBDictionary(strRoot, lstTemp, lstU, '12BV','13BV',dctTJ['450,005'].GetCellVectors())
#%%
def PartitionByTemperature(dctAny: dict(),intTemp):
    lstTempVn = []
    lstTempU = []
    for a in dctAny.keys():
        if (dctAny[a].GetTemp() == intTemp) and (dctAny[a].GetPEParameter() > 0):
            lstTempVn.append(-dctAny[a].GetNormalSpeed(1,4*4.05))
            lstTempU.append(dctAny[a].GetPEPerVolume(1,4*4.05))
            #lstTempU.append(4*dctAny[a].GetPEParameter()*4.05**(-3))
    return lstTempU,lstTempVn
#%%
lstMobility = []
lstMobilityLim = []
for j in lstTemp:
    tupValues = PartitionByTemperature(dct12BV,j)
    popt, popt = optimize.curve_fit(FitLine,tupValues[0],tupValues[1])
    lstMobility.append(popt[0][0])
    lstMobilityLim.append(popt[0][1])
    plt.xlim([0,0.003])
    plt.ylim([0,0.008])
    plt.scatter(tupValues[0],tupValues[1])
plt.legend(lstTemp)
plt.show()

plt.scatter(lstTemp, lstMobility)
plt.show()

#%%
### comparison of Triple line cell and bicrystal cell
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.035:
        plt.scatter(dct12BV[a].GetTemp(), dct12BV[a].GetMobility(), c='red')
        plt.scatter(dctTJ[a].GetTemp(), dctTJ[a].GetMobility(), c='black')
plt.legend(['12BV', 'TJ'])
plt.show()
#%%
for a in dctAny.keys():
    if dctAny[a].GetTemp() == 500:
        plt.scatter(dct12BV[a].GetPEPerVolume(), dct12BV[a].GetMobility(), c='red')
        plt.scatter(dctTJ[a].GetPEPerVolume(), dctTJ[a].GetMobility(), c='black')
plt.legend(['12BV', 'TJ'])
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
%matplotlib qt
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
# %%
strType = '13BV'
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp450/u03/' + strType + '/'
objLog = LT.LAMMPSLog(strFilename + strType + '.log')
objData = LT.LAMMPSData(strFilename + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
arrCellVectors = objAnalysis.GetCellVectors()
fltVolume = np.linalg.det(arrCellVectors)
fltArea = np.linalg.norm(np.cross(arrCellVectors[0], arrCellVectors[2]))
print(objLog.GetColumnNames(0), fltVolume)
arrLogFile = objLog.GetValues(1)
intStart = 50
intFinish = 200
plt.scatter(arrLogFile[intStart:-intFinish, 0],
            arrLogFile[intStart:-intFinish, 2])
popt, pcov = optimize.curve_fit(
    FitLine, arrLogFile[intStart:-intFinish, 0], arrLogFile[intStart:-intFinish, 2])
plt.plot(arrLogFile[intStart:-intFinish, 0],
         FitLine(arrLogFile[intStart:-intFinish, 0], popt[0], popt[1]), c='black')
plt.show()
fltPEdt = popt[0]
print(popt, -fltPEdt/fltArea)
print(np.corrcoef(arrLogFile[intStart:-intFinish,
                             0], arrLogFile[intStart:-intFinish, 2]))
# %%
fltU = 0.03
intStart = np.round(intStart/5, 0).astype('int')
intFinish = np.round(intFinish/5).astype('int')
arrVolumeSpeed = np.loadtxt(strFilename + 'Volume' + strType + '.txt')
popt, pcov = optimize.curve_fit(
    FitLine, arrVolumeSpeed[0, intStart:-intFinish], arrVolumeSpeed[1, intStart:-intFinish])
fltVolumedt = popt[0]
plt.plot(arrVolumeSpeed[0, intStart:-intFinish], FitLine(
    arrVolumeSpeed[0, intStart:-intFinish], popt[0], popt[1]), c='black')
plt.scatter(arrVolumeSpeed[0, intStart:-intFinish],
            arrVolumeSpeed[1, intStart:-intFinish])
plt.show()
fltPEPerVolume = fltPEdt/fltVolumedt
fltUParameter = 4*fltU*4.05**(-3)
print(fltPEPerVolume, fltUParameter, (fltPEPerVolume-fltUParameter)/fltUParameter,
      np.corrcoef(arrVolumeSpeed[0, intStart:-intFinish], arrVolumeSpeed[1, intStart:-intFinish]))
print(len(arrVolumeSpeed[0, :])/len(arrLogFile[:, 0]))
# %%
# mobility
m = -fltVolumedt/(fltArea*fltPEPerVolume)
print(m)
lstMobility.append(m)
lstUPerV.append(fltPEPerVolume)
# %%
lstU = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
arrU = 4*4.05**(-3)*np.array(lstU)
plt.scatter(lstUPerV, lstMobility)
plt.scatter(arrU, lstMobility)
plt.show()
# %%
# def VolumeRateChange(strDirectory,strType, intLow,intHigh,intStep,blnReverse = False):
#     lstVolume = []
#     lstTime = []
#     lstPE = []
#     objData = LT.LAMMPSData(strDirectory + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
#     objAnalysis = objData.GetTimeStepByIndex(-1)
#     intVColumn = objAnalysis.GetColumnIndex('c_v[1]')
#     intPEColumn = objAnalysis.GetColumnIndex('c_pe1')
#     arrCellVectors = objAnalysis.GetCellVectors()
#     fltCrossSection = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
#     for t in range(intLow,intHigh+intStep,intStep):
#         objData = LT.LAMMPSData(strDirectory + '1Sim' + str(t) + '.dmp', 1, 4.05, LT.LAMMPSAnalysis3D)
#         objAnalysis = objData.GetTimeStepByIndex(-1)
#         if blnReverse:
#             if strType == 'TJ':
#                 arrIDs1 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
#                 arrIDs2 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_2[2]',1)
#                 arrIDs = np.append(arrIDs1, arrIDs2, axis=0)
#             else:
#                 arrIDs = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
#         else:
#             arrIDs = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
#         if len(arrIDs) > 0:
#             fltVolume = np.sum(objAnalysis.GetAtomsByID(arrIDs)[:,intVColumn])
#             fltPE = np.sum(objAnalysis.GetAtomsByID(arrIDs)[:,intPEColumn])
#         else:
#             fltVolume = 0
#         lstVolume.append(fltVolume/fltCrossSection)
#         lstPE.append(fltPE)
#         lstTime.append(t)
#     return lstTime, lstVolume, lstPE
# strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp550/u'
# lstUValues = [0.005,0.01, 0.015,0.02,0.025,0.03]
# strUValues = list(map(lambda s: str(s).split('.')[1], lstUValues))
# for u in strUValues:
#     lstFilenames = ['TJ', '12BV','13BV']
#     for k in lstFilenames:
#         lstTime,lstVolume,lstPE = VolumeRateChange(strFilename + u + '/'  + str(k) + '/',k, 1000, 50000, 1000,True)
#         np.savetxt(strFilename + u + '/' + str(k) + '/Volume' + str(k) + '.txt', np.array([np.array(lstTime),np.array(lstVolume),np.array(lstPE)]))

# %%
