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
from scipy import optimize
#%%
def FitLine(x,a,b):
    return a*x + b
#%%
lstMobility = []
lstUPerV = []
#%%
class CSLMobility(object):
    def __init__(self,arrCellVectors: np.array, strLogFile: str, strVolumeFile: str,strType: str, fltTemp: float, fltUPerVolume):
        self.__Log = LT.LAMMPSLog(strLogFile) 
        self.__CellVectors = arrCellVectors
        self.__VolumeSpeed = np.loadtxt(strVolumeFile)
        self.__Temp = fltTemp
        self.__UValue = fltUPerVolume
        self.__Type = strType
        self.__Volume = np.abs(np.linalg.det(arrCellVectors))
        self.__Area = np.linalg.norm(np.cross(arrCellVectors[0], arrCellVectors[2]))
        self.__Mobility = 0
        self.__PEPerVolume = 0
    def FitLine(self,x,a,b):
        return a*x + b
    def GetLogObject(self):
        return self.__Log
    def GetCellVectors(self):
        return self.__CellVectors
    def GetVolumeSpeed(self, intColumn = None):
        if intColumn is None:
            return self.__VolumeSpeed
        else:
            return self.__VolumeSpeed[:,intColumn]
    def GetType(self):
        return self.__Type
    def GetTemp(self):
        return self.__Temp
    def GetPEParameter(self):
        return self.__UValue
    def GetPEString(self):
        return str(self.__UValue).split('.')[1]
    def GetOverlapRows(self, intStage: int):
        arrValues = self.__Log.GetValues(intStage)
        arrRows = np.where(np.isin(arrValues[:,0], self.__VolumeSpeed[0,:]))
        return arrRows
    def GetLowVolumeCutOff(self,intStage: int, fltDistance :float):
        arrCellVectors = self.GetCellVectors()
        fltArea = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
        arrValues = self.__VolumeSpeed[1,:]
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
    def GetPEPerVolume(self):
        return self.__PEPerVolume
#%%
strType ='TJ'
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp650/u03/' + strType + '/'
strLogFile = strFilename + strType + '.log'
strVolumeFile = strFilename + 'Volume' + strType + '.txt'
objData = LT.LAMMPSData(strFilename + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
arrCellVectors = objAnalysis.GetCellVectors()
objCSLMobility = CSLMobility(arrCellVectors,strLogFile,strVolumeFile,strType,650, 0.005)

objLog = objCSLMobility.GetLogObject()
arrVolumeSpeed = objCSLMobility.GetVolumeSpeed()

#plt.scatter(objLog.GetValues(1)[50:,0],objLog.GetValues(1)[50:,2])
plt.scatter(arrVolumeSpeed[0,:], arrVolumeSpeed[1,:])
plt.show()

#%%
fltArea = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
arrRows = np.where(arrVolumeSpeed[1,:] < 3*4.05*fltArea)[0]
intFinish =np.min(arrRows)
print(fltArea,arrRows, arrVolumeSpeed[0,intFinish])
print(objCSLMobility.GetPEString(),objCSLMobility.GetPEParameter())
#%%
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
lstTemp = [450,500,550,600,650]
lstU = [0.005,0.01,0.015,0.02,0.025,0.03]
dctTJ = dict()
strType = 'TJ'
for T in lstTemp:
    for u in lstU:
        strU = str(u).split('.')[1]
        strDir = strRoot + str(T) +  '/u' + strU  + '/'+ strType + '/'
        strLogFile = strDir + strType + '.log'
        objLog = LT.LAMMPSLog(strLogFile)
        print(strU,T, objLog.GetColumnNames(1))

#%%
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
lstTemp = [450,500,550,600,650]
lstU = [0.005,0.01,0.015,0.02,0.025,0.03]
dct12BV = dict()
strType = 'TJ'
for T in lstTemp:
    for u in lstU:
        strU = str(u).split('.')[1]
        strDir = strRoot + str(T) +  '/u' + strU  + '/'+ strType + '/'
        strLogFile = strDir + strType + '.log'
        strVolumeFile = strDir + 'Volume' + strType + '.txt'
        objData = LT.LAMMPSData(strDir + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
        objAnalysis = objData.GetTimeStepByIndex(-1)
        arrCellVectors = objAnalysis.GetCellVectors()
        objCSLMobility = CSLMobility(arrCellVectors,strLogFile, strVolumeFile, strType, T, u)
        dct12BV[str(T) + ',' + strU] = objCSLMobility
        objLog = objCSLMobility.GetLogObject()
        print(objLog.GetColumnNames(1),strU,str(T))
#%%
dctAny  = dct12BV
for a in dctAny:
    objCSLMobility = dctAny[a]
    arrLog = objCSLMobility.GetLogObject()
    arrPEValues = arrLog.GetValues(1)
    arrVolumeSpeed = objCSLMobility.GetVolumeSpeed()
    arrRows = objCSLMobility.GetOverlapRows(1)
    arrPEValues = arrPEValues[arrRows]
    intFinish = objCSLMobility.GetLowVolumeCutOff(1,4*4.05)
    intStart = np.round(intFinish/2,0).astype('int')
    popt, pop = optimize.curve_fit(FitLine,arrVolumeSpeed[1,intStart:intFinish],arrPEValues[intStart:intFinish,2])
    popt2, pop2 = optimize.curve_fit(FitLine,arrVolumeSpeed[0,intStart:intFinish],arrVolumeSpeed[2,intStart:intFinish])
    dctAny[a].SetPEPerVolume(popt[0])
    dctAny[a].SetMobility(-popt2[0]/popt[0])
    plt.title(str(a) + 'dU/dV')
    plt.plot(arrVolumeSpeed[1,intStart:intFinish],FitLine(arrVolumeSpeed[1,intStart:intFinish],popt[0],popt[1]),c='black')
    plt.scatter(arrVolumeSpeed[1,intStart:intFinish],arrPEValues[intStart:intFinish,2])
    plt.show()
    plt.title(str(a) + 'vn')
    plt.plot(arrVolumeSpeed[0,intStart:intFinish],FitLine(arrVolumeSpeed[0,intStart:intFinish],popt2[0],popt2[1]),c='black')
    plt.scatter(arrVolumeSpeed[0,intStart:intFinish],arrVolumeSpeed[2,intStart:intFinish])
    print(np.corrcoef(arrVolumeSpeed[1,intStart:intFinish],arrPEValues[intStart:intFinish,2]))
    plt.show()
#%%
for a in dctAny:
    if dctAny[a].GetPEParameter() == 0.005:
        plt.scatter(dctAny[a].GetTemp(),dctAny[a].GetMobility())
plt.show() 
#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lstPEPerV = []
lstTemp = []
lstMob = []
for a in dctAny:
    if dctAny[a].GetPEParameter() != 0.005:
        lstPEPerV.append(1000*dctAny[a].GetPEPerVolume())
        lstTemp.append(dctAny[a].GetTemp())
        lstMob.append(dctAny[a].GetMobility())
        #ax.scatter(dctAny[a].GetPEPerVolume(),dctAny[a].GetTemp(), dctAny[a].GetMobility())
ax.scatter(lstPEPerV,lstTemp,lstMob)
ax.plot(lstPEPerV,lstTemp,lstMob)
surf = ax.plot_trisurf(lstPEPerV, lstTemp, lstMob)
fig.colorbar(surf)

fig.tight_layout()
plt.show() # or:
#%%
strType = '13BV'
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49R/Temp450/u03/' + strType + '/'
objLog = LT.LAMMPSLog(strFilename + strType + '.log')
objData = LT.LAMMPSData(strFilename + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
arrCellVectors = objAnalysis.GetCellVectors()
fltVolume = np.linalg.det(arrCellVectors)
fltArea = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
print(objLog.GetColumnNames(0),fltVolume)
arrLogFile = objLog.GetValues(1)
intStart = 50
intFinish = 200
plt.scatter(arrLogFile[intStart:-intFinish,0],arrLogFile[intStart:-intFinish,2])
popt, pcov = optimize.curve_fit(FitLine,arrLogFile[intStart:-intFinish,0], arrLogFile[intStart:-intFinish,2])
plt.plot(arrLogFile[intStart:-intFinish,0], FitLine(arrLogFile[intStart:-intFinish,0], popt[0],popt[1]),c='black')
plt.show()
fltPEdt = popt[0]
print(popt, -fltPEdt/fltArea)
print(np.corrcoef(arrLogFile[intStart:-intFinish,0],arrLogFile[intStart:-intFinish,2]))
#%%
fltU = 0.03
intStart = np.round(intStart/5,0).astype('int')
intFinish = np.round(intFinish/5).astype('int')
arrVolumeSpeed = np.loadtxt(strFilename + 'Volume' + strType + '.txt')
popt, pcov = optimize.curve_fit(FitLine,arrVolumeSpeed[0,intStart:-intFinish], arrVolumeSpeed[1,intStart:-intFinish])
fltVolumedt = popt[0]
plt.plot(arrVolumeSpeed[0,intStart:-intFinish], FitLine(arrVolumeSpeed[0,intStart:-intFinish], popt[0],popt[1]),c='black')
plt.scatter(arrVolumeSpeed[0,intStart:-intFinish],arrVolumeSpeed[1,intStart:-intFinish])
plt.show()
fltPEPerVolume = fltPEdt/fltVolumedt 
fltUParameter = 4*fltU*4.05**(-3)
print(fltPEPerVolume,fltUParameter,(fltPEPerVolume-fltUParameter)/fltUParameter,np.corrcoef(arrVolumeSpeed[0,intStart:-intFinish],arrVolumeSpeed[1,intStart:-intFinish]))
print(len(arrVolumeSpeed[0,:])/len(arrLogFile[:,0]))
#%%
### mobility
m = -fltVolumedt/(fltArea*fltPEPerVolume)
print(m)
lstMobility.append(m)
lstUPerV.append(fltPEPerVolume)
#%%
lstU = [0.005,0.01,0.015,0.02,0.025,0.03]
arrU = 4*4.05**(-3)*np.array(lstU)
plt.scatter(lstUPerV,lstMobility)
plt.scatter(arrU,lstMobility)
plt.show()
#%%
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
