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
strType = 'TJ'
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp450/u005/' + strType + '/'
objLog = LT.LAMMPSLog(strFilename + strType + '.log')
objData = LT.LAMMPSData(strFilename + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
arrCellVectors = objAnalysis.GetCellVectors()
fltVolume = np.linalg.det(arrCellVectors)
fltArea = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
print(objLog.GetColumnNames(0),fltVolume)
arrV = objLog.GetValues(1)
intStart = 50
intFinish = 150
plt.scatter(arrV[intStart:-intFinish,0],arrV[intStart:-intFinish,2])
popt, pcov = optimize.curve_fit(FitLine,arrV[intStart:-intFinish,0], arrV[intStart:-intFinish,2])
plt.plot(arrV[intStart:-intFinish,0], FitLine(arrV[intStart:-intFinish,0], popt[0],popt[1]),c='black')
plt.show()
fltPEdt = popt[0]
print(popt, -fltPEdt/fltArea)
print(np.corrcoef(arrV[intStart:-intFinish,0],arrV[intStart:-intFinish,2]))
#%%
fltU = 0.005
intStart = 5
intFinish = 5
arrValues = np.loadtxt(strFilename + 'Volume' + strType + '.txt')
popt, pcov = optimize.curve_fit(FitLine,arrValues[0,intStart:-intFinish], arrValues[1,intStart:-intFinish])
fltVolumedt = popt[0]
plt.plot(arrValues[0,intStart:-intFinish], FitLine(arrValues[0,intStart:-intFinish], popt[0],popt[1]),c='black')
plt.scatter(arrValues[0,intStart:-intFinish],arrValues[1,intStart:-intFinish])
plt.show()
fltPEPerVolume = fltPEdt/fltVolumedt 
fltUParameter = 4*fltU*4.05**(-3)
print(fltPEPerVolume,fltUParameter,(fltPEPerVolume-fltUParameter)/fltUParameter,np.corrcoef(arrValues[0,intStart:-intFinish],arrValues[1,intStart:-intFinish]))
#%%
### mobility
print(-fltVolumedt/(fltArea*fltPEPerVolume))
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
