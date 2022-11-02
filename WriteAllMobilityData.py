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

strDirectory = str(sys.argv[1])
strType = str(sys.argv[2])
intLow = int(sys.argv[3])
intHigh = int(sys.argv[4])
intStep = int(sys.argv[5])
intReverse = int(sys.argv[6])

lstVolume = []
lstTime = []
lstSpeed = []
objData = LT.LAMMPSData(strDirectory + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
intVColumn = objAnalysis.GetColumnIndex('c_v[1]')
arrCellVectors = objAnalysis.GetCellVectors()
fltCrossSection = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
for t in range(intLow,intHigh+intStep,intStep):        
    objData = LT.LAMMPSData(strDirectory + '1Sim' + str(t) + '.dmp', 1, 4.05, LT.LAMMPSAnalysis3D)
    objAnalysis = objData.GetTimeStepByIndex(-1)
    arrIDs1 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
    objAnalysis.SetPeriodicGrain('1',arrIDs1, 25)
    if strType == 'TJ':
        arrIDs2 =  objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',-1)
        arrIDs3 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_2[2]',-1)
        objAnalysis.SetPeriodicGrain('2',arrIDs2, 25)
        objAnalysis.SetPeriodicGrain('3',arrIDs3, 25)
        arrPoints12 = objAnalysis.FindDefectiveMesh('1','2',25)
        arrPoints13 = objAnalysis.FindDefectiveMesh('1','3',25)
        arrPoints23 = objAnalysis.FindDefectiveMesh('2','3',25)
        if intReverse == 1:
            arrVolumeIDs = np.append(arrIDs1, arrIDs2, axis=0)
        else: 
            arrVolumeIDs = arrIDs1
    else:
        arrIDs2 =  objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',-1)
        objAnalysis.SetPeriodicGrain('2',arrIDs2, 10)
        arrPoints12 = objAnalysis.FindDefectiveMesh('1','2',10)
        arrVolumeIDs = arrIDs2
    if len(arrVolumeIDs) > 0:
        arrPoints = objAnalysis.GetAtomsByID(arrVolumeIDs)
    else:
        blnSplitGrains = True
    if len(arrVolumeIDs) > 0:
        fltVolume = np.sum(objAnalysis.GetAtomsByID(arrVolumeIDs)[:,intVColumn])
    else: 
        fltVolume = 0
    if len(arrPoints12) > 0:
        arrPoints12 = objAnalysis.WrapVectorIntoSimulationBox(arrPoints12)
        np.savetxt(strDirectory + '/Mesh12' + strType + str(t) + '.txt', arrPoints12)
    if strType == 'TJ':
        if len(arrPoints13) > 0:
            arrPoints13 = objAnalysis.WrapVectorIntoSimulationBox(arrPoints13)
            np.savetxt(strDirectory + '/Mesh13' + strType + str(t) + '.txt', arrPoints13)
        if len(arrPoints23) > 0:
            arrPoints23 = objAnalysis.WrapVectorIntoSimulationBox(arrPoints23)
            np.savetxt(strDirectory + '/Mesh23' + strType + str(t) + '.txt', arrPoints23)
    lstSpeed.append(fltVolume/fltCrossSection)
    lstVolume.append(fltVolume)
    lstTime.append(t)
np.savetxt(strDirectory + '/Volume' + strType + '.txt', np.array([np.array(lstTime),np.array(lstVolume),np.array(lstSpeed)]))



