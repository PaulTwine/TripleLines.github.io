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
    if intReverse == 1:
        if strType == 'TJ':
            arrIDs1 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
            arrIDs2 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_2[2]',1)
            arrIDs = np.append(arrIDs1, arrIDs2, axis=0)
        else:
            arrIDs = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
    else:
        arrIDs = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
    if len(arrIDs) > 0:
        arrPoints = objAnalysis.GetAtomsByIDs(arrIDs)
    else:
        blnSplitGrains = True
    if len(arrIDs) > 0:
        fltVolume = np.sum(objAnalysis.GetAtomsByID(arrIDs)[:,intVColumn])
    else: 
        fltVolume = 0
    lstSpeed.append(fltVolume/fltCrossSection)
    lstVolume.append(fltVolume)
    lstTime.append(t)
np.savetxt(strDirectory + '/Volume' + strType + '.txt', np.array([np.array(lstTime),np.array(lstVolume),np.array(lstSpeed)]))



