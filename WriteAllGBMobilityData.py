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
intStep =  int(sys.argv[5])
intReverse =  int(sys.argv[6])

lstVolume = []
lstTime = []
lstSpeed = []
objData = LT.LAMMPSData(strDirectory + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
intVColumn = objAnalysis.GetColumnIndex('c_v[1]')
arrCellVectors = objAnalysis.GetCellVectors()
fltCrossSection = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
t = intLow
blnStop = False
intEco = 1
blnWrap = False
if intReverse == 1:
    intEco = -intEco
while t <= intHigh and not(blnStop): 
    objData = LT.LAMMPSData(strDirectory + '1Sim' + str(t) + '.dmp', 1, 4.05, LT.LAMMPSAnalysis3D)
    objAnalysis = objData.GetTimeStepByIndex(-1)
    arrIDs1 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',intEco)
    arrIDs2 =  objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',-intEco)
    if (len(arrIDs1) > 0) and (len(arrIDs2) > 0):
        objAnalysis.SetPeriodicGrain('1',arrIDs1, 25)
        objAnalysis.SetPeriodicGrain('2',arrIDs2, 25)
        arrPoints12 = objAnalysis.FindDefectiveMesh('1','2',25)
        fltVolume = np.sum(objAnalysis.GetAtomsByID(arrIDs1)[:,intVColumn])
        if intReverse == 1:
            fltVolume = np.linalg.det(objAnalysis.GetCellVectors()) - fltVolume
        if (len(arrPoints12) > 0):
            if blnWrap:
                arrPoints12 = objAnalysis.WrapVectorIntoSimulationBox(arrPoints12)
            np.savetxt(strDirectory + '/Mesh12' + strType + str(t) + '.txt', arrPoints12)
            lstSpeed.append(fltVolume/fltCrossSection)
            lstVolume.append(fltVolume)
            lstTime.append(t)
            t += intStep
        else:
            blnStop = True
    else:
        blnStop = True
np.savetxt(strDirectory + '/Volume' + strType + '.txt', np.array([np.array(lstTime),np.array(lstVolume),np.array(lstSpeed)]))



