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
intHigh = int(sys.argv[5])
intStep = int(sys.argv[6])
def VolumeRateChange(strDirectory,strType, intLow,intHigh,intStep,blnReverse = False):
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
        if blnReverse:
            if strType == 'TJ':
                arrIDs1 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
                arrIDs2 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_2[2]',1)
                arrIDs = np.append(arrIDs1, arrIDs2, axis=0)
            else:
                arrIDs = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
        else:
            arrIDs = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',1)
        if len(arrIDs) > 0:
            fltVolume = np.sum(objAnalysis.GetAtomsByID(arrIDs)[:,intVColumn])
        else: 
            fltVolume = 0
        lstSpeed.append(fltVolume/fltCrossSection)
        lstVolume.append(fltVolume)
        lstTime.append(t)
    return lstTime, lstVolume, lstPE
strFilename = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp550/u'
lstUValues = [0.005,0.01, 0.015,0.02,0.025,0.03]
strUValues = list(map(lambda s: str(s).split('.')[1], lstUValues))
for u in strUValues:
    lstFilenames = ['TJ', '12BV','13BV']
    for k in lstFilenames:
        lstTime,lstVolume,lstPE = VolumeRateChange(strFilename + u + '/'  + str(k) + '/',k, 1000, 50000, 1000,True)
        np.savetxt(strFilename + u + '/' + str(k) + '/Volume' + str(k) + '.txt', np.array([np.array(lstTime),np.array(lstVolume),np.array(lstSpeed)]))

