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
strFile = str(sys.argv[2])
intTimeStep = int(sys.argv[3])
objData = LT.LAMMPSData(strDirectory + strFile, 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
lstColumnNames = objAnalysis.GetColumnNames()
while 'GrainBoundary' in lstColumnNames:
    objAnalysis.DeleteColumnByName('GrainBoundary')
    lstColumnNames = objAnalysis.GetColumnNames()
while 'TripleLine' in lstColumnNames:
    objAnalysis.DeleteColumnByName('TripleLine')
    lstColumnNames = objAnalysis.GetColumnNames()
intEco =1
arrIDs1 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',intEco)
arrIDs2 =  objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]',-intEco)
arrIDs3 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_2[2]',-intEco)
if (len(arrIDs1) > 0) and (len(arrIDs2) > 0) and (len(arrIDs3) > 0):
    lstOverlap = []
    lstOverlap.extend(list(set(arrIDs1).intersection(arrIDs2.tolist())))
    lstOverlap.extend(list(set(arrIDs1).intersection(arrIDs3.tolist())))
    lstOverlap.extend(list(set(arrIDs2).intersection(arrIDs3.tolist())))
    arrPTMIDs = objAnalysis.GetNonPTMAtomIDs()
    lstOverlap.extend(arrPTMIDs)
    lstGrainZeroIDs = objAnalysis.GetGrainAtomIDs(0)
    lstOverlap.extend(lstGrainZeroIDs)
    lstOverlap = np.unique(lstOverlap).tolist()
    arrIDs1 = np.array(list(set(arrIDs1).difference(lstOverlap)))
    arrIDs2 = np.array(list(set(arrIDs2).difference(lstOverlap)))
    arrIDs3 = np.array(list(set(arrIDs3).difference(lstOverlap)))
    objAnalysis.SetMaxGBWidth(4*4.05)
    objAnalysis.SetPeriodicGrain(1,arrIDs1, 25)
    objAnalysis.SetPeriodicGrain(2,arrIDs2, 25)
    objAnalysis.SetPeriodicGrain(3,arrIDs3, 25)
    objAnalysis.SetPeriodicGrain(0,lstOverlap,25)
    objAnalysis.FindGrainBoundaries(3*4.05)
    arrTJMesh = objAnalysis.FindJunctionMesh(3*4.05,3)
    objAnalysis.FindJunctionLines(3*4.05,3)
    for i in range(len(arrTJMesh)):
        np.savetxt(strDirectory + 'TJ' + str(i) + 'Mesh' + str(intTimeStep) +  '.txt', arrTJMesh[i])
    objAnalysis.WriteDumpFile(strDirectory + strFile)
    



