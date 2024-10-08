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
from sklearn.cluster import DBSCAN


#strDirectory =  str(sys.argv[1])
#strFile = str(sys.argv[2])
#intTimeStep = int(sys.argv[3])

a= 4.05
strDirectory = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp600/u03L/TJ/'
strFile = '1Sim90000.dmp'  # str(sys.argv[2])
intTimeStep = 90000  # int(sys.argv[3])
objData = LT.LAMMPSData(strDirectory + strFile, 1, a, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
lstColumnNames = objAnalysis.GetColumnNames()
lstNames = ['GrainNumber', 'GrainBoundary', 'TripleLine']
arrCellVectors = objAnalysis.GetCellVectors()
arrTJPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
objTJTree = gf.PeriodicWrapperKDTree(arrTJPositions,arrCellVectors, gf.FindConstraintsFromBasisVectors(arrCellVectors),20)
for k in lstNames:
    if k in lstColumnNames:
        objAnalysis.SetColumnToZero(k)
intEco = 1
arrIDs1 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]', intEco)
arrIDs2 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_1[2]', -intEco)
arrIDs3 = objAnalysis.GetGrainAtomIDsByEcoOrient('f_2[2]', -intEco)
lstAllZeroGrainIDs = []
lstGrainLabels = []
blnStop = False
b = 1
while not(blnStop) and b < 6:
    objAnalysis.ResetGrainNumbers()
    objAnalysis.PartitionGrains(b, 75, 5*a)
    lstGrainLabels = objAnalysis.GetGrainLabels()
    if len(lstGrainLabels) > 0:
        objAnalysis.MergePeriodicGrains(25)
        lstGrainLabels = objAnalysis.GetGrainLabels()
    fltWidth = objAnalysis.EstimateLocalGrainBoundaryWidth()
    if lstGrainLabels == list(range(4)):
        blnStop = True
    b += 1

if blnStop:
    lstMatches = objAnalysis.MatchPreviousGrainNumbers(
        [1, 2, 3], [arrIDs1, arrIDs2, arrIDs3])
    objAnalysis.RelabelGrainNumbers([1, 2, 3], lstMatches)
    objAnalysis.FindGrainBoundaries(3*a)
    arrTJMesh = objAnalysis.FindJunctionMesh(3*a, 3)
    objAnalysis.FindJunctionLines(3*a, 3,5*a)
    for i in range(len(arrTJMesh)):
        np.savetxt(strDirectory + 'TJ' + str(i) + 'Mesh' +
                   str(intTimeStep) + '.txt', arrTJMesh[i])
    objAnalysis.WriteDumpFile(strDirectory + strFile)

