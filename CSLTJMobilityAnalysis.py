import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
import copy as cp
from scipy import spatial
import MiscFunctions
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = plt.axes(projection='3d')

strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis221/Sigma9_9_9/'

objData = LT.LAMMPSData(strRoot + 'TJ/1Sim1100.dmp',1, 4.05,LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
objLT.PartitionGrains(0.99, 5000,100)
objLT.MergePeriodicGrains(25)
arrCellVectors = objLT.GetCellVectors()
ids = objLT.FindMeshAtomIDs([1,2])
ptsTJ = objLT.GetAtomsByID(ids)[:,1:4]

objData = LT.LAMMPSData(strRoot + '12BV/2Min.lst',1, 4.05,LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
objLT.PartitionGrains(0.99, 5000,100)
objLT.MergePeriodicGrains(25)
ids = objLT.FindMeshAtomIDs([1,2])
pts12BV = objLT.GetAtomsByID(ids)[:,1:4] +0.5*arrCellVectors[0]

objData = LT.LAMMPSData(strRoot + '13BV/2Min.lst',1, 4.05,LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
objLT.PartitionGrains(0.99, 5000,100)
objLT.MergePeriodicGrains(25)
ids = objLT.FindMeshAtomIDs([1,2])
pts13BV = objLT.GetAtomsByID(ids)[:,1:4] 

objData = LT.LAMMPSData(strRoot + '32BH/2Min.lst',1, 4.05,LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
objLT.PartitionGrains(0.99, 5000,100)
objLT.MergePeriodicGrains(25)
ids = objLT.FindMeshAtomIDs([1,2])
pts32BH = objLT.GetAtomsByID(ids)[:,1:4] +0.5*arrCellVectors[1]

ax.scatter(*tuple(zip(*pts12BV)),s=0.04)
ax.scatter(*tuple(zip(*pts13BV)),s=0.04)
ax.scatter(*tuple(zip(*pts32BH)),s=0.04)
ax.scatter(*tuple(zip(*ptsTJ)),s=0.04)

#plt.scatter(*tuple(zip(*pts2)))
#gf.EqualAxis3D(ax)
plt.show()
