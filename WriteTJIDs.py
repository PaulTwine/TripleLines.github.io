import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy import optimize
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT 
import LatticeDefinitions as ld
import re
import sys 
import MiscFunctions as mf
import itertools as it
from sklearn.cluster import DBSCAN



objData = LT.LAMMPSData('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp550/u04/TJ/1Sim95000.dmp',1,4.05, LT.LAMMPSGlobal)
objTJ = objData.GetTimeStepByIndex(-1)
objTJ.PartitionGrains(0.995)
objTJ.MergePeriodicGrains(30)
print(objTJ.GetGrainLabels())

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

strDirectory = '/home/p17992pt/csf4_scratch/TJ/Axis001/TJSigma5/' #str(sys.argv[1])

intDirs = 10
intFiles = 10
lstStacked = []
intMax = 0
intDir = 0
intDelta = 6
strFile = strDirectory + str(intDir) + '/TJ' + str(intDelta) + '.lst'
objData = LT.LAMMPSData(strFile,1,4.05, LT.LAMMPSGlobal)
objTJ = objData.GetTimeStepByIndex(-1)
objTJ.PartitionGrains(0.995)
objTJ.MergePeriodicGrains(30)
arrIDs = []
lstTemp = []
lstGrainLabels = objTJ.GetGrainLabels() 
fltWidth = objTJ.EstimateLocalGrainBoundaryWidth()
print(fltWidth)
lstTJs = []
objTJ.AddColumn(np.zeros([objTJ.GetNumberOfAtoms(),1]),'GrainBoundary', strFormat = '%i')
objTJ.AddColumn(np.zeros([objTJ.GetNumberOfAtoms(),1]),'TripleLine', strFormat = '%i')
intGB = objTJ.GetColumnIndex('GrainBoundary')
intTJ = objTJ.GetColumnIndex('TripleLine')
if lstGrainLabels == list(range(5)):
    lstGrainLabels.remove(0)
    lstThrees = list(it.combinations(lstGrainLabels, 3))
    t = 1
    for i in lstThrees:
        ids = objTJ.FindMeshAtomIDs(i,25)
        pts = objTJ.GetAtomsByID(ids)[:,1:4]
        if len(ids) > 0:
            clustering = DBSCAN(eps=1.05*objTJ.GetRealCell().GetNearestNeighbourDistance(),min_samples=5).fit(pts)
            print(clustering.labels_)
            objTJ.SetColumnByIDs(ids,intTJ,t*np.ones(len(ids)))
            lstTJs.append(ids)
            t +=1
for j in lstTJs:
    pts = objTJ.GetAtomsByID(j)[:,1:4]
  #  pts = gf.MergeTooCloseAtoms(pts,objTJ.GetCellVectors(),4.05/np.sqrt(2))
    #ax.scatter(pts[:,0], pts[:,1],pts[:,2])
#gf.EqualAxis3D(ax)
arrCellVectors = objTJ.GetCellVectors()
# plt.xlim([0,arrCellVectors[0,0]])
# plt.ylim([0,arrCellVectors[1,1]])
# plt.show()
# print(pts)
lstAllTJIDs = mf.FlattenList(lstTJs)
lstGBs = []
if len(lstTJs) > 0:
    lstTwos = list(it.combinations(lstGrainLabels,2))
    g=1
    for k in lstTwos:
        ids2 = objTJ.FindMeshAtomIDs(k,25)
        ids2 = list(set(ids2).difference(lstAllTJIDs))
        if len(ids2)> 0:
            objTJ.SetColumnByIDs(ids2,intGB,g*np.ones(len(ids2)))
            lstGBs.append(ids2)
            g +=1
for l in lstGBs:
    pts2 = objTJ.GetAtomsByID(l)[:,1:4]
    ax.scatter(pts2[:,0],pts2[:,1],pts2[:,2])
plt.xlim([0,arrCellVectors[0,0]])
plt.ylim([0,arrCellVectors[1,1]])
plt.show()
objTJ.WriteDumpFile(strDirectory+'TJ0.lst')


# if  lstGrainLabels == [0,1,2,3,4]:
#     lstMeshPoints = []
#     intLength = len(lstGrainLabels)
#     for i in range(1,intLength):
#         for j in range(i+1, intLength):
#             arrMesh = objTJ.FindDefectiveMesh(i,j)
#             if len(arrMesh) > 0:
#                 lstMeshPoints.append(arrMesh)
# #    lstTJMeshPoints = gf.FindIntersectionsNPointSets(lstMeshPoints, objTJ.GetCellVectors(),2*fltWidth,3)
# #arrTJMeshPoints = np.unique(np.vstack(lstMeshPoints),axis=0)
# #arrTJMeshPoints = gf.MergeTooCloseAtoms(arrTJMeshPoints,objTJ.GetCellVectors(),4.05/np.sqrt(2))
# for i in lstMeshPoints:
#     ax.scatter(*tuple(zip(*i)))
# plt.show()
#np.savetxt(strDirectory + str(intDir) + '/TJMesh' + str(intDelta) + '.txt',arrTJMeshPoints)
