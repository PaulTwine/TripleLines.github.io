# %%
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
#import GeneralLattice as gl
import LAMMPSTool as LT
#import LatticeDefinitions as ld
import sys
#import MiscFunctions as mf
#import itertools as it
#from sklearn.cluster import DBSCAN
import itertools as it
# %%
# str(sys.argv[1])
strDirectory = '/home/p17992pt/csf4_scratch/TJ/Axis101/TJSigma27/'
#strDirectory = str(sys.argv[1])
intDir = 8  # int(sys.argv[2])
intDelta = 5  # int(sys.argv[3])
strType = 'TJ'  # str(sys.argv[4])
strFile = strDirectory + str(intDir) + '/' + strType + str(intDelta) + '.lst'
objData = LT.LAMMPSData(strFile, 1, 4.05, LT.LAMMPSGlobal)
lstGrainLabels = []
intCount = 0
a = 1
blnStop = False
objTJ = objData.GetTimeStepByIndex(-1)
while not(blnStop) and a <= 10:
    objTJ.ResetGrainNumbers()
    objTJ.PartitionGrains(a, 25, 25)
    lstGrainLabels = objTJ.GetGrainLabels()
    if len(lstGrainLabels) > 0:
        objTJ.MergePeriodicGrains(30)
        lstGrainLabels = objTJ.GetGrainLabels()
    objTJ.WriteDumpFile('/home/p17992pt/Step' + strType + str(a) + '.dmp')
    fltWidth = objTJ.EstimateLocalGrainBoundaryWidth()
    if lstGrainLabels == list(range(5)) and fltWidth > 20 and fltWidth < 50:
        blnStop = True
    a += 1
# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
print(fltWidth)
lstAllMeshPoints = []
fltWidth = np.min([fltWidth, 50])
lstGBMeshPoints = objTJ.FindGrainBoundaries(3*4.05)
intGB = len(lstGBMeshPoints)
arrOverlap = objTJ.GetAtomsByID(objTJ.GetGrainBoundaryIDs(-1))[:,1:4]
ax.plot(*tuple(zip(*arrOverlap)),alpha=0.3,
                linestyle='None', marker='.', markersize=2)
plt.show()
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#ax.set_box_aspect((1,1,1))
ax.set_axis_off()
lstAllMeshPoints.extend(lstGBMeshPoints)
lstTJMeshPoints = []
if strType == 'TJ':
    lstTJMeshPoints = objTJ.FindJunctionMesh(2*4.05,3)
    lstAllMeshPoints.extend(lstTJMeshPoints)
intCounter = 0
arrLengths = list(map(lambda x: len(x), lstAllMeshPoints))
arrOrder = np.argsort(arrLengths)[::-1]
# for j in lstAllMeshPoints:
for j in lstGBMeshPoints:
    ax.plot(*tuple(zip(*j)), alpha=0.3,
                linestyle='None', marker='.', markersize=2)
if len(lstTJMeshPoints) > 0:
    for k in lstTJMeshPoints:
        plt.plot(*tuple(zip(*k)), alpha=1, linestyle='None',
                 marker='.', markersize=2, c='black')
# for i in arrOrder:
#     j = lstAllMeshPoints[i]
#    # j = objTJ.WrapVectorIntoSimulationBox(j)
#     if intCounter < intGB:
#         ax.plot(*tuple(zip(*j)), alpha=0.3,
#                 linestyle='None', marker='.', markersize=2)
#     else:
#         plt.plot(*tuple(zip(*j)), alpha=1, linestyle='None',
#                  marker='.', markersize=2, c='black')
#     intCounter += 1

gf.EqualAxis3D(ax)
ax.set_zlim3d([0,4*objTJ.GetCellVectors()[2,2]])
plt.show()

if strType == 'TJ':
     objTJ.FindJunctionLines(3*4.05, 3)
objTJ.WriteDumpFile(strDirectory+str(intDir) + '/' +
                    strType + str(intDelta) + 'P.lst')


# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_axis_off()
arrOverlap = objTJ.GetAtomsByID(objTJ.GetGrainBoundaryIDs(-1))[:,1:4]
ax.plot(*tuple(zip(*arrOverlap)),
                linestyle='None', marker='.', markersize=2,c='darkgrey')
gf.EqualAxis3D(ax)
ax.set_zlim3d([0,4*objTJ.GetCellVectors()[2,2]])
plt.show()
# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_axis_off()
if len(lstTJMeshPoints) > 0:
    for k in lstTJMeshPoints:
        plt.plot(*tuple(zip(*k)), alpha=1, linestyle='None',
                 marker='.', markersize=2, c='black')
gf.EqualAxis3D(ax)
ax.set_zlim3d([0,4*objTJ.GetCellVectors()[2,2]])
plt.show()
# %%
