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
strDirectory = '/home/p17992pt/csf4_scratch/TJ/Axis111/TJSigma31/'
#strDirectory = str(sys.argv[1])
intDir = 0  # int(sys.argv[2])
intDelta = 0  # int(sys.argv[3])
strType = 'GB'  # str(sys.argv[4])
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
    fltWidth = objTJ.EstimateLocalGrainBoundaryWidth()
    if lstGrainLabels == list(range(5)) and fltWidth > 0 and fltWidth < 50:
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
#fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#ax.set_box_aspect((1,1,1))
ax.set_axis_off()
lstAllMeshPoints.extend(lstGBMeshPoints)
if strType == 'TJ':
    lstTJMeshPoints = objTJ.FindJunctionMesh(2*4.05,3)
    lstAllMeshPoints.extend(lstTJMeshPoints)
intCounter = 0
arrLengths = list(map(lambda x: len(x), lstAllMeshPoints))
arrOrder = np.argsort(arrLengths)[::-1]
for i in arrOrder:
    j = lstAllMeshPoints[i]
   # j = objTJ.WrapVectorIntoSimulationBox(j)
    if intCounter < intGB:
        ax.plot(*tuple(zip(*j)), alpha=0.3,
                linestyle='None', marker='.', markersize=10)
    else:
        plt.plot(*tuple(zip(*j)), alpha=1, linestyle='None',
                 marker='.', markersize=10, c='black')
    intCounter += 1

gf.EqualAxis3D(ax)
ax.set_zlim3d([0,20])
plt.show()
# %%
# lstTJs = []
# lstpts = objTJ.FindJunctionMesh(2*4.05,3)
# for i in lstpts:
#     ax.scatter(*tuple(zip(*i)))
# plt.show()
# lstGrainLabels.remove(0)
# lstTwos = it.combinations(lstGrainLabels,2)
# for i in lstTwos:
#     pts = objTJ.FindDefectiveMesh(i[0],i[1])
#     if len(pts) > 0:
#         ax.set_axis_off()
#         ax.scatter(*tuple(zip(*pts)))
# plt.show()
if strType == 'TJ':
    objTJ.FindJunctionLines(3*4.05, 3)
objTJ.WriteDumpFile(strDirectory+str(intDir) + '/' +
                    strType + str(intDelta) + 'P.lst')

# if len(pts) > 0:
#     ax.scatter(*tuple(zip(*pts)),c='r')
# plt.show()

# if strType == 'GB':
#     objTJ.WriteDumpFile(strDirectory+str(intDir) + '/GB' + str(intDelta) + 'P.lst')
# elif lstGrainLabels == list(range(1,5)) and strType == 'TJ':
#     objTJ.AddColumn(np.zeros([objTJ.GetNumberOfAtoms(),1]),'TripleLine', strFormat = '%i')
#     intTJ = objTJ.GetColumnIndex('TripleLine')
#     lstThrees = list(it.combinations(lstGrainLabels, 3))
#     t = 1
#     for i in lstThrees:
#         ids,mpts = objTJ.FindMeshAtomIDs(i,fltWidth/2)
#        # ids,mpts = objTJ.FindJunctionMeshAtoms(fltWidth/2,i)
#         # if len(mpts)>0:
#         #     ax.scatter(*tuple(zip(*mpts)))
#         #     plt.show()
#         if len(ids) > 0:
#             pts = objTJ.GetAtomsByID(ids)[:,1:4]
#             clustering = DBSCAN(4.05,min_samples=10).fit(pts)
#             arrLabels = clustering.labels_
#             lstSplitIDs = []
#             lstSplitPoints = []
#             for a in np.unique(arrLabels):
#                 if a != -1:
#                     arrRows = np.where(arrLabels == a)[0]
#                     lstSplitIDs.append(np.array(ids)[arrRows])
#                     lstSplitPoints.append(pts[arrRows])
#             lstMatches = gf.GroupClustersPeriodically(lstSplitPoints, objTJ.GetCellVectors(),2*4.05)
#             for l in lstMatches:
#                 lstMergedIDs = []
#                 for m in l:
#                     lstMergedIDs.append(lstSplitIDs[m])
#                 lstMergedIDs = np.unique(np.concatenate(lstMergedIDs))
#                 objTJ.SetColumnByIDs(lstMergedIDs,intTJ,t*np.ones(len(lstMergedIDs)))
#                 lstTJs.append(ids)
#                 t +=1
#         lstMatches = []
#     lstAllTJIDs = mf.FlattenList(lstTJs)
#     lstAllTJIDs = list(np.unique(lstAllTJIDs))
#     intGB = objTJ.GetColumnIndex('GrainBoundary')
#     objTJ.SetColumnByIDs(lstAllTJIDs,intGB,0*np.ones(len(lstAllTJIDs)))
#    objTJ.WriteDumpFile(strDirectory+str(intDir) + '/TJ' + str(intDelta) + 'P.lst')
