import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT 
import LatticeDefinitions as ld
import sys 
import MiscFunctions as mf
import itertools as it
from sklearn.cluster import DBSCAN




#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')

#strDirectory = '/home/p17992pt/csf4_scratch/TJ/Axis001/TJSigma37/' #str(sys.argv[1])
strDirectory = str(sys.argv[1])
intDir = int(sys.argv[2])
intDelta = int(sys.argv[3])
strType = str(sys.argv[4])
strFile = strDirectory + str(intDir) + '/' + strType + str(intDelta) + '.lst'
objData = LT.LAMMPSData(strFile,1,4.05, LT.LAMMPSGlobal)
lstGrainLabels = []
intCount = 0
a = 1
blnStop = False
while not(blnStop) and a < 10:
    objTJ = objData.GetTimeStepByIndex(-1)
    objTJ.PartitionGrains(a,25,25)
    lstGrainLabels = objTJ.GetGrainLabels()
    if len(lstGrainLabels) > 0 :
        objTJ.MergePeriodicGrains(30)
        lstGrainLabels = objTJ.GetGrainLabels()
    fltWidth = objTJ.EstimateLocalGrainBoundaryWidth()
    if lstGrainLabels == list(range(5)) and fltWidth > 0:
        blnStop = True
    a += 1

print(fltWidth)
objTJ.FindGrainBoundaries(2*4.05)
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
       objTJ.FindJunctionLines(2*4.05, 3)
objTJ.WriteDumpFile(strDirectory+str(intDir) + '/' + strType + str(intDelta) + 'P.lst')

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


