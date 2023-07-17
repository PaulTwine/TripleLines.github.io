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

strDirectory = str(sys.argv[1])
intDir =  int(sys.argv[2])
intDelta = int(sys.argv[3])
strType = str(sys.argv[4])
strFile = strDirectory + str(intDir) + '/' + strType + str(intDelta) + '.lst'
objData = LT.LAMMPSData(strFile,1,4.05, LT.LAMMPSGlobal)
objTJ = objData.GetTimeStepByIndex(-1)
objTJ.PartitionGrains(0.999,25,25)
objTJ.MergePeriodicGrains(30)
arrIDs = []
lstTemp = []
lstGrainLabels = objTJ.GetGrainLabels() 
fltWidth = objTJ.EstimateLocalGrainBoundaryWidth()
print(fltWidth)
lstTJs = []
objTJ.FindGrainBoundaries(fltWidth/2)
if strType == 'GB':
    objTJ.WriteDumpFile(strDirectory+str(intDir) + '/GB' + str(intDelta) + 'P.lst')
elif lstGrainLabels == list(range(5)) and strType == 'TJ':
    lstGrainLabels.remove(0)
    objTJ.AddColumn(np.zeros([objTJ.GetNumberOfAtoms(),1]),'TripleLine', strFormat = '%i')
    intTJ = objTJ.GetColumnIndex('TripleLine')
    lstThrees = list(it.combinations(lstGrainLabels, 3))
    t = 1
    for i in lstThrees:
        ids,mpts = objTJ.FindMeshAtomIDs(i,fltWidth/2)
        #ids,mpts = objTJ.FindJunctionMeshAtoms(fltWidth/2,i)
        # if len(mpts)>0:
        #     ax.scatter(*tuple(zip(*mpts)))
        #     plt.show()
        if len(ids) > 0:
            pts = objTJ.GetAtomsByID(ids)[:,1:4]
            clustering = DBSCAN(2*4.05,min_samples=10).fit(pts)
            arrLabels = clustering.labels_
            lstSplitIDs = []
            lstSplitPoints = []
            for a in np.unique(arrLabels):
                if a != -1:
                    arrRows = np.where(arrLabels == a)[0]
                    lstSplitIDs.append(np.array(ids)[arrRows])
                    lstSplitPoints.append(pts[arrRows])
            lstMatches = gf.GroupClustersPeriodically(lstSplitPoints, objTJ.GetCellVectors(),2*4.05)
            for l in lstMatches:
                lstMergedIDs = []
                for m in l:
                    lstMergedIDs.append(lstSplitIDs[m])
                lstMergedIDs = np.unique(np.concatenate(lstMergedIDs))
                objTJ.SetColumnByIDs(lstMergedIDs,intTJ,t*np.ones(len(lstMergedIDs)))
                lstTJs.append(ids)
                t +=1
        lstMatches = []
    lstAllTJIDs = mf.FlattenList(lstTJs)
    lstAllTJIDs = list(np.unique(lstAllTJIDs))
    intGB = objTJ.GetColumnIndex('GrainBoundary')
    objTJ.SetColumnByIDs(lstAllTJIDs,intGB,0*np.ones(len(lstAllTJIDs)))
    objTJ.WriteDumpFile(strDirectory+str(intDir) + '/TJ' + str(intDelta) + 'P.lst')


