import numpy as np
#import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from matplotlib import transforms
#from scipy import optimize
#from scipy import stats
#from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import LatticeDefinitions as ld
from sklearn.cluster import DBSCAN
import sys


strDirectory = str(sys.argv[1])
lstAxis = eval(str(sys.argv[2]))
arrAxis = np.array(lstAxis)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
intDirs = 10
intFiles = 10
lstStacked = []
arrQuaternion = gf.GetQuaternionFromVector(arrAxis, 0)
for j in range(100):
    arrRow = np.zeros(9)
    i = np.mod(j, 10)
    intDir = int((j-i)/10)
    arrRow[0] = intDir
    arrRow[1] = i
    objData = LT.LAMMPSData(strDirectory + str(intDir) +
                            '/GB' + str(i) + '.lst', 1, 4.05, LT.LAMMPSGlobal)
    objGB = objData.GetTimeStepByIndex(-1)
    GBIds = objGB.GetAtomIDsByOrientation(arrQuaternion, 1, 0.001)
    GBpts = objGB.GetAtomsByID(GBIds)[:,1:4]
    clustering = DBSCAN(eps=1.05*objGB.GetRealCell().GetNearestNeighbourDistance()).fit(GBpts)
    arrLabels = clustering.labels_
    arrUniqueLabels,arrCounts = np.unique(arrLabels,return_counts=True)
    arr2Labels = arrUniqueLabels[np.argsort(arrCounts)[::-1]][:2]
    arrRows = np.where(np.isin(arrLabels,arr2Labels))[0]
    GBIds = GBIds[arrRows]
    GBpts = GBpts[arrRows]     
    # ax.scatter(*tuple(zip(*GBpts)))
    # gf.EqualAxis3D(ax)
    # plt.show()
    objData = LT.LAMMPSData(strDirectory + str(intDir) +
                            '/TJ' + str(i) + '.lst', 1, 4.05, LT.LAMMPSGlobal)
    objTJ = objData.GetTimeStepByIndex(-1)
    TJIds = objTJ.GetAtomIDsByOrientation(arrQuaternion,1 ,0.001)
    TJpts = objTJ.GetAtomsByID(TJIds)[:,1:4]
    clustering = DBSCAN(eps=1.05*objTJ.GetRealCell().GetNearestNeighbourDistance()).fit(TJpts)
    arrLabels = clustering.labels_
    arrUniqueLabels,arrCounts = np.unique(arrLabels,return_counts=True)
    arr3Labels = arrUniqueLabels[np.argsort(arrCounts)[::-1]][:3]
    arrRows = np.where(np.isin(arrLabels,arr3Labels))[0]
    TJIds = TJIds[arrRows]
    TJpts = TJpts[arrRows]
    # ax.scatter(*tuple(zip(*TJpts)))
    # gf.EqualAxis3D(ax)
    # plt.show()
    intVCol = objGB.GetColumnIndex('c_v[1]')
    arrRow[2] = len(GBIds)
    arrRow[3] = len(TJIds)
    arrRow[4] = np.sum(objGB.GetAtomsByID(GBIds)[:, intVCol])
    arrRow[5] = np.sum(objTJ.GetAtomsByID(TJIds)[:, intVCol])
    arrRow[6] = np.mean(GBpts[:,2])
    arrRow[7] = np.mean(TJpts[:,2])
    arrRow[8] = np.linalg.norm(objTJ.GetCellVectors()[2])
    lstStacked.append(arrRow)
np.savetxt(strDirectory + 'CylinderValues.txt', np.vstack(lstStacked))
