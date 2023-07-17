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


strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp600/u04/'
fig = plt.figure()
ax = plt.axes(projection='3d')

objCSL = gl.CSLTripleLine(np.array([1,1,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(75)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[3,7,21],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,True)

# arrBasis1 = objCSL.GetLatticeBasis(0)
# arrBasis2 = objCSL.GetLatticeBasis(1)
# arrBasis3 = objCSL.GetLatticeBasis(2)
# arrQuaternion1 = gf.GetQuaternionFromBasisMatrix(arrBasis1)
# arrQuaternion2 = gf.GetQuaternionFromBasisMatrix(arrBasis2)
# arrQuaternion3 = gf.GetQuaternionFromBasisMatrix(arrBasis3)


objData = LT.LAMMPSData(strRoot + 'TJ/1Sim50000.dmp',1, 4.05,LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
# ids1 = objLT.GetAtomIDsByOrientation(arrQuaternion1,1,0.05)
# pts1 = objLT.GetAtomsByID(ids1)[:,1:4]
# ids2 = objLT.GetAtomIDsByOrientation(arrQuaternion2,1,0.05)
# pts2 = objLT.GetAtomsByID(ids2)[:,1:4]
# ids3 = objLT.GetAtomIDsByOrientation(arrQuaternion3,1,0.05)
# pts3 = objLT.GetAtomsByID(ids3)[:,1:4]

ids2 = objLT.GetAtomIDsByOrderParameter(1)[1]
pts2 = objLT.GetAtomsByID(ids2)[:,1:4]
ids3 = objLT.GetAtomIDsByOrderParameter(2)[1]
pts3 = objLT.GetAtomsByID(ids3)[:,1:4]

ids = list(set(ids3).intersection(ids2))
pts1 = objLT.GetAtomsByID(ids)[:,1:4]



#ax.scatter(*tuple(zip(*pts3)))
#ax.scatter(*tuple(zip(*pts2)))
ax.scatter(*tuple(zip(*pts1)))
plt.show()