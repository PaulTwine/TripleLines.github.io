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

strRoot ='/home/p17992pt/csf3_scratch/CSLGrowthCylinder/Axis100/GBSigma13/0/'
objdct = dict()
intSigma = 13
arrAxis = np.array([0,0,1])
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
gf.CubicCSLGenerator(arrAxis, 5)
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
objData = LT.LAMMPSData(strRoot+ 'read4.lst',1,4.05, LT.LAMMPSGlobal)
objSim = objData.GetTimeStepByIndex(-1)
print(objSim.GetColumnNames())
arrPoints = np.unique(np.round(objSim.GetLatticeAtoms()[:,9:14],2),axis=0)
print(len(arrPoints))
#arrIDs = objSim.GetAtomData()[:,0]
arrQuaternion = gf.GetQuaternionFromVector(np.array([0,0,1]),0)
#arrQuaternion = gf.GetQuaternionFromVector(np.array([0,0,1]),fltAngle1)
#arrQuaternion = gf.GetQuaternionFromVector(np.array([0,0,1]),fltAngle2)
arrIDs = objSim.GetAtomIDsByOrientation(arrQuaternion,1,0.001)
#arrRows = np.where((np.abs(np.matmul(arrPoints[:,1:5],arrQuaternion)) > 0.995) & (arrPoints[:,0] != 0))
#print(arrRows)
#arrIDs = arrIDs[arrRows]
arrPlot = objSim.GetAtomsByID(arrIDs)[:,1:4]
fltVolume = np.sum(objSim.GetAtomsByID(arrIDs)[:,objSim.GetColumnIndex('c_v[1]')])
fltRadius = np.sqrt(fltVolume/(np.linalg.norm(objSim.GetCellVectors()[:,2])*np.pi))
print(fltVolume,fltRadius/4.05)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*tuple(zip(*arrPlot)))
gf.EqualAxis3D(ax)
plt.show()
# for j in os.listdir(strRoot):
#     if j.endswith('.lst'):
#         objData = LT.LAMMPSData(strRoot+ str(j),1,4.05, LT.LAMMPSGlobal)
#         objHex = objData.GetTimeStepByIndex(-1)
#         objHex.ReadInDefectData(strRoot + str(j[:-3]) + 'dfc')
#         objdct[j] = objHex
