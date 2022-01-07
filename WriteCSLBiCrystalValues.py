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

strDirectory = str(sys.argv[1])
intFiles = 10
lstStacked = []
for j in range(intFiles):
    arrRow = np.zeros(5)
    arrRow[0] = 0.1*j
    objData = LT.LAMMPSData(strDirectory + 'GB' +str(j) + '.lst',1,4.05, LT.LAMMPSGlobal)
    objGB = objData.GetTimeStepByIndex(-1)
    intPECol = objGB.GetColumnIndex('c_pe1')
    arrCellVectors = objGB.GetCellVectors()
    arrRow[1] = np.sum(objGB.GetColumnByName('c_pe1'))
    arrRow[2] = np.mean(objGB.GetLatticeAtoms()[:,intPECol])
    arrRow[3] = objGB.GetNumberOfAtoms()
    arrRow[4] = np.linalg.norm(np.cross(arrCellVectors[1],arrCellVectors[2]))
    lstStacked.append(arrRow)
np.savetxt(strDirectory + 'Values.txt',np.vstack(lstStacked),fmt='%f')