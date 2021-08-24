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
intDirs = 10
intFiles = 10
lstStacked = []
for j in range(100):
    arrRow = np.zeros(9)
    i = np.mod(j,10) 
    intDir = int((j-i)/10)
    arrRow[0] = intDir
    arrRow[1] = i
    objData = LT.LAMMPSData(strDirectory+ str(intDir) + '/left' +str(i) + '.lst',1,4.05, LT.LAMMPSGlobal)
    objLeft = objData.GetTimeStepByIndex(-1)
    objData = LT.LAMMPSData(strDirectory+ str(intDir) + '/right' + str(i) + '.lst',1,4.05, LT.LAMMPSGlobal)
    objRight = objData.GetTimeStepByIndex(-1)
    objData = LT.LAMMPSData(strDirectory+ str(intDir) + '/TJ' + str(i) + '.lst',1,4.05, LT.LAMMPSGlobal)
    objTJ = objData.GetTimeStepByIndex(-1)
    intPECol = objLeft.GetColumnIndex('c_pe1')
    arrRow[2] = np.sum(objLeft.GetColumnByName('c_pe1'))
    arrRow[3] = np.sum(objRight.GetColumnByName('c_pe1'))
    arrRow[4] = np.sum(objTJ.GetColumnByName('c_pe1'))
    arrRow[4] = np.mean(objLeft.GetLatticeAtoms()[:,intPECol])
    arrRow[5] = np.mean(objRight.GetLatticeAtoms()[:,intPECol])
    arrRow[5] = np.mean(objTJ.GetLatticeAtoms()[:,intPECol])
    arrRow[6] = objLeft.GetNumberOfAtoms()
    arrRow[7] = objRight.GetNumberOfAtoms()
    arrRow[8] = objTJ.GetNumberOfAtoms()
    lstStacked.append(arrRow)
np.savetxt(strDirectory + 'Values.txt',np.vstack(lstStacked))