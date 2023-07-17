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

strDirectory = '/home/p17992pt/csf4_scratch/CSLTJ/Axis001/Sigma5_5_25/Min/' #str(sys.argv[1])
intDirs = 10
intFiles = 10
lstStacked = []
arrRow = np.zeros(8)
lstNames = ['TJ123.lst','G1.lst','G2.lst', 'BV12.lst','BV13.lst','BH32.lst']
dctTimeSteps = dict()
i = 0
lstPETJ = []
lstAtomsTJ = []
lstPEGB = []
lstAtomsGB = []
for j in lstNames:
    objData = LT.LAMMPSData(strDirectory + j,1,4.05, LT.LAMMPSGlobal)
    objTimeStep = objData.GetTimeStepByIndex(-1)
    if j == 'TJ123.lst':
        arrCellVectors = objTimeStep.GetCellVectors()
        #objTimeStep.WriteDataFile(strDirectory + 'mob.dat')
    if i < 3:
        lstPETJ.append(np.sum(objTimeStep.GetColumnByName('c_pe1')))
        lstAtomsTJ.append(objTimeStep.GetNumberOfAtoms())
    else:
        lstPEGB.append(np.sum(objTimeStep.GetColumnByName('c_pe1')))
        lstAtomsGB.append(objTimeStep.GetNumberOfAtoms())
    dctTimeSteps[j] = objTimeStep
    i += 1
fltVArea = np.linalg.norm(np.cross(arrCellVectors[1],arrCellVectors[2]))
fltHArea = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))    
fltH = np.linalg.norm(objTimeStep.GetCellVectors()[2])
fltLineTension = (np.sum(lstPETJ)-np.sum(lstPEGB) +(-3.36*(np.sum(lstAtomsGB)-np.sum(lstAtomsTJ))))/fltH

flt12Excess = (lstPEGB[0] + 3.36*lstAtomsGB[0])/fltHArea
flt13Excess = (lstPEGB[1] + 3.36*lstAtomsGB[1])/fltHArea
flt32Excess = (lstPEGB[2] + 3.36*lstAtomsGB[2])/fltVArea

print(fltLineTension, flt12Excess, flt32Excess,flt13Excess, flt12Excess/flt13Excess)

