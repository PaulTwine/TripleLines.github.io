import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp
from scipy import spatial
import os 
import re


strHome = sys.argv[1]
strStart = sys.argv[2]

lstFilenames = []
for j in os.listdir(strHome):
    if j.endswith('.dmp') and j.startswith(strStart):        
        lstFilenames.append(j)
lstFilenames = sorted(lstFilenames, key=lambda x:float(re.findall("(\d+)",x)[0]))        
objData = LT.LAMMPSData(strHome + lstFilenames[0], 1, 4.05, LT.LAMMPSAnalysis3D)
objTimeStep = objData.GetTimeStepByIndex(-1)
arrPrevious=objTimeStep.GetAtomData()[:,0:4]
arrPrevious=arrPrevious[np.argsort(arrPrevious[:,0])]
arrPeriodic=np.zeros([objTimeStep.GetNumberOfAtoms(),4])    
for k in lstFilenames[1:]:
    strName = strHome + k
    objData = LT.LAMMPSData(strName, 1, 4.05, LT.LAMMPSAnalysis3D)
    objTimeStep = objData.GetTimeStepByIndex(-1)
    arrNew = objTimeStep.GetAtomData()[:,0:4]
    arrNew = arrNew[np.argsort(arrNew[:,0])]
    arrDifference = arrNew[:,1:4]- arrPrevious[:,1:4]
    arrPoints, arrDistance =  gf.PeriodicMinDisplacement(arrPrevious, objTimeStep.GetCellVectors())
    arrPeriodic[:,:3] += arrPoints
    arrPeriodic[:,-1] += arrDistance
    arrPrevious = arrNew
np.savetxt(strHome + strStart + 'Displacements.txt', arrPeriodic)

