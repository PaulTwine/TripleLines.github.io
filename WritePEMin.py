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
strExtension = 'lst'
intSteps  = 20 #how many divisions of the neareset neighbour distance
lstNames = []
lstj = []
lstPE = []
for j in os.listdir(strDirectory):
        if j.endswith(strExtension):
            lstNames.append(j)
lstNames.sort(key=lambda test_string : list(map(int, re.findall(r'\d+', test_string)))[0])

objBase = LT.LAMMPSData(strDirectory+ lstNames[0],1,4.05, LT.LAMMPSGlobal)
objSim = objBase.GetTimeStepByIndex(-1)
fltBasePE = np.sum(objSim.GetColumnByName('c_pe1'))
intBaseAtoms = objSim.GetNumberOfAtoms()
fltPEMin = fltBasePE
lstExcessPE = []
lstj = []

for j in lstNames:
    objData = LT.LAMMPSData(strDirectory+ str(j),1,4.05, LT.LAMMPSGlobal)
    objSim = objData.GetTimeStepByIndex(-1)
    intIndex = objSim.GetColumnIndex('c_pe1')
    fltNewPE = np.sum(objSim.GetColumnByIndex(intIndex))
    fltLatticePE = np.mean(objSim.GetPTMAtoms()[:,intIndex])
    intAtoms = objSim.GetNumberOfAtoms()
    lstj.append(int(re.findall(r'\d+',j)[0]))
    lstExcessPE.append(fltNewPE-fltBasePE+fltLatticePE*(intBaseAtoms-intAtoms))

fltFactor = lstj[np.argmin(lstExcessPE)]/intSteps
np.savetxt(strDirectory+'fltFactor.txt',np.array([fltFactor])