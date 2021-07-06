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
#arrValues = np.loadtxt(strDirectory +'Values.txt')
#print(arrValues)
#plt.scatter(list(range(100)),arrValues[0,:])
#plt.show()
#print(np.argmax(arrValues[0,:]))
lstPE = []
lstAtoms = []
lstLattice = []
for j in range(100):
    objData = LT.LAMMPSData(strDirectory+ str(j) + '/read0.lst',1,4.05, LT.LAMMPSGlobal)
    obj0 = objData.GetTimeStepByIndex(-1)
    objData = LT.LAMMPSData(strDirectory+ str(j) + '/read1.lst',1,4.05, LT.LAMMPSGlobal)
    obj1 = objData.GetTimeStepByIndex(-1)
    lstAtoms.append(obj0.GetNumberOfAtoms() - obj1.GetNumberOfAtoms())
    intPECol = obj1.GetColumnIndex('c_pe1')
    fltSum1 = np.sum(obj1.GetColumnByName('c_pe1'))
    fltSum0 = np.sum(obj0.GetColumnByName('c_pe1'))
    lstLattice.append(np.mean(obj1.GetLatticeAtoms()[:,intPECol]))
    lstPE.append(fltSum1-fltSum0 + lstAtoms[-1]*lstLattice[-1])
print(max(lstPE))
np.savetxt(strDirectory + 'Values.txt',np.array([lstPE,lstAtoms,lstLattice]))