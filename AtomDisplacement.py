import numpy as np
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
lstFilenames = []
for j in os.listdir(strHome):
    if j.endswith('.dmp'):
        lstFilenames.append(j)
lstFilenames = sorted(lstFilenames, key=lambda x:float(re.findall("(\d+)",x)[0]))        
objData = LT.LAMMPSData(strHome + lstFilenames[0], 1, 4.05, LT.LAMMPSAnalysis3D)
objTimeStep = objData.GetTimeStepByIndex(-1)
arrPrevious=objTimeStep.GetAtomData()[:,0:4]
arrPrevious=arrPrevious[np.argsort(arrPrevious[:,0])]
arrPeriodic=np.zeros([objTimeStep.GetNumberOfAtoms(),3])    
for k in lstFilenames[1:]:
    strName = strHome + k
    objData = LT.LAMMPSData(strName, 1, 4.05, LT.LAMMPSAnalysis3D)
    objTimeStep = objData.GetTimeStepByIndex(-1)
    arrNew = objTimeStep.GetAtomData()[:,0:4]
    arrNew = arrNew[np.argsort(arrNew[:,0])]
    arrDifference = arrNew[:,1:4]- arrPrevious[:,1:4]
   # arrPeriodic += objTimeStep.PeriodicShiftAllCloser(np.zeros(3),arrDifference) 
    arrPeriodic += gf.PeriodicAllMinDisplacement(arrDifference,objTimeStep.GetCellVectors(), np.array([0,1,2]))
    arrPrevious = arrNew
np.savetxt(strHome + 'TotalDisplacements.txt', arrPeriodic)


# # arrFirst =  np.loadtxt(strHome + 'TwoCell/Axis111/TJSigma19/4/TJ7.dat',skiprows=9,usecols=(0,2,3,4))
# # arrFirst = arrFirst[np.argsort(arrFirst[:,0])]
# # arrDifference = arrLast[:,1:4] - arrFirst[:,1:4]
# # arrPeriodic = objTimeStep.PeriodicShiftAllCloser(np.zeros(3),arrDifference) 
# arrDistances = np.linalg.norm(arrPeriodic,axis=1)
# arrRows = np.where(arrDistances > 4.05/np.sqrt(2))[0]
# arrDistances = arrDistances[arrRows]
# plt.scatter(list(range(len(arrDistances))),arrDistances, s=0.1)
# plt.show()
# #print((objTimeStep.GetNumberOfAtoms() -len(arrDistances))/objTimeStep.GetNumberOfAtoms())
# fig = plt.figure(figsize=plt.figaspect(1)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
# ax = fig.gca(projection='3d')
# ax.scatter(*tuple(zip(*arrNew[arrRows,1:4])),s=0.1)
# #ax.scatter(*tuple(zip(*arrLast[arrRows,1:4])))
# #plt.xlim([0,np.linalg.norm(objTimeStep.GetCellVectors()[0])])
# #plt.ylim([0,np.linalg.norm(objTimeStep.GetCellVectors()[1])])
# #gf.EqualAxis3D(ax)
# plt.show()

# print(np.mean(np.linalg.norm(arrPeriodic,axis=1)))