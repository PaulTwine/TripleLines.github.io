#%%
import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D
import copy as cp
#%%
strFile = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp600/u03/TJ/'

objData = LT.LAMMPSData(strFile+'1Sim25000.dmp',1,4.05,LT.LAMMPSAnalysis3D)
objTJ = objData.GetTimeStepByIndex(-1)
#%%
lstIDs = []
lstV =[]
lstPE =[]
lstStress = []
intV = objTJ.GetColumnIndex('c_v[1]')
intPE = objTJ.GetColumnIndex('c_pe1')
for i in range(4):
    lstIDs.append(objTJ.GetTripleLineIDs(i))
    lstV.append(np.mean(objTJ.GetAtomsByID(lstIDs[-1])[:,intV]))
    lstPE.append(np.mean(objTJ.GetAtomsByID(lstIDs[-1])[:,intPE]))
# %%
plt.scatter(range(4),lstV)
plt.show()
plt.scatter(range(4),lstPE)
plt.show()

# %%
