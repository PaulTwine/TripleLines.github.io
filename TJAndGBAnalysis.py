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
from scipy import spatial
from scipy import optimize
#%%
strRoot ='/home/paul/csf4_scratch/TJ/Axis001/TJSigma5/0/'
objData = LT.LAMMPSData(strRoot + 'TJ0P.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
ids = objAnalysis.GetGrainBoundaryIDs(1)
intVolume = objAnalysis.GetColumnIndex('c_v[1]')
intPE = objAnalysis.GetColumnIndex('c_pe1')
print(objAnalysis.GetColumnNames())
vals = objAnalysis.GetAtomsByID(ids)
print(np.mean(vals[:,intPE]))
plt.hist(vals[:,intPE]+ 3.36*np.ones(len(vals)),bins = 10)
plt.show()
# %%
