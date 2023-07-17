#%%
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
#%%
strDir = '/home/p17992pt/csf4_scratch/TJ/Axis001/TJSigma5/' #str(sys.argv[1])
intDirs = 10
intFiles = 10
lstStacked = []
intMax = 0
arrIDs = np.loadtxt(strDir+ 'TJIDs.txt')
for j in range(100):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    i = np.mod(j,10) 
    intDir = int((j-i)/10)
    arrCurrentIDs = arrIDs[j]
    arrCurrentIDs = (arrCurrentIDs[arrCurrentIDs >= 0]).astype('int')
    objData = LT.LAMMPSData(strDir + str(intDir) + '/TJ' + str(i) + '.lst' ,1,4.05, LT.LAMMPSGlobal)
    objTJ = objData.GetTimeStepByIndex(-1)
    arrValues =  objTJ.GetAtomsByID(arrCurrentIDs)
    pts = arrValues[:,1:4]
    if len(pts) > 0:
        ax.scatter(*tuple(zip(*pts)))
        plt.show()
    else:
        print('Whoops ' + str(i) + ',' + str(j))
 

# %%
