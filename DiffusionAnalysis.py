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

lstTJMean = []
lstGBMean = []

#for j in range(10):
#strRoot = '/home/p17992pt/csf3_scratch/Mobility/TwoCell/Axis101/TJSigma3/' + str(j) + '/' 
strRoot = '/home/p17992pt/csf3_scratch/Mobility/TwoCell/ECTest/'

arrTJ = np.loadtxt(strRoot + 'TJNewDisplacements.txt')
dispTJ = np.linalg.norm(arrTJ[:,:-1], axis=1)
distTJ = arrTJ[:,-1]
fltDist = 250
fltDisp = 2*4.05/np.sqrt(2)
rowTJDisp = np.where(dispTJ > fltDisp)[0]
rowTJDist = np.where(dispTJ > fltDisp)[0]

arrGB = np.loadtxt(strRoot + 'GBNewDisplacements.txt')
dispGB = np.linalg.norm(arrGB[:,:-1], axis=1)
distGB = arrGB[:,-1]
rowGBDisp = np.where(dispGB > fltDisp)[0]
rowGBDist = np.where(distGB > fltDist)[0]


# plt.hist(dispTJ[rowTJDisp],alpha=0.5, bins=20,label='TJ')
# plt.hist(dispGB[rowGBDisp],alpha=0.5, bins=20,label='GB')
# plt.legend(loc="upper right")
# plt.show()


    # plt.hist(distTJ[rowTJDist],alpha=0.5, bins=20,label='TJ')
    # plt.hist(distGB[rowGBDist],alpha=0.5, bins=20,label='GB')
    # plt.legend(loc="upper right")
    # plt.show()
lstTJMean.append(np.mean(dispTJ[rowTJDisp])) 
lstGBMean.append(np.mean(dispGB[rowGBDisp]))

plt.bar(list(range(0,20,2)),lstTJMean,-0.35, label='TJ Cell')
plt.bar(list(range(1,21,2)),lstGBMean,+0.35,label='GB Cell')
plt.legend(loc='lower right')
plt.show()
arrGB = np.loadtxt(strRoot + 'GBNewDisplacements.txt')
distTJ = arrTJ[:,-1] # np.linalg.norm(arrTJ, axis=1)
distGB = arrGB[:,-1] #np.linalg.norm(arrGB, axis=1)
print(gf.NormaliseVector(np.sum(arrTJ[distTJ > 4.05/np.sqrt(2)],axis=0)))
lstk = []
lstTJMean = []
lstGBMean = []
for k in range(1,12):
    fltTol = 250
    rowTJ = np.where(distTJ > fltTol)[0]
    rowGB = np.where(distGB > fltTol)[0]
    objData = LT.LAMMPSData(strRoot + 'TJNew0.dmp', 1, 4.05, LT.LAMMPSAnalysis3D)
    intAtoms = objData.GetAtomNumbers()[0]
    lstk.append(fltTol)
    lstTJMean.append(len(rowTJ)/intAtoms)
    lstGBMean.append(len(rowGB)/intAtoms)
    objTimeStep = objData.GetTimeStepByIndex(-1)
ptsTJ  = objTimeStep.GetAtomsByID(rowTJ+np.ones(len(rowTJ)))[:,1:4]
ptsGB =  objTimeStep.GetAtomsByID(rowGB+np.ones(len(rowGB)))[:,1:4]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*tuple(zip(*ptsTJ)))
#ax.scatter(*tuple(zip(*ptsGB)))
plt.show()
#print(np.std(distTJ[rowTJ]), np.std(distGB[rowGB]))
# plt.scatter(lstk[1:],lstTJMean[1:])
# plt.scatter(lstk[1:],lstGBMean[1:])
# plt.legend(['TJ','GB'])
# plt.show()