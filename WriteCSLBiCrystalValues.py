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

#print(gf.CubicCSLGenerator(np.array([1,0,1]),5))

strDirectory = '/home/p17992pt/csf4_scratch/BiCrystal/Axis101/Sigma19/' #str(sys.argv[1])
intFiles = 10
lstStacked = []
# for j in range(intFiles):
#     arrRow = np.zeros(5)
#     arrRow[0] = 0.1*j
#     objData = LT.LAMMPSData(strDirectory + 'GB' +str(j) + '.lst',1,4.05, LT.LAMMPSGlobal)
#     objGB = objData.GetTimeStepByIndex(-1)
#     intPECol = objGB.GetColumnIndex('c_pe1')
#     arrCellVectors = objGB.GetCellVectors()
#     arrRow[1] = np.sum(objGB.GetColumnByName('c_pe1'))
#     arrRow[2] = np.mean(objGB.GetLatticeAtoms()[:,intPECol])
#     arrRow[3] = objGB.GetNumberOfAtoms()
#     arrRow[4] = np.linalg.norm(np.cross(arrCellVectors[1],arrCellVectors[2]))
#     lstStacked.append(arrRow)
# arrValues = np.vstack(lstStacked)
# np.savetxt(strDirectory + 'Values.txt',arrValues,fmt='%f')
arrValues = np.loadtxt(strDirectory+'Values.txt')
#pts = (arrValues[:,1]+(np.ones(len(arrValues))*arrValues[0,3]-arrValues[:,3])*arrValues[:,2] -arrValues[0,3]*arrValues[:,2])/arrValues[0,4]
#pts= (arrValues[:,1]+(np.ones(len(arrValues))*arrValues[0,3]-arrValues[:,3])*(-3.36) -arrValues[0,3]*np.ones(len(arrValues))*(-3.36))/arrValues[0,4]
# 
pts = (arrValues[:,1] -(-3.36*arrValues[:,3]))/(2*arrValues[:,4])         
print(pts)
arrUnique,index = np.unique(pts,return_index=True)
arrUnique = arrUnique[np.argsort(index)]
lstStable = []
i = 0

lstRemove = []

if len(arrUnique) > 1:
    while i < len(arrUnique):
        blnRemove = False
        f = 1.00
        if i == 0:
            if arrUnique[i] > f*arrUnique[1]:
                blnRemove= True
        elif i == len(arrUnique) -1: 
            if arrUnique[i] > f*arrUnique[i-1]:
                blnRemove= True
        elif (arrUnique[i] > f*arrUnique[i-1]) or (arrUnique[i] > f*arrUnique[i+1]):
            blnRemove= True
        if blnRemove:
            lstRemove.extend(np.where(pts == arrUnique[i])[0].tolist())
        i +=1

print(np.array(list(set(range(len(arrValues))).difference(lstRemove))))

    
plt.scatter(arrValues[:,0],pts)
plt.show()