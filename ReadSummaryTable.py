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

strDirectory = '/home/p17992pt/csf4_scratch/Axis101/TJSigma11/' #str(sys.argv[1])
intDirs = 10
intFiles = 10
lstStacked = []
arrValues = np.loadtxt(strDirectory +'Values.txt')
for j in arrValues:
    print(j[0],j[1],j[3]-j[2] + j[5]*(j[6]-j[7]),j[6]-j[7])

#arrRows = np.isin(arrValues[:,1],  [3,6])
#arrValues = arrValues[arrRows]
#print(arrValues)
arrValues, index = np.unique(arrValues,axis=0, return_index=True)
arrValues = arrValues[np.argsort(index)]
print(np.mean(arrValues[:,3]-arrValues[:,2] +arrValues[:,5]*(arrValues[:,6]-arrValues[:,7])),np.std(arrValues[:,3]-arrValues[:,2] +arrValues[:,5]*(arrValues[:,6]-arrValues[:,7])))

plt.show()
plt.scatter(arrValues[:,1],arrValues[:,3]-arrValues[:,2] + arrValues[:,5]*(arrValues[:,6]-arrValues[:,7]))
plt.show()