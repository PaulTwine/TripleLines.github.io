import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
from scipy import spatial
from scipy import stats
import os

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
l = 4
h= 4*np.sqrt(3)
w= 4
strDataFile = 'new.data'
strDumpFile = 'dump.eam'
strPMFile = strDumpFile + 'PM'
arrSigma = gf.CubicCSLGenerator(np.array([1,1,1]), 25)
print(arrSigma)
fltAngle, arrVector = gf.FindRotationVectorAndAngle(np.array([1,1,1]), np.array([0,0,1]))
arrBasisVectors = gf.RotatedBasisVectors(fltAngle,arrVector)
objFirstLattice = gl.ExtrudedRectangle(l,w,h,arrBasisVectors, ld.FCCCell, np.ones(3),np.zeros(3))
objSecondLattice = gl.ExtrudedRectangle(l,w,h,gf.RotateVectors(arrSigma[0,1],np.array([0,0,1]),arrBasisVectors),ld.FCCCell,np.ones(3),np.zeros(3))
arrPoints1 = objFirstLattice.GetRealPoints()
arrPoints2 = objSecondLattice.GetRealPoints()
arrDistanceMatrix = spatial.distance_matrix(arrPoints1, arrPoints2)
lstPoints = np.where(arrDistanceMatrix < 1e-5)[0]
arrCSLPoints = arrPoints1[lstPoints]
plt.plot(*tuple(zip(*arrPoints1)), 'bo', c='b')
plt.plot(*tuple(zip(*arrPoints2)), 'bo', c='r')
plt.plot(*tuple(zip(*arrCSLPoints)), 'bo', c='black')
plt.show()


