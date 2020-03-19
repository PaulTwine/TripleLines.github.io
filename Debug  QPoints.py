# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# ## Image Analysis Techniques to Detect Triple Lines
# This looks at grain boundary atoms as defined by Ovitos using PTM and attempts to find intersection points which are potential triple lines. The main routine converted the atom positions into an integer array position initially form a square grid of side length $a=4.05$. The issue then becomes looking at the neighbouring squares to determine whether this is a grain boundary or potential triple line. At present the issue is more complicated as the skeletonize command is not consistently reduing the image to a continous set of curves all of which are one pixel wide.


import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import scipy as sc
from skimage.morphology import skeletonize, thin, medial_axis
import os
from skimage.measure import label, regionprops
from IPython.core.debugger import set_trace



# #fig = plt.figure()
# #ax = fig.gca(projection='3d')
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
# l = 3
# h= 4
# z = a2*np.array([0,0,h])
# #ax.set_xlabel('x')
# #ax.set_ylabel('y')
# #ax.set_zlabel('z')
# strDataFile = 'new.data'
# #strDumpFile = 'dump.eam'
# strDumpFile = 'VolumeTest/dump.eam'
# strPMFile = strDumpFile + 'PM'
# arr111BasisVectors = gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2))
# arrHorizontalVector = np.array([l*a2,0,0])
# arrDiagonalVector =  np.array([a2*l/2, a2*l*np.sqrt(3)/2,0])


# MySimulationCell = gl.SimulationCell(np.array([3*arrHorizontalVector,3*arrDiagonalVector, z])) 
# objHex1 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, arr111BasisVectors, ld.FCCCell, np.array([a1,a1,a1]))
# objHex2 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, gf.RotateVectors(gf.DegreesToRadians(20),z, arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),-arrDiagonalVector+2*arrHorizontalVector)
# objHex3 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, gf.RotateVectors(gf.DegreesToRadians(40),z,arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]), arrHorizontalVector + arrDiagonalVector)
# MySimulationCell.AddGrain(objHex1)
# MySimulationCell.AddGrain(objHex2)
# MySimulationCell.AddGrain(objHex3)
# MySimulationCell.WrapAllPointsIntoSimulationCell()
# MySimulationCell.RemovePlaneOfAtoms(np.array([[0,0,1,a2*h]]),0.1)
# #MySimulationCell.WriteLAMMPSDataFile(strDataFile)


strPMFile = '/home/paul/csf3_scratch/TripleLines/data1/dump.eam1PM'
objData = LD.LAMMPSData(strPMFile,1)
objProcess = objData.GetTimeStepByIndex(0)
#objProcess.FindTriplePoints(4.05)
#objProcess.GetGrainBoundaries()
#objProcess.CategoriseAtoms()         
#objQPoints = LD.Quantised2DPoints(objProcess.GetOtherAtoms()[:,1:3], a1, objProcess.GetCellVectors()[0:2,0:2],11) 
objQPoints = LD.QuantisedRectangularPoints(objProcess.GetOtherAtoms()[:,1:3],objProcess.GetUnitBasisConversions()[0:2,0:2],5,a2, 5)
fig,ax = plt.subplots(1,6)
ax[0].imshow(objQPoints.GetArrayGrid())
#objQPoints.CopyPointsToWrapper()
ax[1].imshow(objQPoints.GetExtendedArrayGrid())
arrTriplePoints = objQPoints.FindTriplePoints()
#objQPoints.FindGrainBoundaries()
ax[2].imshow(objQPoints.GetExtendedSkeletonPoints())
#print(objQPoints.ClearWrapperValues())
arrLabel = objQPoints.GetGrainBoundaryLabels()
ax[3].imshow(objQPoints.GetExtendedSkeletonPoints())
print(objQPoints.GetGrainBoundaries())
ax[4].imshow(arrLabel)
#print(np.unique(objQPoints.FindGrainBoundaries()))
objQPoints.FindDislocations()
print(objQPoints.GetDislocations())
ax[5].imshow(objQPoints.GetExtendedSkeletonPoints())
plt.show()

# arrTripleLines  = objProcess.FindTriplePoints(4.05,True)
# lstGB = objProcess.GetGrainBoundaries()
# print(len(lstGB))
# print(lstGB[1])
# lstMerged = objProcess.MergePeriodicTripleLines(4.05*np.sqrt(2))
# arrEnergies = objProcess.EstimateTripleLineEnergy(-3.36,4.05, True)

# print(arrEnergies[lstMerged[0]])
# #ax.scatter(*arrTripleLines[lstMerged[0]])
# #plt.show()
# print(lstMerged)
#for j in range(0,objQPoints.GetNumberOfGrainBoundaries()):
#    print(gf.SortInDistanceOrder(objQPoints.GetGrainBoundaries()[j]))
print(gf.SortInDistanceOrder(objQPoints.GetGrainBoundaries()[1]))
#print(np.mean(objProcess.GetLatticeAtoms()[:,7]), np.mean(objProcess.GetOtherAtoms()[:,7]))
