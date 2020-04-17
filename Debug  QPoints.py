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



a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
#strDumpFile = '../../PythonLAMMPS/VolumeTest/dump.eamPM'
strDumpFile = '/home/paul/csf3_scratch/TripleLines/data17/dump.eam17PM'
objData = LD.LAMMPSData(strDumpFile,1)
objProcess = objData.GetTimeStepByIndex(-1)
objProcess.CategoriseAtoms()         
#objQPoints = LD.Quantised2DPoints(objProcess.GetOtherAtoms()[:,1:3], a1, objProcess.GetCellVectors()[0:2,0:2],11) 
objQPoints = LD.QuantisedRectangularPoints(objProcess.GetOtherAtoms()[:,1:3],objProcess.GetUnitBasisConversions()[0:2,0:2],10,a2/2, 1)
fig,ax = plt.subplots(1,3)
ax[0].imshow(objQPoints.GetArrayGrid())
#objQPoints.CopyPointsToWrapper()
arrTriplePoints = objQPoints.FindTriplePoints()
ax[1].imshow(objQPoints.GetExtendedSkeletonPoints())
objQPoints.FindGrainBoundaries()
#print(objQPoints.GetNumberOfGrainBoundaries())
#objQPoints.ClassifyGBPoints(3,True)
#objQPoints.FindGrainBoundaries()
ax[2].imshow(objQPoints.GetExtendedSkeletonPoints())
#fig.colorbar(pos, ax=ax[2])
#print(objQPoints.ClearWrapperValues())
#print(objQPoints.GetGrainBoundaries())
objQPoints.MakeAdjacencyMatrix()
print(objQPoints.FindAdjacentTriplePoints(2))



#objQPoints.MakeGrainBoundaries(1,4)
#objQPoints.ClearWrapper()
#arrLabel = objQPoints.GetGrainBoundaryLabels()
#objQPoints.MergeGrainBoundaries()
#print(objQPoints.GetNumberOfGrainBoundaries())
#ax[3].imshow(objQPoints.GetExtendedSkeletonPoints())
#print(objQPoints.GetGrainBoundaryLabels())
#ax[4].imshow(arrLabel)
#print(np.unique(objQPoints.FindGrainBoundaries()))
#objQPoints.MergeGrainBoundaries()
#print(objQPoints.GetDislocations())
#ax[5].imshow(objQPoints.GetExtendedSkeletonPoints())
plt.show()
#print(np.min(sc.spatial.distance_matrix(objQPoints.GetGrainBoundaries()[0], objQPoints.GetGrainBoundaries()[-1])))
# lstMerged = objProcess.MergePeriodicTripleLines(4.05*np.sqrt(2))
# arrEnergies = objProcess.EstimateTripleLineEnergy(-3.36,4.05, True)

# print(arrEnergies[lstMerged[0]])
# #ax.scatter(*arrTripleLines[lstMerged[0]])
# #plt.show()
# print(lstMerged)
#for j in range(0,objQPoints.GetNumberOfGrainBoundaries()):
#    print(gf.SortInDistanceOrder(objQPoints.GetGrainBoundaries()[j]))
#print(gf.SortInDistanceOrder(objQPoints.GetGrainBoundaries()[1]))
#print(np.mean(objProcess.GetLatticeAtoms()[:,7]), np.mean(objProcess.GetOtherAtoms()[:,7]))
