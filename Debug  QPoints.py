# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'


# ## Image Analysis Techniques to Detect Triple Lines
# This looks at grain boundary atoms as defined by Ovitos using PTM and attempts to find intersection points which are potential triple lines. The main routine converted the atom positions into an integer array position initially form a square grid of side length $a=4.05$. The issue then becomes looking at the neighbouring squares to determine whether this is a grain boundary or potential triple line. At present the issue is more complicated as the skeletonize command is not consistently reduing the image to a continous set of curves all of which are one pixel wide.

import lammps
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.ticker import FormatStrFormatter
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import scipy as sc
from skimage.morphology import skeletonize, thin, medial_axis
import os
from skimage.measure import label, regionprops
from IPython.core.debugger import set_trace


fig = plt.figure(figsize=plt.figaspect(1)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
#ax = fig.gca(projection='3d')
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
#strDumpFile = '../../PythonLAMMPS/VolumeTest/dump.eamPM'
#strRoot = '/home/p17992pt/csf3_scratch/CSL/Axis111/Temp700/Sigma7/' + '1Sim1500.dmp'
strDumpFile = '/home/p17992pt/csf3_scratch/Hex/Axis100/data1/1Sim1500.dmp'
objData = LT.LAMMPSData(strDumpFile,1, 4.05,LT.LAMMPSGlobal)
objProcess = objData.GetTimeStepByIndex(-1)
objProcess.CategoriseAtoms()         
objQPoints = LT.QuantisedCuboidPoints(objProcess.GetNonLatticeAtoms()[:,1:4],objProcess.GetUnitBasisConversions(), objProcess.GetCellVectors(), 4.05*np.ones(3),10)
#fig,ax = plt.subplots(1,2)
tr = transforms.Affine2D().rotate_deg(90)
objQPoints.FindJunctionLines()
#pts2 = objQPoints.GetJunctionLinePoints(1)
#ax.scatter(*tuple(zip(*pts2)))
#gf.EqualAxis3D(ax)
#plt.show()
#for k in objQPoints.GetGrainBoundaryIDs():
for k in range(3,4):
    pts = objQPoints.GetSurfaceMesh(k)[-10:]
    i = 1
    for l in pts:
        ax = fig.add_subplot(1,len(pts),i,projection='3d')
        ax.scatter(*tuple(zip(*l)))
        gf.EqualAxis3D(ax)
        i += 1
plt.show()


# objArray = objQPoints.GetGrainBoundaryArray()
# ax[0].matshow(np.flip(objArray[:,:,0]),origin='lower')
# ax[1].matshow(np.flip(objArray[:,:,1]), origin = 'lower')

# #objQPoints.FindTriplePoints()
# ax[0].matshow(np.flip(objQPoints.GetArrayGrid()),origin='lower')
# #print(objQPoints.GetTriplePoints())
# ax[1].matshow(np.flip(objQPoints.GetExtendedArrayGrid()),origin='lower')
# #objQPoints.FindGrainBoundaries()
# #print(objQPoints.GetNumberOfGrainBoundaries())
# #objQPoints.ClassifyGBPoints(3,True)
# objQPoints.FindTriplePoints()
# ax[2].matshow(np.flip(objQPoints.GetExtendedSkeletonPoints()),origin='lower')
# objQPoints.FindGrainBoundaries()
# #ax[3].imshow(objQPoints.GetExtendedSkeletonPoints(),cmap='gist_earth')
# #fig.colorbar(pos, ax=ax[2])
# #print(objQPoints.ClearWrapperValues())
# #print(objQPoints.GetGrainBoundaries())
# print(objQPoints.MakeAdjacencyMatrix())
# for j in range(len(objQPoints.GetTriplePoints())):
#     print(objQPoints.GetEquivalentTriplePoints(j))



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
#plt.show()
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
