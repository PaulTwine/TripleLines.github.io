import numpy as np
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')#
strDumpFile = '../../PythonLAMMPS/VolumeTest/dump.eam'
#strDumpFile = '/home/paul/csf3_scratch/TripleLines/data17/dump.eam17'
intTripleLine = 3
a1=4.05
a2= np.sqrt(3)*a1
strPMFile = strDumpFile + 'PM'
objData = LD.LAMMPSData(strPMFile,1)
objProcess = objData.GetTimeStepByIndex(-1)
objProcess.CategoriseAtoms()
h = objProcess.CellHeight
objProcess.FindTripleLines(a2,3*a2, 2)
print(objProcess.MergePeriodicTripleLines(a2))
print(objProcess.GetUniqueTripleLines())
print(objProcess.GetAdjacentTripleLines(1))
#print(objProcess.MergePerodicGrainBoundaries(a2/2))
for j in range(objProcess.GetNumberOfTripleLines()):
    print(objProcess.FindTripleLineEnergy(j,a1/4,a1))
# lstAtomsID = objProcess.FindCylindricalAtoms(objProcess.GetNonLatticeAtoms()[:,0:4],objProcess.GetTripleLines(intTripleLine),fltRadius,h)
# arrCPoints =objProcess.GetAtomsByID(lstAtomsID)[:,0:4]
# scDistanceMatrix = sc.spatial.distance_matrix(arrCPoints[:,1:4], arrCPoints[:,1:4])
# lstTJID = np.unique(np.argwhere((scDistanceMatrix < 1.01*a2) &(scDistanceMatrix > 0.99*a2)))
# arrTJPoints = objProcess.GetAtomsByID(arrCPoints[lstTJID,0])
# arrPlotPoints = objProcess.PeriodicShiftAllCloser(objProcess.GetTripleLines(intTripleLine),arrTJPoints[:,1:4])
# ax.scatter(arrPlotPoints[:,0],arrPlotPoints[:,1],arrPlotPoints[:,2])
# ax.scatter(objProcess.GetTripleLines(intTripleLine)[0],objProcess.GetTripleLines(intTripleLine)[1],objProcess.GetTripleLines(intTripleLine)[2])
#plt.axis('equal')
#plt.show()
#objProcess.OptimiseTripleLinePosition(intTripleLine, a2/8,a2, 5*a2)
lstR,lstV,lstI = objProcess.FindThreeGrainStrips(intTripleLine,a2,a2/4, 'mean')
plt.scatter(lstR,lstV)
plt.title('Mean PE per atom in eV')
plt.xlabel('Distance from triple line in $\AA$')
plt.show()
plt.axis('equal')
for j in range(len(objProcess.GetGrainBoundaries())):
    plt.scatter(objProcess.GetGrainBoundaries(j).GetPoints()[:,0], objProcess.GetGrainBoundaries(j).GetPoints()[:,1])
plt.legend(list(range(len(objProcess.GetGrainBoundaries()))))
plt.scatter(objProcess.GetAtomsByID(lstI)[:,1],objProcess.GetAtomsByID(lstI)[:,2], c='black')
plt.title('Diagram of grain boundaries and selected atom region shown in black')
plt.show()
# plt.axis('equal')
arrPoints = objProcess.FindValuesInCylinder(objProcess.GetLatticeAtoms()[:,0:4], 
                                             objProcess.GetTripleLines(intTripleLine), 5*a1,h,[1,2,3])
arrMovedPoints = objProcess.PeriodicShiftAllCloser( objProcess.GetTripleLines(intTripleLine), arrPoints)
plt.scatter(arrMovedPoints[:,0],arrMovedPoints[:,1])
plt.scatter(objProcess.GetTripleLines(intTripleLine)[0], objProcess.GetTripleLines(intTripleLine)[1])
plt.title('Region of atoms surrounding a triple line')
plt.axis('equal')
plt.show()
