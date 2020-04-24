import numpy as np
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline
from skimage.transform import rescale, resize, downscale_local_mean



#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')#
#strDumpFile = '../../PythonLAMMPS/VolumeTest/dump.eam'
strDumpFile = '/home/paul/csf3_scratch/TripleLines/data3/dump.eam3'
intTripleLine = 3
a1=4.05
a2= np.sqrt(3)*a1
strPMFile = strDumpFile + 'PM'
objData = LD.LAMMPSData(strPMFile,1)
objProcess = objData.GetTimeStepByIndex(-1)
objProcess.CategoriseAtoms()
h = objProcess.CellHeight
objProcess.FindTripleLines(2*a2,3*a2, 3)
objProcess.MergePeriodicTripleLines(2*a2)
#print(objProcess.GetTripleLines('TJ6').GetEquivalentTripleLines())
objProcess.MakeGrainBoundaries()
print(objProcess.GetTripleLineIDs())
for strVal in objProcess.GetTripleLineIDs():
    print (objProcess.GetTripleLines(strVal).GetEquivalentTripleLines())
#for strVal in objProcess.GetUniqueTripleLineIDs():
#    print (objProcess.GetUniqueTripleLines(strVal).GetEquivalentTripleLines())
#print(objProcess.FindThreeGrainStrips('UTJ0',3*a2,a2/4))
#print(objProcess.GetGrainBoundaryIDs())
n=2
CellArray = plt.imread(strDumpFile+'PM.png')
#CellArray = plt.imread('../../PythonLAMMPS/VolumeTest/CellView.png')
CellArray0 = np.copy(CellArray[:,:,0])
arrPoints = np.where(CellArray0 !=1)
xmin = min(arrPoints[0])
xmax = max(arrPoints[0])
ymin = min(arrPoints[1])
ymax = max(arrPoints[1])
CellArray = CellArray[xmin:xmax,ymin:ymax,:]
fltSize = np.shape(CellArray)[1]
CellArray = rescale(CellArray, objProcess.GetBoundingBox()[0]/fltSize)
CellArray = np.flip(CellArray[:,:,:], axis=0)
#plt.matshow(CellArray[:,:,:], origin = 'lower')
for j in objProcess.GetUniqueTripleLineIDs():
    print(objProcess.FindTripleLineEnergy(j,a1/8,a2), j)
    lstGBIDs = []
    for h in objProcess.GetUniqueTripleLines(j).GetUniqueAdjacentGrainBoundaries():
        lstGBIDs.extend(objProcess.FindGBAtoms(h,2*objProcess.GetUniqueTripleLines(j).GetRadius(),3*a2))
    print(np.mean(objProcess.GetAtomsByID(lstGBIDs)[:,7]))

for i in objProcess.GetUniqueTripleLineIDs():
    print(objProcess.GetUniqueTripleLines(i).GetUniqueAdjacentGrainBoundaries())
for k in objProcess.GetUniqueGrainBoundaryIDs():
    j = objProcess.MoveToSimulationCell(objProcess.GetUniqueGrainBoundaries(k).GetPoints(a1))
    plt.scatter(j[:,0],j[:,1])
 
for k in objProcess.GetUniqueGrainBoundaryIDs():
     for j in objProcess.MoveToSimulationCell(objProcess.GetUniqueGrainBoundaries(k).GetPoints(2)):
             CellArray[np.round(j[1]).astype('int')-n:np.round(j[1]).astype('int')+n, 
                     np.round(j[0]).astype('int')-n:np.round(j[0]).astype('int')+n,0] =1

for k in objProcess.GetUniqueTripleLineIDs():
    j = objProcess.MoveToSimulationCell(objProcess.GetUniqueTripleLines(k).GetCentre())
    CellArray[np.round(j[1]).astype('int')-n:np.round(j[1]).astype('int')+n,  np.round(j[0]).astype('int')-n:np.round(j[0]).astype('int')+n,0] =-1

plt.legend(objProcess.GetUniqueGrainBoundaryIDs())
plt.matshow(CellArray[:,:,:], origin = 'lower')
#plt.axis('square')
plt.show()




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
# lstR,lstV,lstI = objProcess.FindThreeGrainStrips(intTripleLine,a2,a2/4, 'mean')
# plt.scatter(lstR,lstV)
# plt.title('Mean PE per atom in eV')
# plt.xlabel('Distance from triple line in $\AA$')
# plt.show()
# plt.axis('equal')
# for j in range(len(objProcess.GetGrainBoundaries())):
#     plt.scatter(objProcess.GetGrainBoundaries(j).GetPoints()[:,0], objProcess.GetGrainBoundaries(j).GetPoints()[:,1])
# plt.legend(list(range(len(objProcess.GetGrainBoundaries()))))
# plt.scatter(objProcess.GetAtomsByID(lstI)[:,1],objProcess.GetAtomsByID(lstI)[:,2], c='black')
# plt.title('Diagram of grain boundaries and selected atom region shown in black')
# plt.show()
# # plt.axis('equal')
# arrPoints = objProcess.FindValuesInCylinder(objProcess.GetLatticeAtoms()[:,0:4], 
#                                              objProcess.GetTripleLines(intTripleLine), 5*a1,h,[1,2,3])
# arrMovedPoints = objProcess.PeriodicShiftAllCloser( objProcess.GetTripleLines(intTripleLine), arrPoints)
# plt.scatter(arrMovedPoints[:,0],arrMovedPoints[:,1])
# plt.scatter(objProcess.GetTripleLines(intTripleLine)[0], objProcess.GetTripleLines(intTripleLine)[1])
# plt.title('Region of atoms surrounding a triple line')
# plt.axis('equal')
# plt.show()
