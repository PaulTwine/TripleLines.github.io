import LAMMPSDump as  Ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import GeometryFunctions as gf
from scipy import spatial

#the /ovitos PMStructure.py strFilename script must be run first as this runs the Ovitos Polyhedral matching 
#function which populates the data file with the quarterninon data

objData = Ld.LAMMPSData('data/30and40.eamPM')
objTimeStep = objData.GetTimeStepByIndex(0)

arrQuart1 = gf.GetQuaternionFromVector(np.array([0,0,1]),gf.DegreesToRadians(0))
arrQuart2 = gf.GetQuaternionFromVector(np.array([0,0,1]),gf.DegreesToRadians(30))
arrQuart3 = gf.GetQuaternionFromVector(np.array([0,0,1]),gf.DegreesToRadians(40))
etol = np.dot(gf.GetQuaternionFromVector(np.array([1,0,0]),gf.DegreesToRadians(10)),gf.GetQuaternionFromVector(np.array([1,0,0]),0))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
objPostProcess = Ld.OVITOSPostProcess(np.array([arrQuart1,arrQuart2,arrQuart3]), objTimeStep, 1)
objPostProcess.FindTriplePoints()
#ax.scatter(*objPostProcess.PlotGBOnlyAtoms(),c='red')
#ax.scatter(*objPostProcess.PlotTripleLine(),c='red')
#ax.scatter(*objPostProcess.PlotTripleLineAtoms(), c='green')
#ax.scatter(*objPostProcess.PlotGBAtoms(),c='blue')
#ax.scatter(*objPostProcess.PlotTriplePoints(),c='black')
ax.scatter(*objPostProcess.PlotUnknownAtoms(), c='blue')
#ax.scatter(*objPostProcess.PlotGrain('0'))
#ax.scatter(*objPostProcess.PlotGrain('1'))
#ax.scatter(*objPostProcess.PlotGrain('2'))
print(objPostProcess.GetMeanGrainBoundaryWidth())
print(len(objPostProcess.GetTriplePoints()))
objPostProcess.PartitionTripleLines()
objPostProcess.MergePeriodicTripleLines()
n = objPostProcess.GetNumberOfTripleLines()
print(n)
print(etol)
#for j in range(n):
#    ax.scatter(*objPostProcess.PlotNthTripleLine(j))
plt.show()
#intColumn = objTimeStep.GetColumnNames().index('c_pe1')
#print("Mean triple line PE in eV is",np.mean(objPostProcess.GetTripleLineAtoms()[:,intColumn]))
#print("Mean GB atom PE in eV is", np.mean(objPostProcess.GetGBOnlyAtoms()[:,intColumn]))
#print(objPostProcess.PartitionTripleLines())
# dbscan= DBSCAN(eps=objPostProcess.GetMeanGrainBoundaryWidth(),min_samples=3)
# model = dbscan.fit(objPostProcess.GetTripleLine())
# labels = model.labels_
# core_samples= np.zeros_like(labels,dtype = bool)
# core_samples[dbscan.core_sample_indices_] = True
# print(core_samples)