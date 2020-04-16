import numpy as np
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import scipy as sc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline



#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')#
#strDumpFile = '../../PythonLAMMPS/VolumeTest/dump.eam'
strDumpFile = '/home/paul/csf3_scratch/TripleLines/data17/dump.eam17'
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
print(objProcess.GetUniqueTripleLines('UTJ3').GetCentre())
print(objProcess.GetTripleLines('TJ0').GetCentre())
#print(objProcess.GetAdjacentTripleLines(1))
# arrStart = objProcess.GetUniqueTripleLines(0)
# arrSecondTriple = objProcess.PeriodicShiftCloser(arrStart,objProcess.GetUniqueTripleLines(1))
# arrLength = arrSecondTriple-arrStart
# arrLength[2] = 12.15
# arrWidth = 10*a1*np.cross(gf.NormaliseVector(arrLength), np.array([0,0,1]))
# arrPoints = objProcess.FindValuesInBox(objProcess.GetNonLatticeAtoms()[:,0:4], arrStart,arrLength,arrWidth,
#                         objProcess.CellHeight*np.array([0,0,1]), [1,2,3]) 
# arrPoints = objProcess.PeriodicShiftAllCloser(arrStart,arrPoints)
# arrPointsM = arrPoints - arrStart
# arrLinearVector = arrSecondTriple[0:2]- arrStart[0:2]
# arrUnitVector = gf.NormaliseVector(arrLinearVector)
# arrProjection = np.zeros([len(arrPointsM),2])

# def ProjectPoint(in2DArray: np.array)->np.array:
#     return np.array([np.dot(in2DArray,arrUnitVector), np.cross(arrUnitVector,in2DArray)])

# for pos,j in enumerate(arrPointsM[:,0:2]):
#     #arrCross = np.cross(arrUnitVector,j)
#     arrProjection[pos] = ProjectPoint(j)
#     #arrProjection[pos,1] = arrCross
#     #arrProjection[pos,0] = np.dot(j-arrCross,arrUnitVector)
#     #arrProjection[pos,0] = np.dot(j, arrUnitVector)
# #print(arrProjection)
# arrProjectedEnd = ProjectPoint(arrLinearVector)
# arrProjection = arrProjection[np.where((arrProjection[:,0] >0) 
#                               & (arrProjection[:,0] < np.linalg.norm(arrLinearVector)))]
# #plt.axis('square')
# arrProjection = arrProjection[arrProjection[:,0].argsort()]

# fltLength = np.linalg.norm(arrLinearVector,axis=0)
# arrProjection = np.append(np.array([[0,0]]), arrProjection, axis=0)
# arrProjection = np.append(arrProjection,np.array([arrProjectedEnd]), axis=0)

# arrProjection = arrProjection/fltLength

# plt.scatter(arrProjection[:,0],arrProjection[:,1])
# #plt.axis('square')

#def GrainCurve(t,a0,a1, a2,a3):
#    return a0+ a1*t + a2*t**2 + a3*t**3 
#def Length(t,a2,a3):
#    return GrainCurve(t,*popt)/(np.sqrt((fltLength-a2-a3+2*a2*t+3*a3**2)**2+t**2))
#popt, popv = sc.optimize.curve_fit(GrainCurve, arrProjection[0], arrProjection[1])
t =np.linspace(0,1,20)
arrWeights = np.ones(len(arrProjection))
arrWeights[0] = 100
arrWeights[-1] = 100
cs = sc.interpolate.UnivariateSpline(arrProjection[:,0] , arrProjection[:,1],arrWeights,s=0.5)
plt.plot(t, cs(t), label='Cubic Spline', c='r')
#plt.axis('square')
plt.show()
#t =np.linspace(0,1,25)
arrPointsOut = np.zeros([len(t),2])
counter = 0
for tval in t:
    arrPointsOut[counter]= arrStart[0:2]+ tval*arrLinearVector + fltLength*cs(tval)*np.cross(arrUnitVector, np.array([0,1])) 
    counter +=1 
plt.scatter(arrPointsOut[:,0], arrPointsOut[:,1])
#plt.plot(t,GrainCurve(t,n))
plt.show()


objGrainBoundary = gl.GrainBoundaryCurve(arrStart,arrSecondTriple, [1,2], arrPoints)
arrPlotPoints = objGrainBoundary.GetPoints(2*a2)
plt.scatter(arrPlotPoints[:,0],arrPlotPoints[:,1])
plt.show()
#print(objProcess.MergePerodicGrainBoundaries(a2/2))
# for j in range(objProcess.GetNumberOfTripleLines()):
#     print(objProcess.FindTripleLineEnergy(j,a1/4,a1))
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
# #objProcess.OptimiseTripleLinePosition(intTripleLine, a2/8,a2, 5*a2)
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
