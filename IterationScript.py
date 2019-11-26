import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
from scipy import stats
import os

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
l = 3
h= 4
z = a2*np.array([0,0,h])
blnFullRun = False
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
strDataFile = 'new.data'
strDumpFile = 'dump.eam'
strPMFile = strDumpFile + 'PM'
arr111BasisVectors = gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2))
arrHorizontalVector = np.array([l*a2,0,0])
arrDiagonalVector =  np.array([a2*l/2, a2*l*np.sqrt(3)/2,0])
MySimulationCell = gl.SimulationCell(np.array([3*arrHorizontalVector,3*arrDiagonalVector, z])) 
objHex1 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, arr111BasisVectors, ld.FCCCell, np.array([a1,a1,a1]))
objHex2 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, gf.RotateVectors(gf.DegreesToRadians(17),z, arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),-arrDiagonalVector+2*arrHorizontalVector)
objHex3 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, gf.RotateVectors(gf.DegreesToRadians(38),z,arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]), arrHorizontalVector + arrDiagonalVector)
MySimulationCell.AddGrain(objHex1)
MySimulationCell.AddGrain(objHex2)
MySimulationCell.AddGrain(objHex3)
#MySimulationCell.WrapAllPointsIntoSimulationCell()
#MySimulationCell.RemovePlaneOfAtoms(np.array([[0,0,1,a2*h]]),0.1)
if blnFullRun:
    MySimulationCell.WriteLAMMPSDataFile(strDataFile)
    os.system('lmp_serial -in TemplateNVT.in')
#os.system('lmp_linux -in TemplateNVT.in')
    os.system('ovitos PMStructure.py ' + strDumpFile)
#ax.scatter(*MySimulationCell.PlotSimulationCellAtoms())
#objData = LD.LAMMPSData(strPMFile)
objData = LD.LAMMPSData('data/80and30.eamPM')
objTimeStep = objData.GetTimeStepByIndex(0)
#objTimeStep.StandardiseOrientationData()
objPostProcess = LD.OVITOSPostProcess(np.array([objHex1.GetQuaternionOrientation(), objHex2.GetQuaternionOrientation(), objHex3.GetQuaternionOrientation()]), objTimeStep, 1)
objPostProcess.ClassifyNonGrainAtoms()
#print(gf.FCCQuaternionEquivalence(objHex1.GetQuaternionOrientation()))
#print(gf.GetQuaternionFromVector(np.array([1,-1,0])/np.sqrt(2),np.arccos(1/np.sqrt(3))))
#print(np.transpose(objHex1.GetUnitBasisVectors()))
#plt.hist(objTimeStep.GetColumnByName('OrientationZ'))
#print(gf.GetQuaternionFromVector(np.array([1,-1,0])/np.sqrt(2),np.arccos(1/np.sqrt(3))))
#ax.scatter(*objPostProcess.PlotGBAtoms())
#ax.scatter(*objPostProcess.PlotGBAtoms())
#ax.scatter(*objPostProcess.PlotGrain('2'), marker = 'o')
# print(objHex2.GetUnitBasisVectors())
#ax.scatter(*objPostProcess.PlotGBAtoms())
ax.scatter(*objPostProcess.PlotTripleLineAtoms())
#ax.scatter(*objPostProcess.PlotDislocations())
#ax.scatter(*objPostProcess.PlotGBAtoms())
#ax.scatter(*objPostProcess.PlotUnknownAtoms())
#print(np.argwhere(objPostProcess.GetGBAtoms()[:,2] > 10))
#plt.hist(objPostProcess.GetUnknownAtoms()[:,9])
plt.show()