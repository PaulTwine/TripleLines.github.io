import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
from scipy import stats
import os

blnFullRun = False
blnPlot = True
if blnPlot:
    fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    #ax= plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
l = 3
h= 4
z = a2*np.array([0,0,h])
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
objData = LD.LAMMPSData(strPMFile)
#objData = LD.LAMMPSData('data/60and30.eamPM')
objTimeStep = objData.GetTimeStepByIndex(0)
objPostProcess = LD.OVITOSPostProcess(np.array([objHex1.GetQuaternionOrientation(), objHex2.GetQuaternionOrientation(), objHex3.GetQuaternionOrientation()]), objTimeStep, 1)
objPostProcess.ClassifyNonGrainAtoms()
objPostProcess.PartitionTripleLines()
#objPostProcess.MergePeriodicTripleLines()
intN = objPostProcess.GetNumberOfTripleLines()
print(intN)
if blnPlot:
#    ax.scatter(*objPostProcess.PlotNthTripleLine(0))
#    ax.scatter(*objPostProcess.PlotDislocations())
#    ax.scatter(*objPostProcess.PlotGBAtoms())
#    ax.scatter(objTimeStep.GetAtomData()[:,1], objTimeStep.GetAtomData()[:,2], objTimeStep.GetAtomData()[:,7])   
    for j in range(intN):
         ax.scatter(*objPostProcess.PlotNthTripleLine(j))
         print(np.sum(objPostProcess.GetMergedTripleLine(j)[:,7]))
    #ax.scatter(*objPostProcess.PlotGBAtoms())
    #ax.scatter(*objPostProcess.PlotTripleLineAtoms())
    #ax.scatter(*objPostProcess.PlotUnknownAtoms())
    plt.show()




