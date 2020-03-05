import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import sys

intCounter = int(sys.argv[1])
intIncrements = int(sys.argv[2])
fltSymmetry = float(sys.argv[3])
fltAngle1, fltAngle2 = np.divmod(intCounter, intIncrements)
fltAngle1 = fltAngle1*fltSymmetry/intIncrements 
fltAngle2 = fltAngle2*fltSymmetry/intIncrements 
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
l = 3
h= 2
z = a2*np.array([0,0,h])
strDirectory = 'data' + str(intCounter) 
strDataFile = 'read.data' +str(intCounter)
strDumpFile = 'dump.eam' +str(intCounter)
arr111BasisVectors = gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2))
arrHorizontalVector = np.array([l*a2,0,0])
arrDiagonalVector =  np.array([a2*l/2, a2*l*np.sqrt(3)/2,0])
MySimulationCell = gl.SimulationCell(np.array([3*arrHorizontalVector,3*arrDiagonalVector, z])) 
objHex1 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, arr111BasisVectors, ld.FCCCell, np.array([a1,a1,a1]))
objHex2 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, gf.RotateVectors(gf.DegreesToRadians(fltAngle1),z, arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),-arrDiagonalVector+2*arrHorizontalVector)
objHex3 = gl.ExtrudedRegularPolygon(l*a2, h*a2, 6, gf.RotateVectors(gf.DegreesToRadians(fltAngle2),z,arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]), arrHorizontalVector + arrDiagonalVector)
MySimulationCell.AddGrain(objHex1)
MySimulationCell.AddGrain(objHex2)
MySimulationCell.AddGrain(objHex3)
MySimulationCell.WrapAllPointsIntoSimulationCell()
MySimulationCell.RemovePlaneOfAtoms(np.array([[0,0,1,z[2]]]),0.1)
MySimulationCell.WriteLAMMPSDataFile(strDirectory + '/' + strDataFile)

fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'rt')
fData = fIn.read()
fData = fData.replace('new.data', strDataFile)
fData = fData.replace('dump.eam', strDumpFile)
fIn.close()
fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'wt')
fIn.write(fData)
fIn.close()





