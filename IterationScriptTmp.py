import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import sys

intCounter = int(1)
intIncrement = float(15)
fltSymmetry = float(90)
fltAngle1, fltAngle2 = gf.AngleGenerator(intCounter, intIncrement, fltSymmetry)
a1 = 4.05 ##lattice parameter
#a2 = a1*np.sqrt(3) #periodic cell repeat multiple
l = 25
h= 8
z = a1*np.array([0,0,h])
strDirectory = '../../data' + str(intCounter) 
strDataFile = 'read.data' +str(intCounter)
strDumpFile = 'dump.eam' +str(intCounter)
#arr111BasisVectors = gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2))
arr100BasisVectors = gf.StandardBasisVectors(3)
arrHorizontalVector = np.array([l*a1,0,0])
arrDiagonalVector =  np.array([a1*l/2, a1*l*np.sqrt(3)/2,0])
MySimulationCell = gl.SimulationCell(np.array([3*arrHorizontalVector,3*arrDiagonalVector, z])) 
objHex1 = gl.ExtrudedRegularPolygon(l*a1, h*a1, 6, arr100BasisVectors, ld.FCCCell, np.array([a1,a1,a1]))
objHex2 = gl.ExtrudedRegularPolygon(l*a1, h*a1, 6, gf.RotateVectors(gf.DegreesToRadians(fltAngle1),z, arr100BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),-arrDiagonalVector+2*arrHorizontalVector)
objHex3 = gl.ExtrudedRegularPolygon(l*a1, h*a1, 6, gf.RotateVectors(gf.DegreesToRadians(fltAngle2),z,arr100BasisVectors), ld.FCCCell, np.array([a1,a1,a1]), arrHorizontalVector + arrDiagonalVector)
MySimulationCell.AddGrain(objHex1)
MySimulationCell.AddGrain(objHex2)
MySimulationCell.AddGrain(objHex3)
MySimulationCell.WrapAllPointsIntoSimulationCell()
MySimulationCell.RemovePlaneOfAtoms(np.array([[0,0,1,z[2]]]),0.1)
MySimulationCell.WriteLAMMPSDataFile(strDirectory + '/' + strDataFile)

# fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'rt')
# fData = fIn.read()
# fData = fData.replace('new.data', strDataFile)
# fData = fData.replace('dump.eam', strDumpFile)
# fIn.close()
# fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'wt')
# fIn.write(fData)
# fIn.close()





