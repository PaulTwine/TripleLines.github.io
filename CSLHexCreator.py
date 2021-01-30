import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

arrAxis = np.array([1,1,1])
intSigma = 7
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
print(gf.CubicCSLGenerator(arrAxis, 5))
arrRotation = np.zeros(3)
arrRotation[1:3] = objSigma.GetLatticeRotations()
#arrRotation[0] = -arrRotation[1]
arrSigmaBasis = objSigma.GetBasisVectors()
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
h= 3
s = np.linalg.norm(arrSigmaBasis, axis=1)[0]
z = a2*np.array([0,0,h])
l = np.round(10/s,0).astype('int')
fltAngle, arrVector = gf.FindRotationVectorAndAngle(arrAxis, np.array([0,0,1]))
arr111BasisVectors = gf.RotatedBasisVectors(fltAngle, arrVector)
arrHorizontalVector = l*a1*arrSigmaBasis[0]
arrDiagonalVector =  l*a1*arrSigmaBasis[1]
lstRotations = [[0,1,1],[0,2,2], [1,2,2], [0,1,2],[1,1,1]]
for j in lstRotations:
    MySimulationCell = gl.SimulationCell(np.array([3*arrHorizontalVector,3*arrDiagonalVector, z]))
    objHex1 = gl.ExtrudedRegularPolygon(l*a1*s, (h-0.1)*a2, 6, gf.RotateVectors(arrRotation[j[0]],z,arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]), arrHorizontalVector + arrDiagonalVector)
    objHex2 = gl.ExtrudedRegularPolygon(l*a1*s, (h-0.1)*a2, 6, gf.RotateVectors(arrRotation[j[1]],z,arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),np.zeros(3))
    objHex3 = gl.ExtrudedRegularPolygon(l*a1*s, (h-0.1)*a2, 6, gf.RotateVectors(arrRotation[j[2]],z, arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),-arrDiagonalVector+2*arrHorizontalVector)
    MySimulationCell.AddGrain(objHex1,'Hex1')
    MySimulationCell.AddGrain(objHex2,'Hex2')
    MySimulationCell.AddGrain(objHex3, 'Hex3')
    MySimulationCell.RemoveTooCloseAtoms(a1)
    MySimulationCell.WrapAllPointsIntoSimulationCell()
    MySimulationCell.SetFileHeader('Sigma ' + str(intSigma) + ' about ' + str(arrAxis) + ' with Hexagonal grains 1,2 and 3 with angle array ' + str(arrRotation))
    strFileName = 'Hex' + str(j[0]+1) + 'and' + str(j[1]+1) + 'and' + str(j[2]+1) + 'Sigma' + str(intSigma) + '.dat'
    MySimulationCell.WriteLAMMPSDataFile('../PythonLAMMPS/' + strFileName)




1