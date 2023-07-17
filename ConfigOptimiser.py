import lammps
import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import os
from mpl_toolkits.mplot3d import Axes3D

objLammps = lammps.PyLammps()
#strDirectory = str(sys.argv[1])
#intSigma = int(sys.argv[2])
#intMax = int(sys.argv[3])
strDirectory = '/home/p17992pt/LAMMPSData/'
arrAxis = np.array([1,1,1])
intSigma = 3
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
gf.CubicCSLGenerator(arrAxis, 5)
arrRotation = np.zeros(3)
arrRotation[0:2] = objSigma.GetLatticeRotations()
arrRotation[2] = np.mean(arrRotation[0:2])
arrSigmaBasis = objSigma.GetBasisVectors()
a1 = 4.05 ##lattice parameter
a2 = a1*np.sqrt(3) #periodic cell repeat multiple
h= 5
s = np.linalg.norm(arrSigmaBasis, axis=1)[0]
z = a2*np.array([0,0,h])
#intStart = np.ceil(max(4, s)).astype('int')
intIncrement = np.ceil(1/s).astype('int')
fltAngle, arrVector = gf.FindRotationVectorAndAngle(arrAxis, np.array([0,0,1]))
arr111BasisVectors = gf.RotatedBasisVectors(fltAngle, arrVector)
l = int(15/s)
arrHorizontalVector = l*a1*arrSigmaBasis[0]
arrDiagonalVector =  l*a1*arrSigmaBasis[1]
MySimulationCell = gl.SimulationCell(np.array([3*arrHorizontalVector,3*arrDiagonalVector, z]))
objHex1 = gl.ExtrudedRegularPolygon(l*a1*s, (h-0.1)*a2, 6, gf.RotateVectors(arrRotation[0],z,arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]), arrHorizontalVector + arrDiagonalVector)
objHex2 = gl.ExtrudedRegularPolygon(l*a1*s, (h-0.1)*a2, 6, gf.RotateVectors(arrRotation[1],z,arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),np.zeros(3))
objHex3 = gl.ExtrudedRegularPolygon(l*a1*s, (h-0.1)*a2, 6, gf.RotateVectors(arrRotation[2],z, arr111BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),-arrDiagonalVector+2*arrHorizontalVector)
MySimulationCell.AddGrain(objHex1,'Hex1')
MySimulationCell.AddGrain(objHex2,'Hex2')
MySimulationCell.AddGrain(objHex3, 'Hex3')
fltNearestNeighbour = objHex1.GetNearestNeighbourDistance()
lstj = []
lstPE = []
lstAdjusted = []
lstAtoms = []
intMin = 1
fltMean = -3.36
MySimulationCell.LAMMPSMinimisePositions(strDirectory, 'Aread.dat','TemplateMin.in',10, -3.36)
#os.path(strDirectory)
for j in range(1, 11):
    lstj.append(fltNearestNeighbour*j/20)
    MySimulationCell.RemoveTooCloseAtoms(lstj[-1])
    lstAtoms.append(MySimulationCell.GetUpdatedAtomNumbers())
    MySimulationCell.WrapAllAtomsIntoSimulationCell()
    MySimulationCell.SetFileHeader('Sigma ' + str(intSigma) + ' about ' + str(arrAxis) + ' with Hexagonal grains 1,2 and 3 with angle array ' + str(arrRotation) + ' and length multiple of ' + str(j))
    MySimulationCell.WriteLAMMPSDataFile(strDirectory + 'read.dat')
    objLammps = lammps.PyLammps()
    objLammps.file(strDirectory + 'TemplateMin.in')
    lstPE.append(objLammps.eval('pe'))
    #objLammps.command('write_data read.data')
    if len(lstj) ==1:
        lstAdjusted.append(0)
    else: 
        lstAdjusted.append(lstPE[-1] - lstPE[-2] + fltMean*(lstAtoms[-2]-lstAtoms[-1]))
    if len(lstAdjusted)-1 == np.argmin(lstAdjusted):
        objLammps.command('write_data ' + strDirectory + 'Aread.dat')
        intMin = j
    objLammps.close()
# #os.rename(strDirectory + '1Min.dmp', strDirectory + 'AMin1.dmp')    
print(intMin)
plt.scatter(lstj,lstAdjusted)
plt.show()



