import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp

# strDirectory = str(sys.argv[1])
# intSigma = int(sys.argv[2])
# arrAxis = np.array([0,0,1])
strDirectory = '/home/p17992pt/LAMMPSData/'
intSigma = 17
arrAxis = np.array([0,0,1])
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
print(gf.CubicCSLGenerator(arrAxis))
print(arrSigmaBasis)
intMax = 60
intHeight = 5
s1 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
s2 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
s3 = np.linalg.norm(arrSigmaBasis, axis=1)[2]
a = 4.05 ##lattice parameter
x = np.round(intMax/s1,0)
if np.mod(x,2) !=0: #ensure an even number of CSL unit cells in the x direction
    x += 1
y = np.round(intMax/s2,0)
if np.mod(y,2) !=0: 
    y += 1
w = x*a
l = y*a
h = a*np.round(intHeight/s3,0)
arrX = w*arrSigmaBasis[0]
arrXY = l*arrSigmaBasis[1]
z = h*arrSigmaBasis[2]
if np.all(arrAxis == np.array([0,0,1])):
    arrBasisVectors = gf.StandardBasisVectors(3)
else:
    fltAngle3, arrRotation = gf.FindRotationVectorAndAngle(arrAxis,np.array([0,0,1]))
    arrBasisVectors = np.round(gf.RotateVectors(fltAngle3, arrRotation,gf.StandardBasisVectors(3)),10)  
arrLatticeParameters= np.array([a,a,a])
fltDatum = -3.36
arrShift = a*(0.5-np.random.ranf())*arrSigmaBasis[1]
arrCentre = 0.5*(arrX+arrXY) + arrShift
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCentre[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCentre[0]) + ')' 
MySimulationCell = gl.SimulationCell(np.array([arrX,arrXY, z])) 
objFullCell1 = gl.ExtrudedParallelogram(arrX,arrXY,s3*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,arrShift)
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)

fltDistance = objFullCell1.GetNearestNeighbourDistance()/5
#fltDistance = 30
objBaseLeft = cp.deepcopy(objLeftCell1)
objBaseRight = cp.deepcopy(objRightCell2)
MySimulationCell.AddGrain(objBaseLeft)
MySimulationCell.AddGrain(objBaseRight)
MySimulationCell.RemoveAtomsOnOpenBoundaries()
MySimulationCell.RemoveTooCloseAtoms(fltDistance)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
MySimulationCell.WriteLAMMPSDataFile(strDirectory + 'read0.dat')
MySimulationCell.RemoveAllGrains()
fIn = open(strDirectory +  'TemplateMin.in', 'rt')
fData = fIn.read()
fData = fData.replace('read.dat', 'read0.dat')
fData = fData.replace('read.dmp', 'read0.dmp')
fData = fData.replace('logfile', 'logfile0')
fData = fData.replace('read.lst', 'read0.lst')   
fIn.close()
fIn = open(strDirectory + 'TemplateMin0.in', 'wt')
fIn.write(fData)
fIn.close()


for j in range(1,np.round(intMax/4,0).astype('int')+1):
    MySimulationCell = gl.SimulationCell(np.array([arrX,arrXY, z])) 
    r = a*j
    strCylinder = gf.ParseConic([arrCentre[0],arrCentre[1]],[r,r],[2,2])
    objCylinder3 = cp.deepcopy(objFullCell3)
    objCylinder3.ApplyGeneralConstraint(strCylinder)
    objLeftChopped1 = cp.deepcopy(objLeftCell1)
    objLeftChopped1.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
    objRightChopped2 = cp.deepcopy(objRightCell2)
    objRightChopped2.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
    MySimulationCell.AddGrain(objCylinder3)
    MySimulationCell.AddGrain(objLeftChopped1)
    MySimulationCell.AddGrain(objRightChopped2)
    MySimulationCell.RemoveAtomsOnOpenBoundaries()
    MySimulationCell.RemoveTooCloseAtoms(fltDistance)
    MySimulationCell.WrapAllAtomsIntoSimulationCell()
    MySimulationCell.SetFileHeader('Grain centre is ' +str(arrCentre))
    MySimulationCell.WriteLAMMPSDataFile(strDirectory + 'read' + str(j) + '.dat')
    MySimulationCell.RemoveAllGrains()
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', 'read' + str(j) + '.dat')
    fData = fData.replace('read.dmp', 'read' + str(j) + '.dmp')
    fData = fData.replace('read.lst', 'read' + str(j) + '.lst')
    fData = fData.replace('logfile', 'logfile' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateMin' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()
