import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp


strDirectory = str(sys.argv[1])
intSigma = int(sys.argv[2])

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.gca(projection='3d')
arrAxis = np.array([0,0,1])
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
gf.CubicCSLGenerator(arrAxis, 5)
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
intMax = 40
s = np.linalg.norm(arrSigmaBasis, axis=1)[0]
a = 4.05 ##lattice parameter
x = np.round(intMax/s,0)
w = x*a*s
l = x*a*s
h= x*a*np.round(s,0)
z = np.array([0,0,h])
arr100BasisVectors = gf.StandardBasisVectors(3)
arrHorizontalVector = np.array([w,0,0])
arrUpVector =  np.array([0, l,0]) 
arrLatticeParameters= np.array([a,a,a])
fltDatum = -3.36
#arrOrigin = a*np.array([arrSigmaBasis[0]*(0.5-np.random.ranf()), arrSigmaBasis[1]*(0.5-np.random.ranf()),arrSigmaBasis[2]*(0.5-np.random.ranf())]) 
MySimulationCell = gl.SimulationCell(np.array([arrHorizontalVector,arrUpVector, z])) 
objFullCell1 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(fltAngle1,z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(fltAngle2,z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint('x -' +str(w) + '/2')
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint('-x +' +str(w) + '/2')

fltDistance = objFullCell1.GetNearestNeighbourDistance()/3
##B

objBaseLeft = cp.deepcopy(objLeftCell1)
objBaseRight = cp.deepcopy(objRightCell2)
MySimulationCell.AddGrain(objBaseLeft)
MySimulationCell.AddGrain(objBaseRight)
MySimulationCell.RemoveTooCloseAtoms(fltDistance)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
MySimulationCell.WriteLAMMPSDataFile(strDirectory + 'read0.dat')
MySimulationCell.RemoveAllGrains()
fIn = open(strDirectory +  'TemplateMin.in', 'rt')
fData = fIn.read()
fData = fData.replace('read.dat', 'read0.dat')
fData = fData.replace('read.dmp', 'read0.dmp')
fData = fData.replace('logfile', 'logfile0')
fIn.close()
fIn = open(strDirectory + 'TemplateMin0.in', 'wt')
fIn.write(fData)
fIn.close()
arrCentre = np.array([w/2,l/2,h/2]) + a*(0.5-np.random.ranf())*arrSigmaBasis[1]


for j in range(1,np.round(intMax/4,0).astype('int')+1):
    r = a*s*j*x/intMax
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
    MySimulationCell.RemoveTooCloseAtoms(fltDistance)
    MySimulationCell.WrapAllAtomsIntoSimulationCell()
    MySimulationCell.SetFileHeader('Grain centre is ' +str(arrCentre))
    MySimulationCell.WriteLAMMPSDataFile(strDirectory + 'read' + str(j) + '.dat')
    MySimulationCell.RemoveAllGrains()
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', 'read' + str(j) + '.dat')
    fData = fData.replace('read.dmp', 'read' + str(j) + '.dmp')
    fData = fData.replace('logfile', 'logfile' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateMin' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()




