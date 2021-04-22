import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp


#strDirectory = sys.argv[1]
#intSigma = sys.argv[2]

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.gca(projection='3d')
arrAxis = np.array([0,0,1])
intSigma = 13
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
print(gf.CubicCSLGenerator(arrAxis, 5))
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
s = np.linalg.norm(arrSigmaBasis, axis=1)[0]
a = 4.05 ##lattice parameter
x = np.round(40/s,0)
w = x*a*s
l = x*a*s
h= x*a*np.round(s,0)
#w = 15*a*s
#l = 15*a*s
#h= 15*a*np.round(s)
z = np.array([0,0,h])
arr100BasisVectors = gf.StandardBasisVectors(3)
arrHorizontalVector = np.array([w,0,0])
arrUpVector =  np.array([0, l,0]) 
arrLatticeParameters= np.array([a,a,a])
fltDatum = -3.36
dctValues = dict()

MySimulationCell = gl.SimulationCell(np.array([arrHorizontalVector,arrUpVector, z])) 
objFullCell1 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(fltAngle1,z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(fltAngle2,z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint('x -' +str(w) + '/2')
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint('-x +' +str(w) + '/2')

fltDistance = objFullCell1.GetNearestNeighbourDistance()/2
##B

objBaseLeft = cp.deepcopy(objLeftCell1)
objBaseRight = cp.deepcopy(objRightCell2)
MySimulationCell.AddGrain(objBaseLeft)
MySimulationCell.AddGrain(objBaseRight)
MySimulationCell.RemoveTooCloseAtoms(fltDistance)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
MySimulationCell.WriteLAMMPSDataFile('/home/p17992pt/LAMMPSData/read0.dat')
MySimulationCell.RemoveAllGrains()


for j in range(1,11):
    r = a*j
    strSphere = gf.ParseConic([w/2,l/2,h/2],[r,r,r],[2,2,2])
    objSphere3 = cp.deepcopy(objFullCell3)
    objSphere3.ApplyGeneralConstraint(strSphere)
    objLeftChopped1 = cp.deepcopy(objLeftCell1)
    objLeftChopped1.ApplyGeneralConstraint(gf.InvertRegion(strSphere))
    objRightChopped2 = cp.deepcopy(objRightCell2)
    objRightChopped2.ApplyGeneralConstraint(gf.InvertRegion(strSphere))
    MySimulationCell.AddGrain(objSphere3)
    MySimulationCell.AddGrain(objLeftChopped1)
    MySimulationCell.AddGrain(objRightChopped2)
    MySimulationCell.RemoveTooCloseAtoms(fltDistance)
    MySimulationCell.WrapAllAtomsIntoSimulationCell()
    MySimulationCell.WriteLAMMPSDataFile('/home/p17992pt/LAMMPSData/read' + str(j) + '.dat')
    MySimulationCell.RemoveAllGrains()




