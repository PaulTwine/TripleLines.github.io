import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp


fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.gca(projection='3d')
arrAxis = np.array([0,0,1])
intSigma = 5
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
gf.CubicCSLGenerator(arrAxis, 5)
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
s = np.linalg.norm(arrSigmaBasis, axis=1)[0]
a = 4.05 ##lattice parameter
#a2 = a1*np.sqrt(3) #periodic cell repeat multiple
w = 10*a*s
l = 10*a*s
h= 10*a
z = np.array([0,0,h])
arr100BasisVectors = gf.StandardBasisVectors(3)
arrHorizontalVector = np.array([w,0,0])
arrUpVector =  np.array([0, l,0]) 
r = h/3
arrLatticeParameters= np.array([a,a,a])
MySimulationCell = gl.SimulationCell(np.array([arrHorizontalVector,arrUpVector, z])) 
objFullCell1 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(fltAngle1,z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(fltAngle2,z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedRectangle(w-0.1,l-0.1, h-0.1, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arr100BasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
strSphere = gf.ParseConic([w/2,l/2,h/2],[r,r,r],[2,2,2])
objFullCell1Chopped = cp.deepcopy(objFullCell1)
objFullCell1Chopped.ApplyGeneralConstraint(gf.InvertRegion(strSphere))
objFullCell2Chopped = cp.deepcopy(objFullCell2)
objFullCell2Chopped.ApplyGeneralConstraint(gf.InvertRegion(strSphere))
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint('x -' +str(w) + '/2')
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint('-x +' +str(w) + '/2')
objSphere1 = cp.deepcopy(objFullCell1)
objSphere1.ApplyGeneralConstraint(strSphere)
objRightHemiSphere1 = cp.deepcopy(objSphere1)
objRightHemiSphere1.ApplyGeneralConstraint('-x +' +str(w) + '/2')
objSphere2 = cp.deepcopy(objFullCell2)
objSphere2.ApplyGeneralConstraint(strSphere)
objLeftHemiSphere2 = cp.deepcopy(objSphere2)
objLeftHemiSphere2.ApplyGeneralConstraint('x -' +str(w) + '/2')
objSphere3 = cp.deepcopy(objFullCell3)
objSphere3.ApplyGeneralConstraint(strSphere)
objLeftChopped1 = cp.deepcopy(objLeftCell1)
objLeftChopped1.ApplyGeneralConstraint(gf.InvertRegion(strSphere))
objRightChopped2 = cp.deepcopy(objRightCell2)
objRightChopped2.ApplyGeneralConstraint(gf.InvertRegion(strSphere))

#ax.scatter(*objLeftCell1.MatLabPlot(), s=0.4, c='y')
#ax.scatter(*objSphere1.MatLabPlot(), s=0.4, c='y')
#ax.scatter(*objRightChopped2.MatLabPlot(), s=0.4)
#gf.EqualAxis3D(ax)
#plt.show()


## (A) Energy of left bulged grain and right reduced grain
MySimulationCell.AddGrain(objLeftCell1)
MySimulationCell.AddGrain(objRightHemiSphere1)
MySimulationCell.AddGrain(objRightChopped2)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
#ax.scatter(*MySimulationCell.PlotSimulationCellAtoms(), s=0.4)
#fig.show()
intA, fltA = MySimulationCell.LAMMPSMinimisePositions('/home/p17992pt/LAMMPSData/','Aread.dat','TemplateMin.in',10, -3.36)

MySimulationCell.RemoveAllGrains()

## Energy of right bulged grain and left reduced grain
MySimulationCell.AddGrain(objRightCell2)
MySimulationCell.AddGrain(objLeftHemiSphere2)
MySimulationCell.AddGrain(objLeftChopped1)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
#ax.scatter(*MySimulationCell.PlotSimulationCellAtoms(), s=0.4)
#fig.show()
intB, fltB = MySimulationCell.LAMMPSMinimisePositions('/home/p17992pt/LAMMPSData/','Bread.dat','TemplateMin.in',10, -3.36)



MySimulationCell.RemoveAllGrains()

##Energy of sphere 1 inside grain 2

MySimulationCell.AddGrain(objSphere1)
MySimulationCell.AddGrain(objFullCell2Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
#ax.scatter(*MySimulationCell.PlotSimulationCellAtoms(), s=0.4)
#plt.show()
intC, fltC = MySimulationCell.LAMMPSMinimisePositions('/home/p17992pt/LAMMPSData/','Cread.dat','TemplateMin.in',10, -3.36)

##Energy of sphere 2 inside grain 1

MySimulationCell.RemoveAllGrains()

MySimulationCell.AddGrain(objSphere2)
MySimulationCell.AddGrain(objFullCell1Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
intD, fltD = MySimulationCell.LAMMPSMinimisePositions('/home/p17992pt/LAMMPSData/','Dread.dat','TemplateMin.in',10, -3.36)
#ax.scatter(*MySimulationCell.PlotSimulationCellAtoms(), s=0.4)

MySimulationCell.RemoveAllGrains()

intStep1 = max([intA,intB,intC,intD])

fltExcess1 = (fltA/intA + fltB/intB -0.5*(fltC/intC + fltD/intD)+3.36)*intStep1

print(fltExcess1)

MySimulationCell.AddGrain(objSphere3)
MySimulationCell.AddGrain(objLeftChopped1)
MySimulationCell.AddGrain(objRightChopped2)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
intE, fltE = MySimulationCell.LAMMPSMinimisePositions('/home/p17992pt/LAMMPSData/','Eread.dat','TemplateMin.in',10, -3.36)

MySimulationCell.RemoveAllGrains()

MySimulationCell.AddGrain(objSphere3)
MySimulationCell.AddGrain(objFullCell1Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
intF, fltF = MySimulationCell.LAMMPSMinimisePositions('/home/p17992pt/LAMMPSData/','Fread.dat','TemplateMin.in',10, -3.36)

MySimulationCell.RemoveAllGrains()

MySimulationCell.AddGrain(objSphere3)
MySimulationCell.AddGrain(objFullCell2Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
intG, fltG = MySimulationCell.LAMMPSMinimisePositions('/home/p17992pt/LAMMPSData/','Gread.dat','TemplateMin.in',10, -3.36)


intStep2 = max([intE,intF,intG])

print((fltE/intE - 0.5*(fltF/intF+fltG/intG +fltExcess1/intStep1))*intStep2

# def ExcessEnergy(fltPE, intAtoms, fltDatum):
#     return (fltPE-intAtoms*fltDatum)*intMax/intAtoms
# print(ExcessEnergy(fltAllPE,intAllAtoms, -3.36) - ExcessEnergy(fltSpherePE,intSphereAtoms,-3.36)- ExcessEnergy(fltTwinPE,intTwinAtoms,-3.36))






