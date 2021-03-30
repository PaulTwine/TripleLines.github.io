import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp


strDirectory = sys.argv[1]
intSigma = sys.argv[2]

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
w = 15*a*s
l = 15*a*s
h= 15*a*np.round(s)
z = np.array([0,0,h])
arr100BasisVectors = gf.StandardBasisVectors(3)
arrHorizontalVector = np.array([w,0,0])
arrUpVector =  np.array([0, l,0]) 
r = l/6
arrLatticeParameters= np.array([a,a,a])
fltDatum = -3.36
dctValues = dict()

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

##A
MySimulationCell.AddGrain(objLeftCell1)
MySimulationCell.AddGrain(objRightHemiSphere1)
MySimulationCell.AddGrain(objRightChopped2)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
dctValues['A'] = MySimulationCell.LAMMPSMinimisePositions(strDirectory,'Aread.dat','TemplateMin.in',10, fltDatum)
MySimulationCell.RemoveAllGrains()

##B
MySimulationCell.AddGrain(objRightCell2)
MySimulationCell.AddGrain(objLeftHemiSphere2)
MySimulationCell.AddGrain(objLeftChopped1)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
dctValues['B'] = MySimulationCell.LAMMPSMinimisePositions(strDirectory,'Bread.dat','TemplateMin.in',10, fltDatum)
MySimulationCell.RemoveAllGrains()

##C
MySimulationCell.AddGrain(objSphere1)
MySimulationCell.AddGrain(objFullCell2Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
dctValues['C'] = MySimulationCell.LAMMPSMinimisePositions(strDirectory,'Cread.dat','TemplateMin.in',10, fltDatum)
MySimulationCell.RemoveAllGrains()

##D
MySimulationCell.AddGrain(objSphere2)
MySimulationCell.AddGrain(objFullCell1Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
dctValues['D'] = MySimulationCell.LAMMPSMinimisePositions(strDirectory,'Dread.dat','TemplateMin.in',10, fltDatum)
MySimulationCell.RemoveAllGrains()

##E
MySimulationCell.AddGrain(objSphere3)
MySimulationCell.AddGrain(objLeftChopped1)
MySimulationCell.AddGrain(objRightChopped2)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
dctValues['E'] = MySimulationCell.LAMMPSMinimisePositions(strDirectory,'Eread.dat','TemplateMin.in',10, fltDatum)
MySimulationCell.RemoveAllGrains()

##F
MySimulationCell.AddGrain(objSphere3)
MySimulationCell.AddGrain(objFullCell1Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
dctValues['F'] = MySimulationCell.LAMMPSMinimisePositions(strDirectory,'Fread.dat','TemplateMin.in',10, fltDatum)
MySimulationCell.RemoveAllGrains()

##G
MySimulationCell.AddGrain(objSphere3)
MySimulationCell.AddGrain(objFullCell2Chopped)
MySimulationCell.WrapAllAtomsIntoSimulationCell()
dctValues['G'] =  MySimulationCell.LAMMPSMinimisePositions(strDirectory,'Gread.dat','TemplateMin.in',10, fltDatum)
MySimulationCell.RemoveAllGrains()

print(dctValues)

np.savetxt('dctValues', str(dctValues))

lstAtoms = []
for j in dctValues.keys():
    lstAtoms.append(dctValues[j][0])
intMax = max(lstAtoms)


dctNormalised = dict()
for j in dctValues.keys():
    dctNormalised[j] = dctValues[j][1]+fltDatum*(intMax-dctValues[j][0])

print(dctNormalised)

np.savetxt('dctNormalised', str(dctNormalised))

fltExcess = 0.5*(dctNormalised['A'] + dctNormalised['B'] -fltDatum*intMax)-0.25*(dctNormalised['C'] + dctNormalised['D']) 

fltTJ = dctNormalised['E']-0.5*(dctNormalised['F']+dctNormalised['G'])

print(fltTJ-fltExcess)

np.savetxt('TJEnergy', str(fltTJ-fltExcess))




