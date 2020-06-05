import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import sys
from mpl_toolkits.mplot3d import Axes3D 

intCounter = int(1)
intIncrement = float(15)
fltSymmetry = float(90)
#fltAngle1, fltAngle2 = gf.AngleGenerator(intCounter, intIncrement, fltSymmetry)
fltAngle1 = 30
fltAngle2 = 60
a1 = 4.05 ##lattice parameter
#a2 = a1*np.sqrt(3) #periodic cell repeat multiple
l = 40*a1
h= 5*a1
z = np.array([0,0,h])
strDirectory = '../../data' # + str(intCounter) 
strDataFile = 'read.data' +str(intCounter)
strDumpFile = 'dump.eam' +str(intCounter)
#arr111BasisVectors = gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2))
arr100BasisVectors = gf.StandardBasisVectors(3)
arrHorizontalVector = np.array([l,0,0])
arrUpVector =  np.array([0, l,0]) 
r =0.99*l
MySimulationCell = gl.SimulationCell(np.array([2*arrHorizontalVector,2*arrUpVector, z])) 
objTop = gl.ExtrudedRectangle(2*l,l, h, gf.RotateVectors(gf.DegreesToRadians(fltAngle2),z,arr100BasisVectors), ld.FCCCell, np.array([a1,a1,a1]),arrUpVector)
objTop.ApplyGeneralConstraint('((x-'+str(l)+')/'+str(l/2)+')**2 +((y -' +str(2*l) +')/' +str(r) + ')**2-1' ,'[x,y,z]')
objBottom = gl.ExtrudedRectangle(2*l,l, h,gf.RotateVectors(gf.DegreesToRadians(fltAngle1),z, arr100BasisVectors), ld.FCCCell, np.array([a1,a1,a1]))
objBottom.ApplyGeneralConstraint('((x-'+str(l)+')/'+str(l/2)+')**2 +(y /' +str(r) + ')**2-1' ,'[x,y,z]')
objMiddle = gl.ExtrudedRectangle(2*l, 2*l, h, arr100BasisVectors, ld.FCCCell, np.array([a1,a1,a1]))
objMiddle.ApplyGeneralConstraint('-((x-'+str(l)+')/'+str(l/2)+')**2 -((y -' +str(2*l) +')/' +str(r) + ')**2+1' ,'[x,y,z]')
objMiddle.ApplyGeneralConstraint('-((x-'+str(l)+')/'+str(l/2)+')**2 -(y /' +str(r) + ')**2+1' ,'[x,y,z]')
MySimulationCell.AddGrain(objTop)
MySimulationCell.AddGrain(objBottom)
MySimulationCell.AddGrain(objMiddle)
MySimulationCell.WrapAllPointsIntoSimulationCell()
MySimulationCell.RemovePlaneOfAtoms(np.array([[0,0,1,z[2]]]),0.001)
MySimulationCell.RemovePlaneOfAtoms(np.array([[0,0,1,l]]),0.001)
MySimulationCell.SetFileHeader('Angle1 = ' + str(fltAngle1) + ' Angle2 = ' + str(fltAngle2) + 'Axis = [100]')
MySimulationCell.WriteLAMMPSDataFile('readPoly.data')
fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'rt')
fData = fIn.read()
fData = fData.replace('new.data', strDataFile)
fData = fData.replace('dump.eam', strDumpFile)
fIn.close()
fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'wt')
fIn.write(fData)
fIn.close()





