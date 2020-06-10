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
fltAngle1 = 20
fltAngle2 = 80
#a = 4.05/np.sqrt(2) ##lattice parameter
a= 3.230676
#a2 = a1*np.sqrt(3) #periodic cell repeat multiple
w = 50*a
l = np.sqrt(3)*w
h= 10*np.sqrt(8/3)*a
z = np.array([0,0,h])
strDirectory = '../../data' # + str(intCounter) 
strDataFile = 'read.data' +str(intCounter)
strDumpFile = 'dump.eam' +str(intCounter)
#arr111BasisVectors = gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2))
arr100BasisVectors = gf.StandardBasisVectors(3)
arrHorizontalVector = np.array([w,0,0])
arrUpVector =  np.array([0, l,0]) 
r =0.96*l/2
arrLatticeParameters= np.array([a,a,np.sqrt(8/3)*a])
MySimulationCell = gl.SimulationCell(np.array([2*arrHorizontalVector,arrUpVector, z])) 
objTop = gl.ExtrudedRectangle(2*w,l/2, h, gf.RotateVectors(gf.DegreesToRadians(fltAngle2),z,arr100BasisVectors), ld.HCPCell, arrLatticeParameters,arrUpVector/2,ld.HCPBasisVectors)
objTop.ApplyGeneralConstraint('((x-'+str(w)+')/'+str(w/2)+')**2 +((y -' +str(l) +')/' +str(r) + ')**2-1' ,'[x,y,z]')
objBottom = gl.ExtrudedRectangle(2*w,l/2, h,gf.RotateVectors(gf.DegreesToRadians(fltAngle1),z, arr100BasisVectors), ld.HCPCell, arrLatticeParameters,np.zeros([3]),ld.HCPBasisVectors)
objBottom.ApplyGeneralConstraint('((x-'+str(w)+')/'+str(w/2)+')**2 +(y /' +str(r) + ')**2-1' ,'[x,y,z]')
objMiddle = gl.ExtrudedRectangle(2*w, l, h, arr100BasisVectors, ld.HCPCell, arrLatticeParameters,np.zeros([3]),ld.HCPBasisVectors)
objMiddle.ApplyGeneralConstraint('-((x-'+str(w)+')/'+str(w/2)+')**2 -((y -' +str(l) +')/' +str(r) + ')**2+1' ,'[x,y,z]')
objMiddle.ApplyGeneralConstraint('-((x-'+str(w)+')/'+str(w/2)+')**2 -(y /' +str(r) + ')**2+1' ,'[x,y,z]')
MySimulationCell.AddGrain(objTop)
MySimulationCell.AddGrain(objBottom)
MySimulationCell.AddGrain(objMiddle)
MySimulationCell.WrapAllPointsIntoSimulationCell()
MySimulationCell.RemovePlaneOfAtoms(np.array([[0,0,1,z[2]]]),0.001)
#MySimulationCell.RemovePlaneOfAtoms(np.array([[0,l,0,l]]),0.001)
MySimulationCell.SetFileHeader('Angle1 = ' + str(fltAngle1) + ' Angle2 = ' + str(fltAngle2) + 'Axis = [100]')
MySimulationCell.WriteLAMMPSDataFile('readPoly.data')
# fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'rt')
# fData = fIn.read()
# fData = fData.replace('new.data', strDataFile)
# fData = fData.replace('dump.eam', strDumpFile)
# fIn.close()
# fIn = open(strDirectory + '/' + 'TemplateNVT.in', 'wt')
# fIn.write(fData)
# fIn.close()





