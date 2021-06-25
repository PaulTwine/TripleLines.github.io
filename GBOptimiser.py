import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp
from scipy import spatial

strDirectory = '/home/p17992pt/LAMMPSData/' #str(sys.argv[1])
intSigma = 13 # int(sys.argv[2])
lstAxis = [1,0,0]  #eval(str(sys.argv[3]))
arrAxis = np.array(lstAxis)
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
intMax = 40
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
if np.all(arrAxis == np.array([1,0,0])):
    arrBasisVectors = gf.StandardBasisVectors(3)
else:
    fltAngle3, arrRotation = gf.FindRotationVectorAndAngle(arrAxis,np.array([0,0,1]))
    arrBasisVectors = gf.RotateVectors(fltAngle3, arrRotation,gf.StandardBasisVectors(3)) 
arrLatticeParameters= np.array([a,a,a])
arrCentre = 0.5*(arrX+arrXY) 
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCentre[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCentre[0]) + ')' 
objFullCell1 = gl.ExtrudedParallelogram(arrX,arrXY,s3*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell1.SetPeriodicity(['n','p','p'])
objFullCell2.SetPeriodicity(['n','p','p'])
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)
objBaseLeft = cp.deepcopy(objLeftCell1)
objBaseRight = cp.deepcopy(objRightCell2)
MySimulationCell = gl.SimulationCell(np.array([arrX,arrXY, z]))     
MySimulationCell.AddGrain(objBaseLeft)
MySimulationCell.AddGrain(objBaseRight)
fltj = objFullCell1.GetNearestNeighbourDistance() 
lstj = [] 
lstAtoms = []  
for j in range(20):
    lstAtoms.append(MySimulationCell.GetUpdatedAtomNumbers())
    MySimulationCell.RemoveAtomsOnOpenBoundaries()
    MySimulationCell.WrapAllAtomsIntoSimulationCell()
    MySimulationCell.RemoveTooCloseAtoms(j*fltj/20)
    lstAtoms.append(MySimulationCell.GetUpdatedAtomNumbers())
    if lstAtoms[-1] != lstAtoms[-2]:
        MySimulationCell.WriteLAMMPSDataFile(strDirectory + str(j) + '.dat')
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
