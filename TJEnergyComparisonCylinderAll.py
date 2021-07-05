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
fltFactor = float(sys.argv[4])
lstAxis = eval(str(sys.argv[3]))
arrAxis = np.array(lstAxis)
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
intMax = 60
intHeight = 5
i = 4 #scaling parameter
s1 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
s2 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
s3 = np.linalg.norm(arrSigmaBasis, axis=1)[2]
a = 4.05 ##lattice parameter
r = 3*a*s2*i/2
###First part runs with displaced cylinder
w = 20*a*i
l = 9*a*i
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
arrShift = (a*(0.5-np.random.ranf())*arrSigmaBasis[1]+a*(0.5-np.random.ranf())*arrSigmaBasis[2])
arrCellCentre = 0.5*(arrX+arrXY+z) 
arrCylinderCentre = 5*a*(arrSigmaBasis[0] + arrSigmaBasis[1])*i
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCellCentre[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentre[0]) + ')' 
objSimulationCell1 = gl.SimulationCell(np.array([arrX,arrXY, z])) 
objFullCell1 = gl.ExtrudedParallelogram(arrX,arrXY,s3*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,arrShift)
objFullCell1.SetPeriodicity(['n','p','p'])
objFullCell2.SetPeriodicity(['n','p','p'])
objFullCell3.SetPeriodicity(['n','n','n'])
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)

fltDistance = objFullCell1.GetNearestNeighbourDistance()

arrGrainCentre0 = 5*a*i*(arrSigmaBasis[0] +arrSigmaBasis[1])

np.savetxt(strDirectory + 'GrainCentre0.txt',arrGrainCentre0)

strCylinder = gf.ParseConic([arrGrainCentre0[0],arrGrainCentre0[1]],[r,r],[2,2])
objCylinder3 = cp.deepcopy(objFullCell3)
objCylinder3.ApplyGeneralConstraint(strCylinder)
objLeftChopped1 = cp.deepcopy(objLeftCell1)
objLeftChopped1.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
objRightChopped2 = cp.deepcopy(objRightCell2)
objRightChopped2.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
objSimulationCell1.AddGrain(objLeftChopped1)
objSimulationCell1.AddGrain(objRightChopped2)
objSimulationCell1.AddGrain(objCylinder3)
objSimulationCell1.RemoveAtomsOnOpenBoundaries()
objSimulationCell1.RemovePeriodicDuplicates()
objSimulationCell1.RemoveTooCloseAtoms(fltDistance*fltFactor, ['1','2'])
objSimulationCell1.RemoveTooCloseAtoms(fltDistance*fltFactor,['1','3'])
objSimulationCell1.WrapAllAtomsIntoSimulationCell()
objSimulationCell1.SetFileHeader('Grain centre is ' +str(arrGrainCentre0))
objSimulationCell1.WriteLAMMPSDataFile(strDirectory + 'read0.dat')
objSimulationCell1.RemoveAllGrains()
objSimulationCell1.RemoveNonGrainAtomPositons()
fIn = open(strDirectory +  'TemplateMin.in', 'rt')
fData = fIn.read()
fData = fData.replace('read.dat', 'read0.dat')
fData = fData.replace('read.dmp', 'read0.dmp')
fData = fData.replace('read.lst', 'read0.lst')
fData = fData.replace('logfile', 'logfile0')
fIn.close()
fIn = open(strDirectory + 'TemplateMin0.in', 'wt')
fIn.write(fData)
fIn.close()

##Second part with triple lines

arrGrainCentre1 = (9*a*arrSigmaBasis[0] +5*a*arrSigmaBasis[1])*i
np.savetxt(strDirectory + 'GrainCentre0.txt',arrGrainCentre1)
w = 18*a*i
l = 10*a*i
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
arrShift = a*(0.5-np.random.ranf())*arrSigmaBasis[1]+a*(0.5-np.random.ranf())*arrSigmaBasis[2]
arrCellCentre = 0.5*(arrX+arrXY+z)
arrCylinderCentre = 5*a*i*(arrSigmaBasis[0] + arrSigmaBasis[1])
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCellCentre[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentre[0]) + ')' 
objSimulationCell2 = gl.SimulationCell(np.array([arrX,arrXY, z])) 
objFullCell1 = gl.ExtrudedParallelogram(arrX,arrXY,s3*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,arrShift)
objFullCell1.SetPeriodicity(['n','p','p'])
objFullCell2.SetPeriodicity(['n','p','p'])
objFullCell3.SetPeriodicity(['n','n','n'])
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)

fltDistance = objFullCell1.GetNearestNeighbourDistance()


strCylinder = gf.ParseConic([arrGrainCentre1[0],arrGrainCentre1[1]],[r,r],[2,2])
objCylinder3 = cp.deepcopy(objFullCell3)
objCylinder3.ApplyGeneralConstraint(strCylinder)
objLeftChopped1 = cp.deepcopy(objLeftCell1)
objLeftChopped1.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
objRightChopped2 = cp.deepcopy(objRightCell2)
objRightChopped2.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
objSimulationCell2.AddGrain(objLeftChopped1)
objSimulationCell2.AddGrain(objRightChopped2)
objSimulationCell2.AddGrain(objCylinder3)
objSimulationCell2.RemoveAtomsOnOpenBoundaries()
objSimulationCell2.RemovePeriodicDuplicates()
objSimulationCell2.RemoveTooCloseAtoms(fltDistance*fltFactor, ['1','2'])
objSimulationCell2.RemoveTooCloseAtoms(fltDistance*fltFactor,['1','3'])
objSimulationCell2.WrapAllAtomsIntoSimulationCell()
objSimulationCell2.SetFileHeader('Grain centre is ' +str(arrGrainCentre1))
objSimulationCell2.WriteLAMMPSDataFile(strDirectory + 'read1.dat')
objSimulationCell2.RemoveAllGrains()
objSimulationCell2.RemoveNonGrainAtomPositons()
fIn = open(strDirectory +  'TemplateMin.in', 'rt')
fData = fIn.read()
fData = fData.replace('read.dat', 'read1.dat')
fData = fData.replace('read.dmp', 'read1.dmp')
fData = fData.replace('read.lst', 'read1.lst')
fData = fData.replace('logfile', 'logfile1')
fIn.close()
fIn = open(strDirectory + 'TemplateMin1.in', 'wt')
fIn.write(fData)
fIn.close()
