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
fig = plt.figure(figsize=plt.figaspect(1)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
ax = fig.gca(projection='3d')

strDirectory = str(sys.argv[1])
intSigma = int(sys.argv[2])
lstAxis = eval(str(sys.argv[3]))
intIncrements = int(sys.argv[4])
arrAxis = np.array(lstAxis)
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
arrSigmaBasis = objSigma.GetBasisVectors()
#i = 3 #scaling parameter
s1 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
s2 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
s3 = np.linalg.norm(arrSigmaBasis, axis=1)[2]
#i = np.sqrt(np.abs(np.dot(np.cross(arrSigmaBasis[0],arrSigmaBasis[1]),arrSigmaBasis[2])))
fltAreaFactor = np.sqrt(intSigma/s3)
i = np.round(8/fltAreaFactor).astype('int')
intHeight = 5
a = 4.05 ##lattice parameter
r = 2*a*s2*i
###First part runs with left displaced cylinder
w = 20*a*i
l = 8*a*i
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
arrCylinderCentreGB = 5*a*(arrSigmaBasis[0] + arrSigmaBasis[1])*i
objSimulationCellGBLeft = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCellCentreLeft = objSimulationCellGBLeft.GetCentre()
objFullCell1 = gl.ExtrudedParallelogram(arrX,arrXY,s3*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,arrShift)
objFullCell1.SetPeriodicity(['n','p','p'])
objFullCell2.SetPeriodicity(['n','p','p'])
objFullCell3.SetPeriodicity(['n','n','n'])
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCellCentreLeft[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentreLeft[0]) + ')' 
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)

arrCylinderCentreLeft = 5*a*i*(arrSigmaBasis[0] +arrSigmaBasis[1])+arrShift
#arrCylinderCentreLeft = np.loadtxt(strDirectory + 'Errors/CylinderCentreLeft.txt')
#np.savetxt(strDirectory + 'CylinderCentreLeft.txt',arrCylinderCentreLeft)

strCylinderLeft = gf.ParseConic([arrCylinderCentreLeft[0],arrCylinderCentreLeft[1]],[r,r],[2,2])
objCylinderLeft = cp.deepcopy(objFullCell3)
objCylinderLeft.ApplyGeneralConstraint(strCylinderLeft)
objLeft1Chopped = cp.deepcopy(objLeftCell1)
objLeft1Chopped.ApplyGeneralConstraint(gf.InvertRegion(strCylinderLeft))
objSimulationCellGBLeft.AddGrain(objLeft1Chopped)
objSimulationCellGBLeft.AddGrain(objRightCell2)
objSimulationCellGBLeft.AddGrain(objCylinderLeft)
objSimulationCellGBLeft.RemoveGrainPeriodicDuplicates()

#Second part runs with right displaced cylinder 

objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)

objSimulationCellGBRight = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCylinderCentreRight = arrCylinderCentreLeft + 10*a*i*arrSigmaBasis[0] 
#arrCylinderCentreRight = np.loadtxt(strDirectory + 'Errors/CylinderCentreRight.txt')
#np.savetxt(strDirectory + 'CylinderCentreRight.txt',arrCylinderCentreRight)

strCylinderRight = gf.ParseConic([arrCylinderCentreRight[0],arrCylinderCentreRight[1]],[r,r],[2,2])
objCylinderRight = cp.deepcopy(objFullCell3)
objCylinderRight.ApplyGeneralConstraint(strCylinderRight)
objRight2Chopped = cp.deepcopy(objRightCell2)
objRight2Chopped.ApplyGeneralConstraint(gf.InvertRegion(strCylinderRight))
objSimulationCellGBRight.AddGrain(objLeftCell1)
objSimulationCellGBRight.AddGrain(objRight2Chopped)
objSimulationCellGBRight.AddGrain(objCylinderRight)
objSimulationCellGBRight.RemoveGrainPeriodicDuplicates()



##Third part with triple lines

w = 16*a*i
l = 10*a*i
h = a*np.round(intHeight/s3,0)
arrX = w*arrSigmaBasis[0]
arrXY = l*arrSigmaBasis[1]
z = h*arrSigmaBasis[2]

objSimulationCellTJ = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCellCentreTJ = objSimulationCellTJ.GetCentre()
objFullCell1 = gl.ExtrudedParallelogram(arrX,arrXY,s3*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,arrShift)
objFullCell1.SetPeriodicity(['n','p','p'])
objFullCell2.SetPeriodicity(['n','p','p'])
objFullCell3.SetPeriodicity(['n','n','n'])
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCellCentreTJ[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentreTJ[0]) + ')' 
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)



arrGrainCentreTJ = 3*a*i*arrSigmaBasis[0] + arrCylinderCentreLeft
#arrGrainCentreTJ = np.loadtxt(strDirectory + 'Errors/GrainCentreTJ.txt')

#np.savetxt(strDirectory + 'GrainCentreTJ.txt',arrGrainCentreTJ)
w = 16*a*i
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
objSimulationCellTJ = gl.SimulationCell(np.array([arrX,arrXY, z])) 

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


strCylinder = gf.ParseConic([arrGrainCentreTJ[0],arrGrainCentreTJ[1]],[r,r],[2,2])
objCylinderTJ = cp.deepcopy(objFullCell3)
objCylinderTJ.ApplyGeneralConstraint(strCylinder)
objLeftChopped1 = cp.deepcopy(objLeftCell1)
objLeftChopped1.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
objRightChopped2 = cp.deepcopy(objRightCell2)
objRightChopped2.ApplyGeneralConstraint(gf.InvertRegion(strCylinder))
objSimulationCellTJ.AddGrain(objLeftChopped1)
objSimulationCellTJ.AddGrain(objRightChopped2)
objSimulationCellTJ.AddGrain(objCylinderTJ)
objSimulationCellTJ.RemoveGrainPeriodicDuplicates()


for j in range(intIncrements):
    fltDistance = objFullCell1.GetNearestNeighbourDistance()*j/10
    objSimulationCellGBLeft.MergeTooCloseAtoms(fltDistance,1,100)
    objSimulationCellGBLeft.WrapAllAtomsIntoSimulationCell()
    objSimulationCellGBLeft.SetFileHeader('Grain centre is ' +str(arrCylinderCentreLeft))
    strFileNameGB = 'left' + str(j)
    objSimulationCellGBLeft.WriteLAMMPSDataFile(strDirectory + strFileNameGB + '.dat')
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', strFileNameGB + '.dat')
    fData = fData.replace('read.dmp', strFileNameGB + '.dmp')
    fData = fData.replace('read.lst', strFileNameGB + '.lst')
    fData = fData.replace('logfile', 'logLeft' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateLeft' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()
    ####
    objSimulationCellGBRight.MergeTooCloseAtoms(fltDistance,1,100)
    objSimulationCellGBRight.WrapAllAtomsIntoSimulationCell()
    objSimulationCellGBRight.SetFileHeader('Grain centre is ' +str(arrCylinderCentreRight))
    strFileNameGB = 'right' + str(j)
    objSimulationCellGBRight.WriteLAMMPSDataFile(strDirectory + strFileNameGB + '.dat')
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', strFileNameGB + '.dat')
    fData = fData.replace('read.dmp', strFileNameGB + '.dmp')
    fData = fData.replace('read.lst', strFileNameGB + '.lst')
    fData = fData.replace('logfile', 'logRight' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateRight' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()

    objSimulationCellTJ.MergeTooCloseAtoms(fltDistance,1,100)
    objSimulationCellTJ.WrapAllAtomsIntoSimulationCell()
    objSimulationCellTJ.SetFileHeader('Grain centre is ' +str(arrGrainCentreTJ))
    strFileNameTJ = 'TJ' + str(j)
    objSimulationCellTJ.WriteLAMMPSDataFile(strDirectory + strFileNameTJ + '.dat')
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', strFileNameTJ + '.dat')
    fData = fData.replace('read.dmp', strFileNameTJ + '.dmp')
    fData = fData.replace('read.lst', strFileNameTJ + '.lst')
    fData = fData.replace('logfile', 'logTJ' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateTJ' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()
