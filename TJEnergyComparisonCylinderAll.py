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
s0 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
s1 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
s2 = np.linalg.norm(arrSigmaBasis, axis=1)[2]
intHeight = 5
intAtoms = 10**5
intAtomsPerCell = 4
a = 4.05 ##lattice parameter
h = a*np.round(intHeight/s2,0)
i = np.sqrt((intAtoms/intAtomsPerCell)*a/(160*h*s0*s1))
i = np.round(i,0).astype('int')
r = 2*a*s1*i
###First part runs with left displaced cylinder
w = 20*a*i
l = 8*a*i

arrX = w*arrSigmaBasis[0]
arrXY = l*arrSigmaBasis[1]
z = h*arrSigmaBasis[2]

if np.all(arrAxis == np.array([1,0,0])):
    arrBasisVectors = gf.StandardBasisVectors(3)
else:
    fltAngle3, arrRotation = gf.FindRotationVectorAndAngle(arrAxis,np.array([0,0,1]))
    arrBasisVectors = gf.RotateVectors(fltAngle3, arrRotation,gf.StandardBasisVectors(3))

arrLatticeParameters= np.array([a,a,a])
objCylinder = gl.ExtrudedCylinder(r,h*s2,arrBasisVectors,ld.FCCCell,arrLatticeParameters,np.zeros(3))
objCylinder.SetPeriodicity(['n','n','p'])

arrRandom = (a*(0.5-np.random.ranf())*arrSigmaBasis[1]+a*(0.5-np.random.ranf())*arrSigmaBasis[2])
objSimulationCellGBLeft = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCellCentreLeft = objSimulationCellGBLeft.GetCentre()
arrCylinderCentreLeft = 0.5*arrXY+0.25*arrX + arrRandom 
objFullLeft = gl.ExtrudedParallelogram(arrX,arrXY,s2*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullRight = gl.ExtrudedParallelogram(arrX,arrXY, s2*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullLeft.SetPeriodicity(['n','p','p'])
objFullRight.SetPeriodicity(['n','p','p'])
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCellCentreLeft[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentreLeft[0]) + ')' 
objLeftCell = cp.deepcopy(objFullLeft)
objLeftCell.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell = cp.deepcopy(objFullRight)
objRightCell.ApplyGeneralConstraint(strConstraint)


np.savetxt(strDirectory + 'CylinderCentreLeft.txt',arrCylinderCentreLeft)

strCylinderLeft = gf.ParseConic([arrCylinderCentreLeft[0],arrCylinderCentreLeft[1]],[r,r],[2,2])
objCylinderLeft = cp.deepcopy(objCylinder)
objCylinderLeft.TranslateGrain(arrCylinderCentreLeft)
objLeftChopped = cp.deepcopy(objLeftCell)
objLeftChopped.ApplyGeneralConstraint(gf.InvertRegion(strCylinderLeft))
objSimulationCellGBLeft.AddGrain(objLeftChopped)
objSimulationCellGBLeft.AddGrain(objRightCell)
objSimulationCellGBLeft.AddGrain(objCylinderLeft)
objSimulationCellGBLeft.RemoveGrainPeriodicDuplicates()

#Second part runs with right displaced cylinder 

objSimulationCellGBRight = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCylinderCentreRight = arrCylinderCentreLeft + 0.5*arrX
arrCellCentreRight = objSimulationCellGBRight.GetCentre()



np.savetxt(strDirectory + 'CylinderCentreRight.txt',arrCylinderCentreRight)

strCylinderRight = gf.ParseConic([arrCylinderCentreRight[0],arrCylinderCentreRight[1]],[r,r],[2,2])
objCylinderRight = cp.deepcopy(objCylinder)
objCylinderRight.TranslateGrain(arrCylinderCentreRight)
objRightChopped = cp.deepcopy(objRightCell)
objRightChopped.ApplyGeneralConstraint(gf.InvertRegion(strCylinderRight))
objSimulationCellGBRight.AddGrain(objLeftCell)
objSimulationCellGBRight.AddGrain(objRightChopped)
objSimulationCellGBRight.AddGrain(objCylinderRight)
objSimulationCellGBRight.RemoveGrainPeriodicDuplicates()



##Third part with triple lines and central cylinder
w = 16*a*i
l = 10*a*i
h = a*np.round(intHeight/s2,0)
arrXTJ = w*arrSigmaBasis[0]
arrXYTJ = l*arrSigmaBasis[1]

objSimulationCellTJ = gl.SimulationCell(np.array([arrXTJ,arrXYTJ, z])) 
arrCellCentreTJ = objSimulationCellTJ.GetCentre()
arrCylinderCentre = 0.5*arrXTJ + 0.4*arrXYTJ+arrRandom
np.savetxt(strDirectory + 'CylinderCentreMiddle.txt',arrCylinderCentre)
objFullLeft = gl.ExtrudedParallelogram(arrXTJ,arrXYTJ,s2*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullRight = gl.ExtrudedParallelogram(arrXTJ,arrXYTJ, s2*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullLeft.SetPeriodicity(['n','p','p'])
objFullRight.SetPeriodicity(['n','p','p'])
strConstraint = str(arrXYTJ[0])+ '*(y -' + str(arrCellCentreTJ[1]) + ') - ' + str(arrXYTJ[1]) + '*(x -' + str(arrCellCentreTJ[0]) + ')' 
objLeftCell = cp.deepcopy(objFullLeft)
objLeftCell.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell = cp.deepcopy(objFullRight)
objRightCell.ApplyGeneralConstraint(strConstraint)


objLeftHalfChopped = cp.deepcopy(objFullLeft)
objLeftHalfChopped.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightHalfChopped = cp.deepcopy(objFullRight)
objRightHalfChopped.ApplyGeneralConstraint(strConstraint)


strCylinderTJ = gf.ParseConic([arrCylinderCentre[0],arrCylinderCentre[1]],[r,r],[2,2])
objCylinderTJ = cp.deepcopy(objCylinder)
objCylinderTJ.TranslateGrain(arrCylinderCentre)
objCylinderTJ.ApplyGeneralConstraint(strCylinderTJ)
objLeftHalfChopped.ApplyGeneralConstraint(gf.InvertRegion(strCylinderTJ))
objRightHalfChopped.ApplyGeneralConstraint(gf.InvertRegion(strCylinderTJ))
objSimulationCellTJ.AddGrain(objLeftHalfChopped)
objSimulationCellTJ.AddGrain(objRightHalfChopped)
objSimulationCellTJ.AddGrain(objCylinderTJ)
objSimulationCellTJ.RemoveGrainPeriodicDuplicates()


for j in range(intIncrements):
    fltDistance = objFullLeft.GetNearestNeighbourDistance()*j/10
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
    objSimulationCellTJ.SetFileHeader('Grain centre is ' +str(arrCylinderCentre))
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
