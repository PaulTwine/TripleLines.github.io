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
intIncrements =  10 #int(sys.argv[4])
arrAxis = np.array(lstAxis)
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
arrSigmaBasis = objSigma.GetBasisVectors()
s0 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
s1 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
s2 = np.linalg.norm(arrSigmaBasis, axis=1)[2]

intHeight = 4
intAtoms = 1.5*10**5
intAtomsPerCell = 4
a = 4.05 ##lattice parameter
h = a*np.round(intHeight/s2,0)

i = np.sqrt(intAtoms*a/(32*12*intAtomsPerCell*h*np.linalg.det(arrSigmaBasis)))
i = np.round(i,0).astype('int')

arrLatticeParameters= np.array([a,a,a])

arrMedianLattice = objSigma.GetMedianLattice()
lstLattices = objSigma.GetLatticeBases()
arrBasis1 = lstLattices[0]
arrBasis2 = lstLattices[1]

###First part runs with two displaced cylinders and no triple lines
r = 2*a*s1*i
w = 32*a*i
l = 12*a*i



arrX = w*arrSigmaBasis[0]
arrXY = l*arrSigmaBasis[1]
z = h*arrSigmaBasis[2]
objCylinder = gl.ExtrudedCylinder(r,h*s2,arrMedianLattice,ld.FCCCell,arrLatticeParameters,np.zeros(3))
objCylinder.SetPeriodicity(['n','n','p'])

#arrRandom = (a*(0.5-np.random.ranf())*arrSigmaBasis[1]+a*(0.5-np.random.ranf())*arrSigmaBasis[2])

#np.savetxt(strDirectory + 'RandomDisplacement.txt',arrRandom)
arrRandom = np.loadtxt(strDirectory + 'RandomDisplacement.txt')
objSimulationCellGB = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCellCentreGB = objSimulationCellGB.GetCentre()
arrCylinderCentreLeftGB = 0.5*arrXY+0.25*arrX + arrRandom
arrCylinderCentreRightGB =  0.5*arrXY+0.75*arrX + arrRandom
objFullLeft = gl.ExtrudedParallelogram(arrX,arrXY,s2*h, arrBasis1, ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullRight = gl.ExtrudedParallelogram(arrX,arrXY, s2*h, arrBasis2, ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullLeft.SetPeriodicity(['n','p','p'])
objFullRight.SetPeriodicity(['n','p','p'])
strConstraintGB = str(arrXY[0])+ '*(y -' + str(arrCellCentreGB[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentreGB[0]) + ')' 
objLeftCell = cp.deepcopy(objFullLeft)
objLeftCell.ApplyGeneralConstraint(gf.InvertRegion(strConstraintGB))
objRightCell = cp.deepcopy(objFullRight)
objRightCell.ApplyGeneralConstraint(strConstraintGB)

strCylinderLeftGB = gf.ParseConic([arrCylinderCentreLeftGB[0],arrCylinderCentreLeftGB[1]],[r,r],[2,2])
objCylinderLeftGB = cp.deepcopy(objCylinder)
objCylinderLeftGB.TranslateGrain(arrCylinderCentreLeftGB)
objLeftChoppedGB = cp.deepcopy(objLeftCell)
objLeftChoppedGB.ApplyGeneralConstraint(gf.InvertRegion(strCylinderLeftGB))
strCylinderRightGB = gf.ParseConic([arrCylinderCentreRightGB[0],arrCylinderCentreRightGB[1]],[r,r],[2,2])
objCylinderRightGB = cp.deepcopy(objCylinder)
objCylinderRightGB.TranslateGrain(arrCylinderCentreRightGB)
objRightChoppedGB = cp.deepcopy(objRightCell)
objRightChoppedGB.ApplyGeneralConstraint(gf.InvertRegion(strCylinderRightGB))
objSimulationCellGB.AddGrain(objLeftChoppedGB)
objSimulationCellGB.AddGrain(objRightChoppedGB)
objSimulationCellGB.AddGrain(objCylinderLeftGB)
objSimulationCellGB.AddGrain(objCylinderRightGB)
objSimulationCellGB.RemoveGrainPeriodicDuplicates()


##Second part with triple lines
w = 24*a*i
l = 16*a*i
h = a*np.round(intHeight/s2,0)
arrXTJ = w*arrSigmaBasis[0]
arrXYTJ = l*arrSigmaBasis[1]

objSimulationCellTJ = gl.SimulationCell(np.array([arrXTJ,arrXYTJ, z])) 
arrCellCentreTJ = objSimulationCellTJ.GetCentre()
arrCylinderLeftTJ = 0.5*arrXYTJ +arrRandom
arrCylinderRightTJ = 0.5*arrXYTJ + arrXTJ + arrRandom
arrCylinderMiddleTJ = 0.5*(arrXYTJ+arrXTJ) + arrRandom
objFullLeft = gl.ExtrudedParallelogram(arrXTJ,arrXYTJ,s2*h, arrBasis1, ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullRight = gl.ExtrudedParallelogram(arrXTJ,arrXYTJ, s2*h, arrBasis2, ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullLeft.SetPeriodicity(['n','p','p'])
objFullRight.SetPeriodicity(['n','p','p'])
strConstraintTJ = str(arrXYTJ[0])+ '*(y -' + str(arrCellCentreTJ[1]) + ') - ' + str(arrXYTJ[1]) + '*(x -' + str(arrCellCentreTJ[0]) + ')' 
objLeftCell = cp.deepcopy(objFullLeft)
objLeftCell.ApplyGeneralConstraint(gf.InvertRegion(strConstraintTJ))
objRightCell = cp.deepcopy(objFullRight)
objRightCell.ApplyGeneralConstraint(strConstraintTJ)


objLeftHalfChoppedTJ = cp.deepcopy(objFullLeft)
objLeftHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strConstraintTJ))
objRightHalfChoppedTJ = cp.deepcopy(objFullRight)
objRightHalfChoppedTJ.ApplyGeneralConstraint(strConstraintTJ)


strCylinderLeftTJ = gf.ParseConic([arrCylinderLeftTJ[0],arrCylinderLeftTJ[1]],[r,r],[2,2])
strCylinderRightTJ = gf.ParseConic([arrCylinderRightTJ[0],arrCylinderRightTJ[1]],[r,r],[2,2])
strCylinderMiddleTJ = gf.ParseConic([arrCylinderMiddleTJ[0],arrCylinderMiddleTJ[1]],[r,r],[2,2])
objCylinderLeftTJ = cp.deepcopy(objCylinder)
objCylinderLeftTJ.TranslateGrain(arrCylinderLeftTJ)
objCylinderMiddleTJ = cp.deepcopy(objCylinder)
objCylinderMiddleTJ.TranslateGrain(arrCylinderMiddleTJ)
objLeftHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderLeftTJ))
objRightHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderLeftTJ))
objLeftHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderRightTJ))
objRightHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderRightTJ))
objLeftHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderMiddleTJ))
objRightHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderMiddleTJ))
objSimulationCellTJ.AddGrain(objLeftHalfChoppedTJ)
objSimulationCellTJ.AddGrain(objRightHalfChoppedTJ)
objSimulationCellTJ.AddGrain(objCylinderLeftTJ)
objSimulationCellTJ.AddGrain(objCylinderMiddleTJ)
objSimulationCellTJ.RemoveGrainPeriodicDuplicates()


for j in range(intIncrements):
    fltDistance = objFullLeft.GetNearestNeighbourDistance()*j/10
    objSimulationCellGB.MergeTooCloseAtoms(fltDistance,1,100)
    #objSimulationCellGB.WrapAllAtomsIntoSimulationCell()
    objSimulationCellGB.SetFileHeader('Grain centre is ' +str(arrCylinderCentreLeftGB))
    strFileNameGB = 'GB' + str(j)
    objSimulationCellGB.WriteLAMMPSDataFile(strDirectory + strFileNameGB + '.dat')
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', strFileNameGB + '.dat')
    fData = fData.replace('read.dmp', strFileNameGB + '.dmp')
    fData = fData.replace('read.lst', strFileNameGB + '.lst')
    fData = fData.replace('logfile', 'logGB' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateGB' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()
    objSimulationCellTJ.MergeTooCloseAtoms(fltDistance,1,100)
    #objSimulationCellTJ.WrapAllAtomsIntoSimulationCell()
    objSimulationCellTJ.SetFileHeader('Grain centre is ' +str(arrCylinderLeftTJ))
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
