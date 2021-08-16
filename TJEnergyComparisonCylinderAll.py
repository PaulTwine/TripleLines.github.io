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
intMax = 60
#i = 3 #scaling parameter
s1 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
s2 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
s3 = np.linalg.norm(arrSigmaBasis, axis=1)[2]
#i = np.sqrt(np.abs(np.dot(np.cross(arrSigmaBasis[0],arrSigmaBasis[1]),arrSigmaBasis[2])))
fltAreaFactor = np.sqrt(intSigma/s3)
i = np.max([1,np.round(10/fltAreaFactor,0)]).astype('int')
intHeight = 5
a = 4.05 ##lattice parameter
r = 2*a*s2*i
###First part runs with displaced cylinder
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
objSimulationCellGB = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCellCentreGB = objSimulationCellGB.GetCentre()
objFullCell1 = gl.ExtrudedParallelogram(arrX,arrXY,s3*h, gf.RotateVectors(fltAngle1,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell2 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(fltAngle2,z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullCell3 = gl.ExtrudedParallelogram(arrX,arrXY, s3*h, gf.RotateVectors(np.mean([fltAngle1,fltAngle2]),z,arrBasisVectors), ld.FCCCell, arrLatticeParameters,arrShift)
objFullCell1.SetPeriodicity(['n','p','p'])
objFullCell2.SetPeriodicity(['n','p','p'])
objFullCell3.SetPeriodicity(['n','n','n'])
strConstraint = str(arrXY[0])+ '*(y -' + str(arrCellCentreGB[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentreGB[0]) + ')' 
objLeftCell1 = cp.deepcopy(objFullCell1)
objLeftCell1.ApplyGeneralConstraint(gf.InvertRegion(strConstraint))
objRightCell2 = cp.deepcopy(objFullCell2)
objRightCell2.ApplyGeneralConstraint(strConstraint)

fltDistance = 0.3*objFullCell1.GetNearestNeighbourDistance()

arrGrainCentreGB = 5*a*i*(arrSigmaBasis[0] +arrSigmaBasis[1])+arrShift

np.savetxt(strDirectory + 'GrainCentreGB.txt',arrGrainCentreGB)

strCylinderGB = gf.ParseConic([arrGrainCentreGB[0],arrGrainCentreGB[1]],[r,r],[2,2])
objCylinderGB = cp.deepcopy(objFullCell3)
objCylinderGB.ApplyGeneralConstraint(strCylinderGB)
objLeftChoppedGB = cp.deepcopy(objLeftCell1)
objLeftChoppedGB.ApplyGeneralConstraint(gf.InvertRegion(strCylinderGB))
objRightChoppedGB = cp.deepcopy(objRightCell2)
objRightChoppedGB.ApplyGeneralConstraint(gf.InvertRegion(strCylinderGB))
objSimulationCellGB.AddGrain(objLeftChoppedGB)
objSimulationCellGB.AddGrain(objRightChoppedGB)
objSimulationCellGB.AddGrain(objCylinderGB)
objSimulationCellGB.RemoveGrainPeriodicDuplicates()


arrGrainCentreTJ = 3*a*i*arrSigmaBasis[0] + arrGrainCentreGB
np.savetxt(strDirectory + 'GrainCentreTJ.txt',arrGrainCentreTJ)
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



##Second part with triple lines


arrGrainCentreTJ = 3*a*i*arrSigmaBasis[0] + arrGrainCentreGB
np.savetxt(strDirectory + 'GrainCentre1.txt',arrGrainCentreTJ)
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
    objSimulationCellGB.MergeTooCloseAtoms(fltDistance,1)
    objSimulationCellGB.WrapAllAtomsIntoSimulationCell()
    objSimulationCellGB.SetFileHeader('Grain centre is ' +str(arrGrainCentreGB))
    strFileNameGB = 'readGB' + str(j)
    objSimulationCellGB.WriteLAMMPSDataFile(strDirectory + strFileNameGB + '.dat')
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', strFileNameGB + '.dat')
    fData = fData.replace('read.dmp', strFileNameGB + '.dmp')
    fData = fData.replace('read.lst', strFileNameGB + '.lst')
    fData = fData.replace('logfile', 'logfileGB' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateGB' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()
    objSimulationCellTJ.MergeTooCloseAtoms(fltDistance,1)
    objSimulationCellTJ.WrapAllAtomsIntoSimulationCell()
    objSimulationCellTJ.SetFileHeader('Grain centre is ' +str(arrGrainCentreTJ))
    strFileNameTJ = 'readTJ' + str(j)
    objSimulationCellTJ.WriteLAMMPSDataFile(strDirectory + strFileNameTJ + '.dat')
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', strFileNameTJ + '.dat')
    fData = fData.replace('read.dmp', strFileNameTJ + '.dmp')
    fData = fData.replace('read.lst', strFileNameTJ + '.lst')
    fData = fData.replace('logfile', 'logfileTJ' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateTJ' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()
