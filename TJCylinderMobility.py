import numpy as np
#import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
#from mpl_toolkits.mplot3d import Axes3D 
import copy as cp
#from scipy import spatial
# fig = plt.figure(figsize=plt.figaspect(1)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
# ax = fig.gca(projection='3d')

strDirectory = str(sys.argv[1])
intSigma = int(sys.argv[2])
lstAxis = eval(str(sys.argv[3]))
arrAxis = np.array(lstAxis)
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma, True)
arrSigmaBasis = objSigma.GetBasisVectors()
print(arrSigmaBasis, np.linalg.det(arrSigmaBasis), gf.FindReciprocalVectors(arrSigmaBasis))
s0 = np.linalg.norm(arrSigmaBasis, axis=1)[0]
s1 = np.linalg.norm(arrSigmaBasis, axis=1)[1]
s2 = np.linalg.norm(arrSigmaBasis, axis=1)[2]

intHeight = 4
intAtoms = 2.5*10**5
intAtomsPerCell = 4
a = 4.05 ##lattice parameter
h = a*np.round(intHeight/s2,0)

i = np.sqrt(intAtoms*a/(32*12*intAtomsPerCell*h*np.linalg.det(arrSigmaBasis)))
i = np.round(i,0).astype('int')
r=4*a*i
arrLatticeParameters= np.array([a,a,a])

arrMedianLattice = objSigma.GetMedianLattice()
lstLattices = objSigma.GetLatticeBases()
arrBasis1 = lstLattices[0]
arrBasis2 = lstLattices[1]

objCylinder = gl.ExtrudedCylinder(r,h*s2,arrMedianLattice,ld.FCCCell,arrLatticeParameters,np.zeros(3))
objCylinder.SetPeriodicity(['n','n','p'])


##Second part with triple lines
w = 32*a*i
l = 16*a*i

h = a*np.round(intHeight/s2,0)
arrXTJ = w*arrSigmaBasis[0]
arrXYTJ = l*arrSigmaBasis[1]

arrOrientBases = np.round(np.append(np.matmul(ld.FCCPrimitive,arrBasis1), np.matmul(ld.FCCPrimitive,arrBasis2), axis=0),7)
np.savetxt(strDirectory + 'Values.ori', arrOrientBases, delimiter=' ')

objSimulationCellTJ = gl.SimulationCell(np.array([arrXTJ,arrXYTJ, s2*h*np.array([0,0,1])])) 
arrCellCentreTJ = objSimulationCellTJ.GetCentre()
arrCylinderMiddleTJ = 0.5*(arrXYTJ+arrXTJ) 
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


strCylinderMiddleTJ = gf.ParseConic([arrCylinderMiddleTJ[0],arrCylinderMiddleTJ[1]],[r,r],[2,2])
objCylinderMiddleTJ = cp.deepcopy(objCylinder)
objCylinderMiddleTJ.TranslateGrain(arrCylinderMiddleTJ)
objLeftHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderMiddleTJ))
objRightHalfChoppedTJ.ApplyGeneralConstraint(gf.InvertRegion(strCylinderMiddleTJ))
objSimulationCellTJ.AddGrain(objLeftHalfChoppedTJ)
objSimulationCellTJ.AddGrain(objRightHalfChoppedTJ)
objSimulationCellTJ.AddGrain(objCylinderMiddleTJ)
#objSimulationCellTJ.RemoveGrainPeriodicDuplicates()



fltDistance = objFullLeft.GetNearestNeighbourDistance()*5/10
objSimulationCellTJ.MergeTooCloseAtoms(fltDistance,1,100)
#objSimulationCellTJ.WrapAllAtomsIntoSimulationCell()
objSimulationCellTJ.SetFileHeader('Grain centre is ' +str(arrCylinderMiddleTJ))
strFileNameTJ = 'TJ' 
objSimulationCellTJ.WriteLAMMPSDataFile(strDirectory + strFileNameTJ + '.dat')
# fIn = open(strDirectory +  'TemplateMin.in', 'rt')
# fData = fIn.read()
# fData = fData.replace('read.dat', strFileNameTJ + '.dat')
# fData = fData.replace('read.dmp', strFileNameTJ + '.dmp')
# fData = fData.replace('read.lst', strFileNameTJ + '.lst')
# fData = fData.replace('logfile', 'logTJ' + str(j))
# fIn.close()
# fIn = open(strDirectory + 'TemplateTJ' + str(j) + '.in', 'wt')
# fIn.write(fData)
# fIn.close()
