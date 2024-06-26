import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
#from mpl_toolkits.mplot3d import Axes3D 
import copy as cp
from scipy import spatial
#fig = plt.figure(figsize=plt.figaspect(1)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
#ax = fig.gca(projection='3d')

strDirectory = str(sys.argv[1])
intSigma = int(sys.argv[2])
lstAxis = eval(str(sys.argv[3]))
intIncrements = 10 # int(sys.argv[4])
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

lstLattices = objSigma.GetLatticeBases()
arrBasis1 = lstLattices[0]
arrBasis2 = lstLattices[1]

w = 32*a*i
l = 12*a*i

arrX = w*arrSigmaBasis[0]
arrXY = l*arrSigmaBasis[1]
z = h*arrSigmaBasis[2]

objSimulationCellGB = gl.SimulationCell(np.array([arrX,arrXY, z])) 
arrCellCentreGB = objSimulationCellGB.GetCentre()
objFullLeft = gl.ExtrudedParallelogram(arrX,arrXY,s2*h, arrBasis1, ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullRight = gl.ExtrudedParallelogram(arrX,arrXY, s2*h, arrBasis2, ld.FCCCell, arrLatticeParameters,np.zeros(3))
objFullLeft.SetPeriodicity(['n','p','p'])
objFullRight.SetPeriodicity(['n','p','p'])
strConstraintGB = str(arrXY[0])+ '*(y -' + str(arrCellCentreGB[1]) + ') - ' + str(arrXY[1]) + '*(x -' + str(arrCellCentreGB[0]) + ')' 
objLeftCell = cp.deepcopy(objFullLeft)
objLeftCell.ApplyGeneralConstraint(gf.InvertRegion(strConstraintGB))
objRightCell = cp.deepcopy(objFullRight)
objRightCell.ApplyGeneralConstraint(strConstraintGB)

objSimulationCellGB.AddGrain(objLeftCell)
objSimulationCellGB.AddGrain(objRightCell)
objSimulationCellGB.RemoveGrainPeriodicDuplicates()

for j in range(intIncrements):
    fltDistance = objFullLeft.GetNearestNeighbourDistance()*j/10
    objSimulationCellGB.MergeTooCloseAtoms(fltDistance,1,100)
    strFileNameGB = 'GB' + str(j)
    objSimulationCellGB.WriteLAMMPSDataFile(strDirectory + strFileNameGB + '.dat')
    fIn = open(strDirectory +  'TemplateMin.in', 'rt')
    fData = fIn.read()
    fData = fData.replace('read.dat', strFileNameGB + '.dat')
    fData = fData.replace('read.dmp', strFileNameGB + '.dmp')
    fData = fData.replace('read.lst', strFileNameGB + '.lst')
    fData = fData.replace('logfile', 'logCSLGB' + str(j))
    fIn.close()
    fIn = open(strDirectory + 'TemplateGB' + str(j) + '.in', 'wt')
    fIn.write(fData)
    fIn.close()
    