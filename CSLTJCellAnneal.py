import numpy as np
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
import copy as cp
import MiscFunctions


strDirectory = '/home/p17992pt/LAMMPSData/' #str(sys.argv[1])
intHeight = 1 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = [1,0,1] # eval(str(sys.argv[2]))
lstSigmaAxis = [3,3,9] #eval(str(sys.argv[3]))
intTemp = 650 #int(sys.argv[4])
intRuns = 400000

arrAxis = np.array(lstAxis)
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intIncrements = 10
fltTolerance = 0.5
a = 4.05
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==lstSigmaAxis,axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,True)
arrBasis = a*objCSL.GetSimulationCellBasis()
arrMatrix = objCSL.GetRotationMatrix()
intTJSigma = objCSL.GetTJSigmaValue(arrCSL)

s = np.round(np.sqrt(4*10**5/(intHeight*np.linalg.det(arrBasis))))
intRound = 6
arrGrainBasis1 = np.round(objCSL.GetLatticeBasis(0),intRound) 
arrGrainBasis2 = np.round(objCSL.GetLatticeBasis(2),intRound)
arrGrainBasis3 = np.round(objCSL.GetLatticeBasis(1),intRound)

arrFullCell = np.array([[4*s,0,0],[0,4*s,0],[0,0,intHeight]])
arrSmallCell = np.array([[2*s,0,0],[0,2*s,0],[0,0,intHeight]])
arrHorizontalCell = np.array([[4*s,0,0],[0,2*s,0],[0,0,intHeight]])
arrVerticalCell = np.array([[2*s,0,0],[0,4*s,0],[0,0,intHeight]])

arrFullBox = np.round(np.matmul(arrFullCell,arrBasis),intRound)
arrSmallBox = np.round(np.matmul(arrSmallCell,arrBasis),intRound)
arrHorizontalBox = np.round(np.matmul(arrHorizontalCell,arrBasis),intRound)
arrVerticalBox = np.round(np.matmul(arrVerticalCell, arrBasis),intRound)

objSimulationCell = gl.SimulationCell(arrFullBox)

arrGrain1 = gl.ParallelopiedGrain(arrHorizontalBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrFullBox[0]+0.5*arrFullBox[1])
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3),0.5*arrFullBox[1])

fltNearestNeighbour = arrGrain1.GetNearestNeighbourDistance()
fltE = fltTolerance*fltNearestNeighbour
strFilename = 'TJ'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)

objSimulationCell.MergeTooCloseAtoms(fltE,1)
objSimulationCell.WriteLAMMPSDataFile(strDirectory + strFilename + '.dat')
MiscFunctions.WriteAnnealTemplate(strDirectory,strFilename, intTemp, intRuns)

