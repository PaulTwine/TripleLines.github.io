import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
import copy as cp
from scipy import spatial
import MiscFunctions
from mpl_toolkits.mplot3d import Axes3D

##Currently Grain 1 is rectangular with nonad-justed PE. Grain 3 has higher PE (+0.04) and Grain 2 has lower PE (-0.04) 
# and Grain 1 is unaffected by fix orient eco.


strRoot = str(sys.argv[1])
intHeight = 1 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = eval(str(sys.argv[2]))
lstSigmaAxis = eval(str(sys.argv[3]))
intTemp = int(sys.argv[4])
arrAxis = np.array(lstAxis)
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intRuns = 5*10**4
fltTolerance = 0.5
a = 4.05
lstOrient = [0.06,0.25,4.05] 
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==lstSigmaAxis,axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,True)
arrBasis = a*objCSL.GetSimulationCellBasis()
arrMatrix = objCSL.GetRotationMatrix()
intTJSigma = objCSL.GetTJSigmaValue(arrCSL)

intNumberOfAtoms = 4*10**5
s = np.round(np.sqrt(intNumberOfAtoms/(4*intTJSigma)))

arrGrainBasis1 = np.round(objCSL.GetLatticeBasis(1),10)
arrGrainBasis2 = np.round(objCSL.GetLatticeBasis(0),10)
arrGrainBasis3 = np.round(objCSL.GetLatticeBasis(2),10)

arrFullCell = np.array([[2*s,0,0],[0,2*s,0],[0,0,intHeight]])
arrSmallCell = np.array([[s,0,0],[0,s,0],[0,0,intHeight]])
arrHorizontalCell = np.array([[2*s,0,0],[0,s,0],[0,0,intHeight]])
arrVerticalCell = np.array([[s,0,0],[0,2*s,0],[0,0,intHeight]])

arrFullBox = np.matmul(arrFullCell,arrBasis)
arrSmallBox = np.matmul(arrSmallCell,arrBasis)
arrHorizontalBox = np.matmul(arrHorizontalCell,arrBasis)
arrVerticalBox = np.matmul(arrVerticalCell, arrBasis)

objSimulationCell = gl.SimulationCell(arrFullBox)
arrGrain1 = gl.ParallelopiedGrain(arrHorizontalBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*(arrFullBox[0]+arrFullBox[1]))
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), 0.5*arrFullBox[1])

fltNearestNeighbour = arrGrain1.GetNearestNeighbourDistance()
fltTolerance = fltTolerance*fltNearestNeighbour

strFilename = 'TJ'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)


objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' + strFilename + '.dat')
MiscFunctions.WriteTJDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrient, ['Values12.ori', 'Values13.ori'] )
arrOrientBases = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis1), np.matmul(a*ld.FCCPrimitive,arrGrainBasis2), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values12.ori', arrOrientBases, delimiter=' ',fmt='%1.5f')
arrOrientBases = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis1), np.matmul(a*ld.FCCPrimitive,arrGrainBasis3), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values13.ori', arrOrientBases, delimiter=' ',fmt='%1.5f')


objSimulationCell = gl.SimulationCell(arrSmallBox)
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = '2G'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' + strFilename + '.dat')
MiscFunctions.WriteMinTemplate(strRoot + strFilename + '/',strFilename)
objSimulationCell.RemoveAllGrains()
strFilename = '3G'
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' + strFilename + '.dat')
MiscFunctions.WriteMinTemplate(strRoot + strFilename + '/',strFilename)

objSimulationCell = gl.SimulationCell(arrVerticalBox)
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrVerticalBox[1])
strFilename = '12BV'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/'  + strFilename + '.dat')
MiscFunctions.WriteGBDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrient, 'Values12.ori')
arrOrientBases = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis1), np.matmul(a*ld.FCCPrimitive,arrGrainBasis2), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values12.ori', arrOrientBases, delimiter=' ',fmt='%1.5f')

objSimulationCell.RemoveAllGrains()
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), 0.5*arrVerticalBox[1])
strFilename = '13BV'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' + strFilename + '.dat')
MiscFunctions.WriteGBDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns,lstOrient, 'Values13.ori')
arrOrientBases = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis1), np.matmul(a*ld.FCCPrimitive,arrGrainBasis3), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values12.ori', arrOrientBases, delimiter=' ',fmt='%1.5f')

objSimulationCell = gl.SimulationCell(arrHorizontalBox)
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrHorizontalBox[0])
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = '32BH'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' +  strFilename +  '.dat')
MiscFunctions.WriteGBDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrient, '')


