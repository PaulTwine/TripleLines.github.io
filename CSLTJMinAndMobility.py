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

###in LAMMPS two orient/eco fixes mean grain 1 is pensalied with a +u0 pe per atom and grains 2 and 3 are both energetically favoured by -u0/2 per atom. 


strRoot = '\home' #str(sys.argv[1])
intHeight = 1 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = [1,0,1] # eval(str(sys.argv[2]))
lstSigmaAxis = [3,9,27] # eval(str(sys.argv[3]))
intTemp = 450 # int(sys.argv[4])
u0 = 1 # float(sys.argv[5])
arrAxis = np.array(lstAxis)
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intRuns = 10**5
fltTolerance = 0.6
a = 4.05
lstOrientGB = [u0,0.25,a]
lstOrientTJ = [np.round(2*u0/3,5),0.25,a] 
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==lstSigmaAxis,axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJBasisVectors(intIndex,True)
arrBasis = a*objCSL.GetSimulationCellBasis()
arrMatrix = objCSL.GetRotationMatrix()
intTJSigma = objCSL.GetTJSigmaValue(arrCSL)
intRatio = np.round(np.linalg.norm(arrBasis[0])/np.linalg.norm(arrBasis[1]))

intNumberOfAtoms = 5*10**5 #choose approximate numbers of atoms here
intAtomsPerVolume = 4 # 4 for FCC and 2 for BCC
#s = np.round(np.sqrt(intNumberOfAtoms/(intAtomsPerCell*intTJSigma))/4) #
fltVolumeScale = np.linalg.det(arrBasis)*4.05**(-3)
y = np.round(np.sqrt(intNumberOfAtoms/(4*intRatio*fltVolumeScale*intAtomsPerVolume)))

x = np.round(y/intRatio)

arrGrainBasis1 = np.round(objCSL.GetLatticeBasis(1),10)
arrGrainBasis2 = np.round(objCSL.GetLatticeBasis(0),10)
arrGrainBasis3 = np.round(objCSL.GetLatticeBasis(2),10)

arrFullCell = np.array([[2*x,0,0],[0,2*y,0],[0,0,intHeight]])
arrSmallCell = np.array([[x,0,0],[0,y,0],[0,0,intHeight]])
arrHorizontalCell = np.array([[2*x,0,0],[0,y,0],[0,0,intHeight]])
arrVerticalCell = np.array([[x,0,0],[0,2*y,0],[0,0,intHeight]])

arrFullBox = np.round(np.matmul(arrFullCell,arrBasis),10)
arrSmallBox = np.round(np.matmul(arrSmallCell,arrBasis),10)
arrHorizontalBox = np.round(np.matmul(arrHorizontalCell,arrBasis),10)
arrVerticalBox = np.round(np.matmul(arrVerticalCell, arrBasis),10)

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
MiscFunctions.WriteTJDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrientTJ, ['Values12.ori','Values13.ori'])
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
MiscFunctions.WriteGBDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrientGB, 'Values12.ori')
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
MiscFunctions.WriteGBDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns,lstOrientGB, 'Values13.ori')
arrOrientBases = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis1), np.matmul(a*ld.FCCPrimitive,arrGrainBasis3), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values13.ori', arrOrientBases, delimiter=' ',fmt='%1.5f')

objSimulationCell = gl.SimulationCell(arrHorizontalBox)
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrHorizontalBox[0])
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = '32BH'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' +  strFilename +  '.dat')
arrOrientBases = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis3), np.matmul(a*ld.FCCPrimitive,arrGrainBasis2), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values32.ori', arrOrientBases, delimiter=' ',fmt='%1.5f')
MiscFunctions.WriteGBDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrientGB, 'Values32.ori')


