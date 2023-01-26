import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
import copy as cp
from scipy import spatial
import MiscFunctions as mf
from mpl_toolkits.mplot3d import Axes3D

 
strRoot = str(sys.argv[1])
intHeight = 1 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = eval(str(sys.argv[2]))
lstSigmaAxis = eval(str(sys.argv[3]))
arrAxis = np.array(lstAxis)
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intRuns = 10**5
fltTolerance = 0.6
a = 4.05 
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==lstSigmaAxis,axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJBasisVectors(intIndex,True)
arrBasis = a*objCSL.GetSimulationCellBasis()
arrMatrix = objCSL.GetRotationMatrix()
intTJSigma = objCSL.GetTJSigmaValue(arrCSL)

arrGrainBasis1 = np.round(objCSL.GetLatticeBasis(1),10)
arrGrainBasis2 = np.round(objCSL.GetLatticeBasis(0),10)
arrGrainBasis3 = np.round(objCSL.GetLatticeBasis(2),10)
arrPBasis1 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis1)
arrPBasis2 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis2)
arrPBasis3 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis3)
lstGrainBases = [arrGrainBasis1,arrGrainBasis2,arrGrainBasis3]
objSimulationCell = gl.SimulationCell(arrBasis)
i= 1
arrOrientBases12 = np.round(np.append(arrPBasis1, arrPBasis2, axis=0),7)
arrOrientBases13 = np.round(np.append(arrPBasis1, arrPBasis3, axis=0),7)
np.savetxt(strRoot + 'Values12.ori', arrOrientBases12, delimiter=' ',fmt='%1.5f')
np.savetxt(strRoot +  'Values13.ori', arrOrientBases13, delimiter=' ',fmt='%1.5f')
for l in lstGrainBases:
    arrGrain = gl.ParallelopiedGrain(arrBasis,l,ld.FCCCell,a*np.ones(3), np.zeros(3))
    objSimulationCell.AddGrain(arrGrain)
    objSimulationCell.RemoveAtomsOnOpenBoundaries()
    objSimulationCell.WriteLAMMPSDataFile(strRoot + str(i) + '.dat' )
    objSimulationCell.RemoveAllGrains()
    mf.WriteDoubleDrivenTemplate(strRoot, str(i),0,100,[0,0.25,a],[0,0.25,a],['Values12.ori','Values13.ori'])
    i +=1

