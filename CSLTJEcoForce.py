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

###in LAMMPS two orient/eco fixes with compensation for partial
# lattice correlations. Values ensure no driving pressure term 
# between grains 1 and 2. 


strRoot = '' #str(sys.argv[1])
intHeight = 1 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = [2,2,1] # eval(str(sys.argv[2]))
lstSigmaAxis = [9,9,9] #eval(str(sys.argv[3]))
intTemp = 500 # int(sys.argv[4])
u0 = 0.08 #float(sys.argv[5])
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
intRatio = np.round(np.linalg.norm(arrBasis[0])/np.linalg.norm(arrBasis[1]))


arrGrainBasis1 = np.round(objCSL.GetLatticeBasis(1),10)
arrGrainBasis2 = np.round(objCSL.GetLatticeBasis(0),10)
arrGrainBasis3 = np.round(objCSL.GetLatticeBasis(2),10)

arrMiniGrain1 = gl.ExtrudedRectangle(2*a,2*a,2*a, arrGrainBasis1, ld.FCCCell,
a*np.ones(3), np.zeros(3))
arrMiniGrain2 = gl.ExtrudedRectangle(2*a,2*a,2*a, arrGrainBasis2, ld.FCCCell,
np.array([a,a,a]),np.zeros(3))
arrMiniGrain3 = gl.ExtrudedRectangle(2*a,2*a,2*a, arrGrainBasis3, ld.FCCCell,
np.array([a,a,a]),np.zeros(3))

arrPBasis1 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis1)
arrPBasis2 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis2)
arrPBasis3 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis3)

arrTBasis1 = np.matmul(arrMiniGrain1.GetRealPoints()-gf.FindMediod(arrMiniGrain1.GetRealPoints()), np.linalg.inv(ld.FCCPrimitive))
arrTBasis2 = np.matmul(arrMiniGrain2.GetRealPoints()-gf.FindMediod(arrMiniGrain2.GetRealPoints()), np.linalg.inv(ld.FCCPrimitive))
arrTBasis3 = np.matmul(arrMiniGrain3.GetRealPoints()-gf.FindMediod(arrMiniGrain3.GetRealPoints()), np.linalg.inv(ld.FCCPrimitive))



# arrPBasis1 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis1)
# arrPBasis2 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis2)
# arrPBasis3 = np.matmul(a*ld.FCCPrimitive,arrGrainBasis3)

objOrient = gf.EcoOrient(4.05,0.25)
#fltpEqr means grain q is to be penalised, grain r is to be favoured and the test orientation is p

flt1E12 = objOrient.GetOrderParameter(gf.GetLinearCombinations(arrPBasis1,4), arrPBasis1,arrPBasis2)
flt2E12 = objOrient.GetOrderParameter(arrPBasis2, arrPBasis1,arrPBasis2)
flt3E12 =  objOrient.GetOrderParameter(arrPBasis3, arrPBasis1,arrPBasis2)


flt1E13 = objOrient.GetOrderParameter(arrTBasis1, arrPBasis1,arrPBasis3)
flt2E13 = objOrient.GetOrderParameter(arrTBasis2, arrPBasis1,arrPBasis3)
flt3E13 =  objOrient.GetOrderParameter(arrTBasis3, arrPBasis1,arrPBasis3)



# flt1E12 = objOrient.GetOrderParameter(arrPBasis1, arrPBasis1,arrPBasis2)
# flt2E12 = objOrient.GetOrderParameter(arrPBasis2, arrPBasis1,arrPBasis2)
# flt3E12 =  objOrient.GetOrderParameter(arrPBasis3, arrPBasis1,arrPBasis2)


# flt1E13 = objOrient.GetOrderParameter(arrPBasis1, arrPBasis1,arrPBasis3)
# flt2E13 = objOrient.GetOrderParameter(arrPBasis2, arrPBasis1,arrPBasis3)
# flt3E13 =  objOrient.GetOrderParameter(arrPBasis3, arrPBasis1,arrPBasis3)

#print(flt1E13,flt2E13,flt3E13,flt1E12,flt2E12,flt3E12)
r = (flt2E12-flt3E12)/(flt3E13-flt2E13) #use exact values rather than assuming +1, -1 values 
u1 = u0/(flt1E12 +r*flt1E13-flt2E12 - r*flt2E13)
u2 = u1*r

lstOrientTJ1 = [np.round(u1,10),0.25,a] 
lstOrientTJ2 = [np.round(u2,10),0.25,a] 

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
MiscFunctions.WriteDoubleDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrientTJ1,lstOrientTJ2, ['Values12.ori','Values13.ori'])
arrOrientBases12 = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis1), np.matmul(a*ld.FCCPrimitive,arrGrainBasis2), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values12.ori', arrOrientBases12, delimiter=' ',fmt='%1.5f')
arrOrientBases13 = np.round(np.append(np.matmul(a*ld.FCCPrimitive,arrGrainBasis1), np.matmul(a*ld.FCCPrimitive,arrGrainBasis3), axis=0),7)
np.savetxt(strRoot + strFilename + '/Values13.ori', arrOrientBases13, delimiter=' ',fmt='%1.5f')


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
MiscFunctions.WriteDoubleDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrientTJ1,lstOrientTJ2, ['Values12.ori','Values13.ori'])
np.savetxt(strRoot + strFilename + '/Values12.ori', arrOrientBases12, delimiter=' ',fmt='%1.5f')
np.savetxt(strRoot + strFilename + '/Values13.ori', arrOrientBases13, delimiter=' ',fmt='%1.5f')

objSimulationCell.RemoveAllGrains()
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), 0.5*arrVerticalBox[1])
strFilename = '13BV'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' + strFilename + '.dat')
MiscFunctions.WriteDoubleDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrientTJ1,lstOrientTJ2, ['Values12.ori','Values13.ori'])
np.savetxt(strRoot + strFilename + '/Values12.ori', arrOrientBases12, delimiter=' ',fmt='%1.5f')
np.savetxt(strRoot + strFilename + '/Values13.ori', arrOrientBases13, delimiter=' ',fmt='%1.5f')

objSimulationCell = gl.SimulationCell(arrHorizontalBox)
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrHorizontalBox[0])
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = '32BH'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename + '/' +  strFilename +  '.dat')
np.savetxt(strRoot + strFilename + '/Values12.ori', arrOrientBases12, delimiter=' ',fmt='%1.5f')
np.savetxt(strRoot + strFilename + '/Values13.ori', arrOrientBases13, delimiter=' ',fmt='%1.5f')
MiscFunctions.WriteDoubleDrivenTemplate(strRoot + strFilename + '/', strFilename, intTemp, intRuns, lstOrientTJ1,lstOrientTJ2, ['Values12.ori','Values13.ori'])

