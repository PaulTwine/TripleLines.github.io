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
import MiscFunctions
from mpl_toolkits.mplot3d import Axes3D



strRoot = str(sys.argv[1])
intHeight = 1 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = eval(str(sys.argv[2]))
lstSigmaAxis = eval(str(sys.argv[3]))
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

arrReciprocal =  gf.FindReciprocalVectors(np.round(arrBasis,10))

print(np.matmul(np.round(arrBasis,10),arrReciprocal))

s = np.round(np.sqrt(10**5/(4*intTJSigma)))

arrGrainBasis1 = objCSL.GetLatticeBasis(1)
arrGrainBasis2 = objCSL.GetLatticeBasis(0)
arrGrainBasis3 = objCSL.GetLatticeBasis(2)

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

strFilename = 'TJ.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)


objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename[:-4] + '.dat')
MiscFunctions.WriteMinTemplate(strRoot,strFilename[:-4])



objSimulationCell = gl.SimulationCell(arrSmallBox)
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = '1G.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename)
#lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
#MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strRoot + strTemplateName,  strRoot +'Template' + strFilename[:-3] + 'in')
MiscFunctions.WriteMinTemplate(strRoot,strFilename[:-4])
objSimulationCell.RemoveAllGrains()
strFilename = '2G.dat'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename)
#lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
#MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strRoot + strTemplateName,  strRoot +'Template' + strFilename[:-3] + 'in')
MiscFunctions.WriteMinTemplate(strRoot,strFilename[:-4])

objSimulationCell = gl.SimulationCell(arrVerticalBox)
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrVerticalBox[1])
strFilename = '12BV.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot + strFilename)
#lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
#MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strRoot + strTemplateName,  strRoot +'Template' + strFilename[:-3] + 'in')
MiscFunctions.WriteMinTemplate(strRoot,strFilename[:-4])

objSimulationCell.RemoveAllGrains()
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), 0.5*arrVerticalBox[1])
strFilename = '13BV.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot+ strFilename)
#lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
#MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strRoot + strTemplateName,  strRoot +'Template' + strFilename[:-3] + 'in')
MiscFunctions.WriteMinTemplate(strRoot,strFilename[:-4])


objSimulationCell = gl.SimulationCell(arrHorizontalBox)
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrHorizontalBox[0])
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = '32BH.dat'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strRoot+ strFilename)
lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
#MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strRoot + strTemplateName,  strRoot +'Template' + strFilename[:-3] + 'in')
MiscFunctions.WriteMinTemplate(strRoot,strFilename[:-4])


