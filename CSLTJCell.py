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

strDirectory = str(sys.argv[1])
intHeight = 2 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = eval(str(sys.argv[2]))
arrAxis = np.array(lstAxis)
a = 4.05
lstOldTemplate = ['read.dat','read.dmp','read.lst', 'logfile']
strTemplateName = 'TemplateMin.in'
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
arrCSL = arrCell[0]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(0)
arrBasis = a*objCSL.GetSimulationCellBasis()
arrMatrix = objCSL.GetRotationMatrix()
intTJSigma = objCSL.GetTJSigmaValue(arrCSL)

s = np.round(np.sqrt(10**4/(8*intTJSigma)))

arrGrainBasis1 = objCSL.GetLatticeBasis(0)
arrGrainBasis2 = objCSL.GetLatticeBasis(1)
arrGrainBasis3 = objCSL.GetLatticeBasis(2)

arrFullCell = np.array([[4*s,0,0],[0,4*s,0],[0,0,2]])
arrSmallCell = np.array([[2*s,0,0],[0,2*s,0],[0,0,2]])
arrHorizontalCell = np.array([[4*s,0,0],[0,2*s,0],[0,0,2]])
arrVerticalCell = np.array([[2*s,0,0],[0,4*s,0],[0,0,2]])

arrFullBox = np.matmul(arrFullCell,arrBasis)
arrSmallBox = np.matmul(arrSmallCell,arrBasis)
arrHorizontalBox = np.matmul(arrHorizontalCell,arrBasis)
arrVerticalBox = np.matmul(arrVerticalCell, arrBasis)

objSimulationCell = gl.SimulationCell(arrFullBox)
arrGrain1 = gl.ParallelopiedGrain(arrHorizontalBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*(arrFullBox[0]+arrFullBox[1]))
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), 0.5*arrFullBox[1])

strFilename = 'TJ123.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(0.1,1)
objSimulationCell.WriteLAMMPSDataFile(strDirectory + strFilename)
lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strDirectory + strTemplateName,  strDirectory +'Template' + strFilename[:-3] + 'in')

objSimulationCell = gl.SimulationCell(arrSmallBox)
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = 'G1.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.WriteLAMMPSDataFile(strDirectory + strFilename)
lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strDirectory + strTemplateName,  strDirectory +'Template' + strFilename[:-3] + 'in')
objSimulationCell.RemoveAllGrains()
strFilename = 'G2.dat'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.WriteLAMMPSDataFile(strDirectory + strFilename)
lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strDirectory + strTemplateName,  strDirectory +'Template' + strFilename[:-3] + 'in')


objSimulationCell = gl.SimulationCell(arrHorizontalBox)
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), 0.5*arrHorizontalBox[0])
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = 'BH21.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.MergeTooCloseAtoms(0.1,1)
objSimulationCell.WriteLAMMPSDataFile(strDirectory + strFilename)
lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strDirectory + strTemplateName,  strDirectory +'Template' + strFilename[:-3] + 'in')


objSimulationCell = gl.SimulationCell(arrVerticalBox)
arrGrain1 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), 0.5*arrVerticalBox[1])
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = 'BV31.dat'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(0.1,1)
objSimulationCell.WriteLAMMPSDataFile(strDirectory+ strFilename)
lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strDirectory + strTemplateName,  strDirectory +'Template' + strFilename[:-3] + 'in')


objSimulationCell.RemoveAllGrains()
arrGrain2 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrVerticalBox[1])
arrGrain3 = gl.ParallelopiedGrain(arrSmallBox,arrGrainBasis3,ld.FCCCell,a*np.ones(3), np.zeros(3))
strFilename = 'BV32.dat'
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)
objSimulationCell.MergeTooCloseAtoms(0.1,1)
objSimulationCell.WriteLAMMPSDataFile(strDirectory+ strFilename)
lstNew = [strFilename, strFilename[:-3]+'dmp', strFilename[:-3]+'lst', strFilename[:-3] + 'log']
MiscFunctions.UpdateTemplate(lstOldTemplate,lstNew, strDirectory + strTemplateName,  strDirectory +'Template' + strFilename[:-3] + 'in')



