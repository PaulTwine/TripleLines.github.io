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



strDirectory = '/home/p17992pt/LAMMPSData/' #str(sys.argv[1])
intHeight = 1 #int(sys.argv[2]) #numbers of repeated CSL layers
lstAxis = [2,2,1] #eval(str(sys.argv[2]))
lstSigmaAxis = [9,9,9] #eval(str(sys.argv[3]))
intTemp = 750 #int(sys.argv[4])

arrAxis = np.array(lstAxis)
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intIncrements = 10
fltTolerance = 0.1
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

s = np.round(np.sqrt(10**5/(np.linalg.det(arrBasis))))

arrGrainBasis1 = objCSL.GetLatticeBasis(1)
arrGrainBasis2 = objCSL.GetLatticeBasis(0)
arrGrainBasis3 = objCSL.GetLatticeBasis(2)

arrFullCell = np.array([[4*s,0,0],[0,4*s,0],[0,0,intHeight]])
arrSmallCell = np.array([[2*s,0,0],[0,2*s,0],[0,0,intHeight]])
arrHorizontalCell = np.array([[4*s,0,0],[0,2*s,0],[0,0,intHeight]])
arrVerticalCell = np.array([[2*s,0,0],[0,4*s,0],[0,0,intHeight]])

arrFullBox = np.matmul(arrFullCell,arrBasis)
arrSmallBox = np.matmul(arrSmallCell,arrBasis)
arrHorizontalBox = np.matmul(arrHorizontalCell,arrBasis)
arrVerticalBox = np.matmul(arrVerticalCell, arrBasis)

objSimulationCell = gl.SimulationCell(arrFullBox)

arrBaseVectors2 = np.array([0.5*arrHorizontalBox[0]+0.5*arrHorizontalBox[1],arrHorizontalBox[1],-0.5*arrHorizontalBox[0],-1.5*arrHorizontalBox[1]])
arrBaseVectors3 = np.array([0.5*arrHorizontalBox[0]-0.5*arrHorizontalBox[1],1.5*arrHorizontalBox[1],-0.5*arrHorizontalBox[0],-arrHorizontalBox[1]])

strPlane1 = gf.ParsePlane(-np.cross(arrHorizontalBox[0]-arrHorizontalBox[1],arrHorizontalBox[2]),arrHorizontalBox[1])
strPlane2 = gf.ParsePlane(-np.cross(arrHorizontalBox[0]+arrHorizontalBox[1],arrHorizontalBox[2]),+arrHorizontalBox[0] + arrHorizontalBox[1])
strDomain = gf.ParsePlane(-np.cross(arrHorizontalBox[1],arrHorizontalBox[2]),0.5*arrHorizontalBox[0])

arrGrain1 = gl.ParallelopiedGrain(arrHorizontalBox,arrGrainBasis1,ld.FCCCell,a*np.ones(3), np.zeros(3))
arrGrain1.ApplyGeneralConstraint(strPlane1,'[x,y,z]',1e-5,strDomain)
arrGrain1.ApplyGeneralConstraint(strPlane2,'[x,y,z]',1e-5,gf.InvertRegion(strDomain))
arrGrain2 = gl.IrregularSlantedGrain(arrBaseVectors2,arrHorizontalBox[2],arrGrainBasis2,ld.FCCCell,a*np.ones(3), 0.5*arrFullBox[0]+0.5*arrHorizontalBox[1])
arrGrain3 = gl.IrregularSlantedGrain(arrBaseVectors3,arrHorizontalBox[2],arrGrainBasis3,ld.FCCCell,a*np.ones(3), 0.5*arrFullBox[1])

fltNearestNeighbour = arrGrain1.GetNearestNeighbourDistance()
fltTolerance = 0.1*fltNearestNeighbour
strFilename = 'TJ'
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.AddGrain(arrGrain2)
objSimulationCell.AddGrain(arrGrain3)

objSimulationCell.MergeTooCloseAtoms(fltTolerance,1)
objSimulationCell.WriteLAMMPSDataFile(strDirectory + strFilename + '.dat')
MiscFunctions.WriteAnnealTemplate(strDirectory,strFilename, intTemp)

