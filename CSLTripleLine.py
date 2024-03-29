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

strDirectory = '/home/p17992pt/LAMMPSData/' #str(sys.argv[1])
strFilename = strDirectory + 'readTJ.dat'
intHeight = 2 #int(sys.argv[2]) #numbers of repeated CSL layers
fltMerge = 0.3# float(sys.argv[3])
lstAxis = [2,2,1] #eval(str(sys.argv[4]))
intSigma = 9#int(sys.argv[5])
arrAxis = np.array(lstAxis)
a = 4.05
t = np.pi/11
arrR = np.array([1,0,0])
arrUR = np.array([np.cos(np.pi/3),np.sin(np.pi/3),0])
arrUL = np.array([-np.cos(np.pi/3),np.sin(np.pi/3),0])
arrDL = -arrUR
arrDR = -arrUL
arrURC = np.array([np.cos(t),np.sin(np.pi/3),0])
arrULC = np.array([-np.cos(t),np.sin(np.pi/3),0])
arrSR = arrR+arrUR+arrULC-arrURC +arrDR

objSigma9 = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma9.MakeCSLCell(intSigma)
fltAngle1, fltAngle2 = objSigma9.GetLatticeRotations()
fltTurn = abs(fltAngle1) + abs(fltAngle2)
arrBasis = objSigma9.GetBasisVectors()

s0 = np.linalg.norm(arrBasis[0])
s1 = np.linalg.norm(arrBasis[1])
s2 = np.linalg.norm(arrBasis[2])

fltAngle, arrRotation = gf.FindRotationVectorAndAngle(arrAxis,np.array([0,0,1]))
arrBasisVectors = gf.RotateVectors(fltAngle, arrRotation,gf.StandardBasisVectors(3))
h = intHeight*a*s2
objHex1 = gl.ExtrudedRegularPolygon(10*s0*a,h,6,gf.RotateVectors(np.pi/2,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),np.zeros(3))
objHex2 = gl.ExtrudedRegularPolygon(10*s0*a,h,6,gf.RotateVectors(np.pi,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),10*a*s0*(arrDR+arrR))
objHex3 = gl.ExtrudedRegularPolygon(10*s0*a,h,6,gf.RotateVectors(0,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),10*a*s0*(arrUR+arrR))

lstBases = []
lstBases.append(gf.RotateVectors(np.pi/2,np.array([0,0,1]),arrBasisVectors))
lstBases.append(gf.RotateVectors(np.pi,np.array([0,0,1]),arrBasisVectors))
lstBases.append(gf.RotateVectors(np.pi/2,np.array([0,0,1]),arrBasisVectors))



gf.TripleLineTensor(np.array([0,0,1]),lstBases, [0,2*np.pi/3,4*np.pi/3])


# objHex1 = gl.IrrregularExtrudedGrain(20*a*np.array([arrR, arrURC, arrULC,-arrR,-arrURC,-arrULC]),h,gf.RotateVectors(0,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),np.zeros(3))
# objHex2 = gl.IrrregularExtrudedGrain(20*a*np.array([arrR, arrUR, arrULC,-arrSR,-arrURC,arrDR]),h,gf.RotateVectors(2*np.pi/3,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),20*a*(arrDR+arrR))
# objHex3 = gl.IrrregularExtrudedGrain(20*a*np.array([arrSR, arrURC, arrUL,-arrR,arrDL,-arrULC]),h,gf.RotateVectors(4*np.pi/3,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),20*a*(arrURC+arrR))
objHex1.SetPeriodicity(['n','n','p'])
objHex2.SetPeriodicity(['n','n','p'])
objHex3.SetPeriodicity(['n','n','p'])

objSimulationCell = gl.SimulationCell(np.array([30*a*s0*arrR,30*a*s0*arrUR,h*np.array([0,0,1])]))
objSimulationCell.AddGrain(objHex1)
objSimulationCell.AddGrain(objHex2)
objSimulationCell.AddGrain(objHex3)
objSimulationCell.MergeTooCloseAtoms(fltMerge*objHex1.GetNearestNeighbourDistance(),1)
objSimulationCell.WriteLAMMPSDataFile(strFilename)
# fIn = open(strDirectory +  'TemplateMobTJ.in', 'rt')
# fData = fIn.read()
# fData = fData.replace('read.dat', strFilename)
# fData = fData.replace('read.dmp', strFilename[:-3] + 'dmp')
# fData = fData.replace('read.lst', strFilename[:-3] + 'lst')
# fData = fData.replace('logfile', strFilename[:-3] + 'log')
# fIn.close()
# fIn = open(strDirectory + 'TemplateMobTJ.in', 'wt')
# fIn.write(fData)
# fIn.close()
