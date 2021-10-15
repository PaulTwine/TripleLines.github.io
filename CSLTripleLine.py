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


strDirectory = str(sys.argv[1])
strFilename = strDirectory + 'ReadTJ9.dat'
fltMerge = float(sys.argv[3])

intHeight = int(sys.argv[2])
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

objSigma9 = gl.SigmaCell(np.array([5,1,1]),ld.FCCCell)
objSigma9.MakeCSLCell(9)
arrBasis = objSigma9.GetBasisVectors()

s = np.linalg.norm(arrBasis[0])
fltAngle, arrRotation = gf.FindRotationVectorAndAngle(np.array([5,1,1]),np.array([0,0,1]))
arrBasisVectors = gf.RotateVectors(fltAngle, arrRotation,gf.StandardBasisVectors(3))
h = intHeight*a*np.linalg.norm(np.array([5,1,1]))
objHex1 = gl.ExtrudedRegularPolygon(10*s*a,h,6,gf.RotateVectors(0,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),np.zeros(3))
objHex2 = gl.ExtrudedRegularPolygon(10*s*a,h,6,gf.RotateVectors(2*np.pi/3,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),10*a*s*(arrDR+arrR))
objHex3 = gl.ExtrudedRegularPolygon(10*s*a,h,6,gf.RotateVectors(4*np.pi/3,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),10*a*s*(arrUR+arrR))


# objHex1 = gl.IrrregularExtrudedGrain(20*a*np.array([arrR, arrURC, arrULC,-arrR,-arrURC,-arrULC]),h,gf.RotateVectors(0,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),np.zeros(3))
# objHex2 = gl.IrrregularExtrudedGrain(20*a*np.array([arrR, arrUR, arrULC,-arrSR,-arrURC,arrDR]),h,gf.RotateVectors(2*np.pi/3,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),20*a*(arrDR+arrR))
# objHex3 = gl.IrrregularExtrudedGrain(20*a*np.array([arrSR, arrURC, arrUL,-arrR,arrDL,-arrULC]),h,gf.RotateVectors(4*np.pi/3,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),20*a*(arrURC+arrR))
objHex1.SetPeriodicity(['n','n','p'])
objHex2.SetPeriodicity(['n','n','p'])
objHex3.SetPeriodicity(['n','n','p'])

objSimulationCell = gl.SimulationCell(np.array([30*a*s*arrR,30*a*s*arrUR,h*np.array([0,0,1])]))
objSimulationCell.AddGrain(objHex1)
objSimulationCell.AddGrain(objHex2)
objSimulationCell.AddGrain(objHex3)
objSimulationCell.MergeTooCloseAtoms(fltMerge*objHex1.GetNearestNeighbourDistance(),1)
objSimulationCell.WriteLAMMPSDataFile(strFilename)
fIn = open(strDirectory +  'TemplateMobTJ.in', 'rt')
fData = fIn.read()
fData = fData.replace('read.dat', strFilename)
fData = fData.replace('read.dmp', strFilename[:-3] + 'dmp')
fData = fData.replace('read.lst', strFilename[:-3] + 'lst')
fData = fData.replace('logfile', strFilename[:-3] + 'log')
fIn.close()
fIn = open(strDirectory + 'TemplateMobTJ.in', 'wt')
fIn.write(fData)
fIn.close()
