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


#strFilename = '/home/p17992pt/LAMMPSData/mobility.dat' #str(sys.argv[1])
strFilename = '/home/p17992pt/csf3_scratch/Axis001/TJSigma5/0/GB6.lst'
objData = LT.LAMMPSData(strFilename, 1, 4.05, LT.LAMMPSAnalysis3D)
objTimeStep = objData.GetTimeStepByIndex(-1)
objTimeStep.WriteDataFile('/home/p17992pt/csf3_scratch/Mobility/TwoCell/GB6Min.dat')
intHeight = 2 #int(sys.argv[2])
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

fltAngle, arrRotation = gf.FindRotationVectorAndAngle(np.array([1,1,1]),np.array([0,0,1]))
arrBasisVectors = gf.RotateVectors(fltAngle, arrRotation,gf.StandardBasisVectors(3))
h = intHeight*a*np.sqrt(3)

objHex1 = gl.IrrregularExtrudedGrain(20*a*np.array([arrR, arrURC, arrULC,-arrR,-arrURC,-arrULC]),h,gf.RotateVectors(0,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),np.zeros(3))
objHex2 = gl.IrrregularExtrudedGrain(20*a*np.array([arrR, arrUR, arrULC,-arrSR,-arrURC,arrDR]),h,gf.RotateVectors(np.pi/6,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),20*a*(arrDR+arrR))
objHex3 = gl.IrrregularExtrudedGrain(20*a*np.array([arrSR, arrURC, arrUL,-arrR,arrDL,-arrULC]),h,gf.RotateVectors(-np.pi/6,np.array([0,0,1]),arrBasisVectors),ld.FCCCell,a*np.ones(3),20*a*(arrURC+arrR))
objHex1.SetPeriodicity(['n','n','p'])
objHex2.SetPeriodicity(['n','n','p'])
objHex3.SetPeriodicity(['n','n','p'])

objSimulationCell = gl.SimulationCell(np.array([60*a*arrR,60*a*arrUR,h*np.array([0,0,1])]))
objSimulationCell.AddGrain(objHex1)
objSimulationCell.AddGrain(objHex2)
objSimulationCell.AddGrain(objHex3)
objSimulationCell.MergeTooCloseAtoms(0.3*objHex1.GetNearestNeighbourDistance(),1)
objSimulationCell.WriteLAMMPSDataFile(strFilename)

