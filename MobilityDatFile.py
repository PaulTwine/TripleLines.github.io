import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import copy as cp
import sys

strDirIn = sys.argv[1]
strDirOut = sys.argv[2]

def MakeEnergyArray(strFilename: str, intSteps=10,intDirs=10, blnExcess = False):
    arrValues = np.loadtxt(strFilename)
    intMax = np.max(arrValues[:,[6,7]])*np.ones(len(arrValues))
    arrAdjusted = cp.deepcopy(arrValues[:,:5])
    arrAdjusted[:,[0,1]] = np.round(arrAdjusted[:,[0,1]],0).astype('int')
    arrAdjusted[:,2] = arrValues[:,2]+(intMax-arrValues[:,6])*arrValues[:,4]
    arrAdjusted[:,3] = arrValues[:,3] + (intMax-arrValues[:,7])*arrValues[:,5]
    if blnExcess:
        arrAdjusted[:,2] = arrAdjusted[:,2]-intMax*arrValues[:,4]
        arrAdjusted[:,3] = arrAdjusted[:,3]-intMax*arrValues[:,5]
    arrAdjusted[:,4] = arrAdjusted[:,3]-arrAdjusted[:,2]
    return arrAdjusted
lstMin = []
arrAdjusted = MakeEnergyArray(strDirIn + 'Values.txt',10,10,True)
for j in range(10):
    arrPoints = np.where(arrAdjusted[:,0] ==j)
    arrPlot = arrAdjusted[arrPoints]
    lstMin.append(np.argmin(arrPlot[:,3])) ##minimum energy TJ Simulation Cell
### Files are encode as TJ5.4 meaning the 5th directory using a float tolerance of 0.4

for k in range(10):
    for strName in ['TJ', 'GB']:
        strFilename = strName + str(lstMin[k]) +'.lst' #str(sys.argv[1])
        objData = LT.LAMMPSData(strDirIn +str(k) + '/' + strFilename, 1, 4.05, LT.LAMMPSAnalysis3D)
        objTimeStep = objData.GetTimeStepByIndex(-1)
        strTemplateName = strDirOut + str(k) + '/' + strFilename[:-3] + 'dat'
        objTimeStep.WriteDataFile(strTemplateName)
        fIn = open(strDirOut +  'TemplateMob.in', 'rt')
        fData = fIn.read()
        fData = fData.replace('read.dat', strTemplateName)
        fData = fData.replace('read.dmp', strTemplateName[:-3] + 'dmp')
        fData = fData.replace('read.lst', strTemplateName[:-3] + 'lst')
        fData = fData.replace('read.log', strTemplateName[:-3] + 'log')
        fIn.close()
        fIn = open(strDirOut + str(k) + '/TemplateMob' + strName + '.in', 'w+')
        fIn.write(fData)
        fIn.close()



