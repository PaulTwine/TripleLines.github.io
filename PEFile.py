import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
from scipy import stats
import os
strPMFile = 'data/60and30.eamPM'
objData = LD.LAMMPSData(strPMFile)
objTimeStep = objData.GetTimeStepByIndex(0)
objPostProcess = LD.OVITOSPostProcess(np.array([1,1,1,1,]), objTimeStep, 1)
print(objTimeStep.GetColumnNames())
MyData = np.zeros([objTimeStep.GetNumberOfAtoms(),4])
MyData[:,0:3] = objTimeStep.GetAtomData()[:,1:4]
MyData[:,3] = objTimeStep.GetAtomData()[:,7]
fltValue = 4.05/2
def MapToMatrix(inArray: np.array, fltParameter: float, blnBiggerBox = False)->np.array:
        if blnBiggerBox:
            inArray =  inArray + fltParameter*np.ones(len(inArray))
        return np.round(inArray/fltParameter).astype('int')
arrDimensions = MapToMatrix(np.array([np.max(MyData[:,0]),np.max(MyData[:,1])]),fltValue, True)
arrRoundedPoints = -3.36/2*5*np.ones(arrDimensions)
for j in MyData:
    if j[2] < 2*fltValue:
        arrPosition = list(MapToMatrix(j[0:2],fltValue, False))
        arrRoundedPoints[arrPosition[0],arrPosition[1]] = np.round(arrRoundedPoints[arrPosition[0],arrPosition[1]] + j[3],0)
plt.matshow(np.transpose(arrRoundedPoints), cmap='CMRmap', interpolation='nearest')
plt.colorbar()
plt.show()
