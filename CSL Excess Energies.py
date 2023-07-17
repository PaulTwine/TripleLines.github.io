import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
import LAMMPSTool as LT

strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp450/u01/'
strType = 'TJ'
strRoot += strType + '/'
#objLog = LT.LAMMPSLog(strRoot + strType +  '.log')
#objDat = LT.LAMMPSDat(strRoot + strType + '.dat')
objData = LT.LAMMPSData(strRoot+'1Min.lst',1,4.05,LT.LAMMPSAnalysis3D)
objTJ = objData.GetTimeStepByIndex(-1)
fltPE = np.sum(objTJ.GetColumnByName('c_pe1'))
intAtoms = objTJ.GetNumberOfAtoms()
fltArea = 4*(objTJ.GetCellVectors()[1,1]*objTJ.GetCellVectors()[2,2]) + 2*(objTJ.GetCellVectors()[0,0]*objTJ.GetCellVectors()[2,2])
#fltPE = objLog.GetValues(0)[-1,2]
fltExcess = fltPE - (- 3.36*intAtoms)
print(fltPE,fltArea, intAtoms,fltExcess, fltExcess/fltArea,objTJ.GetCellVectors()[2])