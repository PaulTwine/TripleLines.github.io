import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
import copy as cp
from scipy import spatial
import MiscFunctions as mf
from mpl_toolkits.mplot3d import Axes3D


#arrFValues the rows are the grains 1,2 and 3
#Fix 1 is column 1
# Fix 2 is column 2 
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis001/Sigma5_5_25/Temp550/' #str(sys.argv[1])
lstGrains = ['1G', '2G', '3G']
arrFValues = np.zeros([3,2])
i = 0
for j in lstGrains:
    objData = LT.LAMMPSData(strRoot + j + '/1Sim1000.dmp',1,4.05, LT.LAMMPSGlobal)
    objG = objData.GetTimeStepByIndex(-1)
    fltF12 = np.mean(objG.GetColumnByName('f_1[2]'))
    fltF22 = np.mean(objG.GetColumnByName('f_2[2]'))
    arrFValues[i,0] = fltF12 
    arrFValues[i,1] = fltF22
    i +=1
if arrFValues[1,1] ==  arrFValues[2,1]:
    u_1 = 0
    u_2 = 2/(arrFValues[1,0]-arrFValues[1,1])
else:
    r = (arrFValues[2,0]-arrFValues[1,0])/(arrFValues[1,1]-arrFValues[2,1])
    u_1 = 2/(arrFValues[0,0]-arrFValues[1,0]+ r*(arrFValues[0,1]-arrFValues[1,1]))
    u_2 = r*u_1
print(u_1,u_2)
np.savetxt(strRoot + '../arrU.txt', np.array([u_1,u_2]))
