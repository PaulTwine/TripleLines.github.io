import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy import optimize
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT 
import re
strRoot = '/home/p17992pt/csf3_scratch/CSLGrowthCylinder/Axis100/GBSigma5/'
def ReadToDictionary(strDirectory: str, strExtension: str)->dict():
    objdct = dict()
    intSigma = 5
    #strRoot = 'CSLGrowthCylinder/Axis100/GBSigma' + str(intSigma) + '/9/'
    for j in os.listdir(strDirectory):
        if j.endswith(strExtension):
            objData = LT.LAMMPSData(strDirectory + str(j),1,4.05, LT.LAMMPSGlobal)
            objHex = objData.GetTimeStepByIndex(-1)
            objdct[j] = objHex
    return objdct
def CylindricalGrainFitting(indct: dict())->np.array:
    lstKeys = list(indct.keys())
    lstKeys.sort(key=lambda test_string : list(
    map(int, re.findall(r'\d+', test_string)))[0])
    intPE = objdct[lstKeys[0]].GetColumnNames().index('c_pe1')
    fltBasePE = np.sum(indct[lstKeys[0]].GetColumnByName('c_pe1'))
    intBaseAtoms = indct[lstKeys[0]].GetNumberOfAtoms()
    fltBaseMeanPE = fltBasePE/intBaseAtoms
    fltDatumPE = np.mean(indct[lstKeys[0]].GetLatticeAtoms()[:,7])
    lstRadii = []
    lstExcessPE =[]
    intCounter = 1
    lstKeys.remove(lstKeys[0])
    #lstKeys.remove('read1.dmpPM')
    for i in lstKeys:
        lstRadii.append(4.05*intCounter)
        intAtoms = indct[i].GetNumberOfAtoms()
        fltPE = np.sum(indct[i].GetColumnByName('c_pe1'))
        lstExcessPE.append(fltPE-fltBasePE+fltDatumPE*(intBaseAtoms-intAtoms))
        intCounter +=1  
    return np.array(lstExcessPE)
arrValues = np.zeros([10,10])
for j in range(10):
    objdct = ReadToDictionary(strRoot+str(j) +'/','dmpPM')
    arrPE = CylindricalGrainFitting(objdct)
    arrValues[j] = arrPE
print(arrValues)