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
import LatticeDefinitions as ld
import re
import sys
import sympy as sy



print(gf.CubicCSLGenerator(np.array([1,0,1])))
arrTranslation = np.array([0,0,0])
t= 0.3
lstTransforms = []
arrRotate1 = np.array([[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]])
#arrRotate2 = np.array([[np.cos(-t),np.sin(-t),0],[-np.sin(-t),np.cos(-t),0],[0,0,1]])
arrReflect = np.array([[-1,0,0],[0,-3,0],[0,0,2]]) #- np.array([[1,0,0],[0,1,0],[0,0,1]])
#lstTransforms.append(gf.AffineTransformationMatrix(arrRotate1,arrTranslation))
#lstTransforms.append(gf.AffineTransformationMatrix(arrReflect,arrTranslation))
objOLattice = gf.OLattice(arrReflect,np.array([0.25,0.5,0.3]))
print(objOLattice.GetDiscretePoints())
print(objOLattice.GetOVectors())
print(objOLattice.GetTranslation())
strRoot = '/home/p17992pt/csf3_scratch/CSLGrowthCylinder/Axis100/GBSigma25/' #str(sys.argv[1])
strType = 'C'#str(sys.argv[2])
intDirs = 25  #number of test runs each in a separate directory
if strType == 'C':
	intFiles = 16
elif strType == 'S':
	intFiles = 11
def Linear(x,a,b):
    return a*x + b
def Quadratic(x,a,b):
    return a*x**2+b*x

def ReadToDictionary(strDirectory: str, strExtension: str)->dict():
    objdct = dict()
    for j in os.listdir(strDirectory):
        if j.endswith(strExtension):
            objData = LT.LAMMPSData(strDirectory+ str(j),1,4.05, LT.LAMMPSGlobal)
            objHex = objData.GetTimeStepByIndex(-1)
            objdct[j] = objHex
    return objdct

def CylindricalGrainFitting(indct: dict())->np.array:
    lstKeys = list(indct.keys())
    lstKeys.sort(key=lambda test_string : list(
    map(int, re.findall(r'\d+', test_string)))[0])
    intPE = objdct[lstKeys[0]].GetColumnNames().index('c_pe1')
    intVolume = objdct[lstKeys[0]].GetColumnNames().index('c_v[1]')
    fltHeight = np.linalg.norm(objdct[lstKeys[0]].GetCellVectors()[:,2])
    #fltBasePE = np.sum(indct[lstKeys[0]].GetColumnByName('c_pe1'))
    intBaseAtoms = indct[lstKeys[0]].GetNumberOfAtoms()
    #fltBaseMeanPE = fltBasePE/intBaseAtoms
    lstRadii = []
    lstAdjustedPE =[]
    lstPE = []
    lstRadii = []
    intCounter = 1
    lstAtoms = []
    arrQuaternion = gf.GetQuaternionFromVector(np.array([1,0,0]),0)    
    #lstKeys.remove('read1.dmpPM')
    for i in lstKeys:
        #lstRadii.append(4.05*intCounter)
        fltDatumPE = np.mean(indct[i].GetLatticeAtoms()[:,intPE])
        lstAtoms.append(indct[i].GetNumberOfAtoms())
        lstPE.append(np.sum(indct[i].GetColumnByName('c_pe1')))
        lstIDs = indct[i].GetAtomIDsByOrientation(arrQuaternion,1,0.001)
        fltVolume = np.sum(indct[i].GetAtomsByID(lstIDs)[:,intVolume])
        lstRadii.append(np.sqrt(fltVolume/(fltHeight*np.pi)))
        if len(lstPE) ==1:
            lstAdjustedPE.append(lstPE[0]-fltDatumPE*lstAtoms[0])
        else:
            lstAdjustedPE.append(lstPE[-1]+fltDatumPE*(intBaseAtoms-lstAtoms[-1])-fltDatumPE*intBaseAtoms)
        intCounter +=1
        indct[i].WriteDumpFile(strRoot + str(i)[:-3] + 'upd' )
    arrRadii = np.array(lstRadii)
    arrAdjustedPE = np.array(lstAdjustedPE)  
    return np.array([np.sort(arrRadii),arrAdjustedPE[np.argsort(arrRadii)]])

def SphericalGrainFitting(indct: dict())->np.array:
    lstKeys = list(indct.keys())
    lstKeys.sort(key=lambda test_string : list(
    map(int, re.findall(r'\d+', test_string)))[0])
    intPE = objdct[lstKeys[0]].GetColumnNames().index('c_pe1')
    fltBasePE = np.sum(indct[lstKeys[0]].GetColumnByName('c_pe1'))
    intBaseAtoms = indct[lstKeys[0]].GetNumberOfAtoms()
    fltBaseMeanPE = fltBasePE/intBaseAtoms
    fltDatumPE = np.mean(indct[lstKeys[0]].GetLatticeAtoms()[:,intPE])
    lstRadii = []
    lstExcessPE =[]
    intCounter = 0
    for i in lstKeys:
        lstRadii.append(4.05*intCounter)
        intAtoms = indct[i].GetNumberOfAtoms()
        fltPE = np.sum(indct[i].GetColumnByName('c_pe1'))
        lstExcessPE.append(fltPE-fltBasePE+fltDatumPE*(intBaseAtoms-intAtoms))
        intCounter +=1  
    return np.array(lstExcessPE)


arrValues = np.zeros([intDirs,2,intFiles]) #read0.dmpPM is removed for cylindrical fitting

for i in range(intDirs):
    objdct = ReadToDictionary(strRoot+str(i)+'/','.lst')
    if strType == 'C':   
      arrPE = CylindricalGrainFitting(objdct)
    elif strType =='S':
      arrPE = SphericalGrainFitting(objdct)
    print(i, arrPE)
    arrValues[i] = arrPE
    

arrValues = arrValues.reshape((2*intDirs,intFiles))

np.savetxt(strRoot+'PEValues.txt',arrValues)
