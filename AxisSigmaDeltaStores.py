
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp
from scipy import spatial
from scipy import optimize

class DeltaStore(object):
    def __init__(self, arrAxis: int, intSigma,intDirNo: int, intDelta: int, strType: str):
        self.___Axis = arrAxis
        self.__Sigma = intSigma
        self.__DirNo = intDirNo
        self.__intDelta = intDelta
        self.__Type = strType
        self.__Values = dict() 
    def SetValues(self, lstOfValues: list, strKey: str):
        self.__Values[strKey] = lstOfValues
    def GetValues(self, strKey: str):
        return self.__Values[strKey]  
    def WriteFileOfValues(self, strFilename):
        with open(strFilename, 'w') as fdata:
            for k in self.__Values:
                fdata.write(str(k)  + '\n') 
                for i in self.GetValues(k):
                    fdata.write(','.join(map(str,i)))
                    fdata.write('\n')
                    
            fdata.close() 
    def ReadFileOfValues(self, strFilename, lstKeys):
        blnGo = True
        with open(strFilename, 'r') as fdata:
            lstOfValues = []
            while blnGo:
                try:
                    line = next(fdata).strip()
                except StopIteration as EndOfFile:
                    blnGo = False
                if line in lstKeys:
                    if len(lstOfValues) > 0:
                        self.SetValues(lstOfValues,strKey)
                        lstOfValues = []
                    strKey = line
                else: 
                    lstOfValues.append(list(map(lambda x: float(x), line.split(','))))
            if not(blnGo):
                self.SetValues(lstOfValues,strKey)
            fdata.close()
class DirStore(object):
    def __init__(self, arrAxis: int, intSigma,intDirNo: int, strType: str):
        self.___Axis = arrAxis
        self.__Sigma = intSigma
        self.__DirNo = intDirNo
        self.__Type = strType
        self.__Values = dict() 
    def SetDeltaStore(self, inDeltaStore: DeltaStore, strKey: str):
        self.__Values[strKey] = inDeltaStore
    def GetDeltaStore(self, strKey: str):
        return self.__Values[strKey]
class SigmaStore(object):
    def __init__(self, arrAxis: int, intSigma, strType: str):
        self.___Axis = arrAxis
        self.__Sigma = intSigma
        self.__Type = strType
        self.__Values = dict() 
    def SetDirStore(self, inDirStore: DirStore, strKey: str):
        self.__Values[strKey] = inDirStore
    def GetDirStore(self, strKey: str):
        return self.__Values[strKey] 
class AxisStore(object):
    def __init__(self, arrAxis: int, strType: str):
        self.___Axis = arrAxis
        self.__Type = strType
        self.__Values = dict() 
    def SetSigmaStore(self, inDirStore: DirStore, strKey: str):
        self.__Values[strKey] = inDirStore
    def GetSigmaStore(self, strKey: str):
        return self.__Values[strKey]
def PopulateDeltaStore(intSigma: int, arrAxis: np.array,strRootDir:str, strType: str,intDir,intDelta, blnWriteFile = False)->DeltaStore: 
    objDeltaStore = DeltaStore(arrAxis,intSigma,intDir,intDelta,strType)
    strFilename = strRootDir + str(intDir)  + '/' + strType + str(intDelta) + 'P.lst'
    strSavename = strRootDir + str(intDir)  + '/' + strType + str(intDelta) + 'P.txt'
    objData = LT.LAMMPSData(strFilename, 1, 4.05, LT.LAMMPSAnalysis3D)
    objLT = objData.GetTimeStepByIndex(-1)
    if strType == 'TJ':
        lstLabels = objLT.GetLabels('TripleLine')
    elif strType =='GB':
        lstLabels = objLT.GetLabels('GrainBoundary')
    intV = objLT.GetColumnIndex('c_v[1]')
    intPE = objLT.GetColumnIndex('c_pe1')
    intC1 = objLT.GetColumnIndex('c_st[1]')
    intC3 = objLT.GetColumnIndex('c_st[3]')    
    lstPE = [] #Pe per atom
    lstV = [] # vollume per atom
    lstS = [] # hydrostatic stress per atom
    lstG = [] #one entry for the total excess energy stored in the grains compared to an ideal crystal
    if -1 in lstLabels:
        lstLabels.remove(-1)
    if 0 in lstLabels:
        if strType =='GB':
            idsG = objLT.GetGrainBoundaryIDs(0)
        elif strType =='TJ':
            idsG1 = objLT.GetTripleLineIDs(0)
            idsG2 = objLT.GetGrainBoundaryIDs(0)
            idsG = list(set(idsG1).intersection(set(idsG2)))
        idsG = np.unique(idsG).tolist()
        lstG = [np.array([len(idsG), np.sum(objLT.GetAtomsByID(idsG)[:,intPE]), np.sum(objLT.GetAtomsByID(idsG)[:,intV]),np.sum(np.sum(objLT.GetAtomsByID(idsG)[:,intC1:intC3+1],axis=1))])]
        objDeltaStore.SetValues(lstG, 'GE')
        lstLabels.remove(0)
    for k in  lstLabels:
        if strType == 'TJ':
            ids = objLT.GetTripleLineIDs(k)
        elif strType == 'GB':
            ids = objLT.GetGrainBoundaryIDs(k)              
        lstPE.append(objLT.GetColumnByIDs(ids,intPE))
        lstV.append(objLT.GetColumnByIDs(ids,intV))
        lstS.append(np.sum(objLT.GetAtomsByID(ids)[:,intC1:intC3+1],axis=1))          
    objDeltaStore.SetValues(lstPE, 'PE')
    objDeltaStore.SetValues(lstV, 'V')
    objDeltaStore.SetValues(lstS, 'S')
    if blnWriteFile:
        objDeltaStore.WriteFileOfValues(strSavename) 
def PopulateSigmaStore(intSigma: int, arrAxis: np.array,strRootDir:str, strType: str, blnWriteFile = False)->SigmaStore:
    objSigmaStore = SigmaStore(arrAxis,intSigma, strType)
    for j in range(10): #directories
        objDirStore = DirStore(arrAxis,intSigma,j,strType)
        for i in range(10): #delta values
            objDeltaStore = PopulateDeltaStore(intSigma, arrAxis,strRootDir,strType,j,i,blnWriteFile)
            objDirStore.SetDeltaStore(objDeltaStore, i)
        objSigmaStore.SetDirStore(objDirStore,j)
    return objSigmaStore
def PopulateSigmaStoreByFile(intSigma: int, arrAxis: np.array,strRootDir:str, strType: str, lstOfKeys: list)->SigmaStore:
    objSigmaStore = SigmaStore(arrAxis,intSigma, strType)
    for j in range(10): #directories
        objDirStore = DirStore(arrAxis,intSigma,j,strType)
        for i in range(10): #delta values
            objDeltaStore = DeltaStore(arrAxis,intSigma,j,i,strType)
            strLoadname = strRootDir + str(j)  + '/' + strType + str(i) + 'P.txt'
            objDeltaStore.ReadFileOfValues(strLoadname, lstOfKeys)
            objDirStore.SetDeltaStore(objDeltaStore, i)
        objSigmaStore.SetDirStore(objDirStore,j)
    return objSigmaStore


