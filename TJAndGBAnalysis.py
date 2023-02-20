#%%
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
#%%
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
#%%
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
#%%
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
# %%
class AxisStore(object):
    def __init__(self, arrAxis: int, strType: str):
        self.___Axis = arrAxis
        self.__Type = strType
        self.__Values = dict() 
    def SetSigmaStore(self, inDirStore: DirStore, strKey: str):
        self.__Values[strKey] = inDirStore
    def GetSigmaStore(self, strKey: str):
        return self.__Values[strKey] 
#%%
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
fig, ax = plt.subplots()
#%%
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
                    break
                if line in lstKeys:
                    if len(lstOfValues) > 0:
                        self.SetValues(lstOfValues,strKey)
                        lstOfValues = []
                    strKey = line
                else: 
                    lstOfValues.append(list(map(lambda x: float(x), line.split(','))))
            fdata.close()
#%%
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
#%%
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
#%%
class AxisStore(object):
    def __init__(self, arrAxis: int, strType: str):
        self.___Axis = arrAxis
        self.__Type = strType
        self.__Values = dict() 
    def SetSigmaStore(self, inDirStore: DirStore, strKey: str):
        self.__Values[strKey] = inDirStore
    def GetSigmaStore(self, strKey: str):
        return self.__Values[strKey] 
#%%
def PopulateSigmaStore(intSigma: int, arrAxis: np.array,strRootDir:str, strType: str)->SigmaStore:
    objSigmaStore = SigmaStore(arrAxis,intSigma, strType)
    for j in range(1): #directories
        objDirStore = DirStore(arrAxis,intSigma,j,strType)
        for i in range(10): #delta values
            objDeltaStore = DeltaStore(arrAxis,intSigma,j,i,strType)
            strFilename = strRootDir + str(j)  + '/' + strType + str(i) + 'P.lst'
            strSavename = strRootDir + str(j)  + '/' + strType + str(i) + 'P.txt'
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
                    idsG = np.append(idsG1,idsG2, axis=0)
                lstG = [np.array([len(idsG), np.sum(objLT.GetAtomsByID(idsG)[:,intPE]), np.sum(objLT.GetAtomsByID(idsG)[:,intV]),
                             np.sum(np.sum(objLT.GetAtomsByID(idsG)[:,intC1:intC3+1],axis=1))])]
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
            objDirStore.SetDeltaStore(objDeltaStore, i)
            objDeltaStore.WriteFileOfValues(strSavename)
        objSigmaStore.SetDirStore(objDirStore,j)
    return objSigmaStore
#%%
def PopulateSigmaStoreByFile(intSigma: int, arrAxis: np.array,strRootDir:str, strType: str, lstOfKeys: list)->SigmaStore:
    objSigmaStore = SigmaStore(arrAxis,intSigma, strType)
    for j in range(1): #directories
        objDirStore = DirStore(arrAxis,intSigma,j,strType)
        for i in range(10): #delta values
            objDeltaStore = DeltaStore(arrAxis,intSigma,j,i,strType)
            strLoadname = strRootDir + str(j)  + '/' + strType + str(i) + 'P.txt'
            objDeltaStore.ReadFileOfValues(strLoadname, lstOfKeys)
            objDirStore.SetDeltaStore(objDeltaStore, i)
        objSigmaStore.SetDirStore(objDirStore,j)
    return objSigmaStore
#%%
objStore = PopulateSigmaStoreByFile(13, np.array([0,0,1]),'/home/paul/csf4_scratch/TJ/Axis001/TJSigma13/','GB',['GE','PE','V','S'])
objDirStore = objStore.GetDirStore(0)
objDeltaStore = objDirStore.GetDeltaStore(0)
print(objDeltaStore.GetValues('PE')[1])
#%%
objGB = PopulateSigmaStore(17,np.array([0,0,1]),'/home/p17992pt/csf4_scratch/TJ/Axis001/TJSigma13/','GB')
#%%
objTJ = PopulateSigmaStore(17,np.array([0,0,1]),'/home/p17992pt/csf4_scratch/TJ/Axis001/TJSigma13/','TJ')
#%%
#fltDatum = 4.05**3/4 
#fltDatum = -3.36
fltDatum = 0
#strType = 'V' 
strType = 'PE'
strType = 'S'
lstColours = ['b', 'c', 'r', 'g', 'm','y','k','w']
intDeltaMax = 10
intDeltaMin = 0
for j in range(1):
    objDirGB = objGB.GetDirStore(j)
    objDirTJ = objTJ.GetDirStore(j)
    for i in range(intDeltaMin,intDeltaMax):
        objDeltaTJ = objDirTJ.GetDeltaStore(i)
        lstPEValuesTJ = objDeltaTJ.GetValues(strType)
        intCol = 0
        for l in lstPEValuesTJ[1:]:
            plt.scatter(i/10-0.01,(np.mean(l)-fltDatum),c='g',label='Tripleline')
            #plt.scatter(i,np.mean(l)-fltVDatum,c='g',label='Tripleline')#c =lstColours[intCol])
            intCol +=1
        objDeltaGB = objDirGB.GetDeltaStore(i)
        lstPEValuesGB = objDeltaGB.GetValues(strType)
        intCol = 0
        arrLengths  = np.argsort(list(map(lambda x: len(x),lstPEValuesGB)))
        for k in arrLengths:
            arrValues = lstPEValuesGB[k]
            if intCol < 2:
                plt.scatter(i/10+0.01,(np.mean(arrValues)-fltDatum),c='r',label = 'Cylinder')
            elif intCol ==2:
                plt.scatter(i/10+0.01,(np.mean(arrValues)-fltDatum),c='y',label = 'CSL')
            else:
            #plt.scatter(i,np.mean(k)-(4.05**3/4),c= 'b') #c =lstColours[intCol])
                plt.scatter(i/10,(np.mean(arrValues)-fltDatum),c='b',label = 'Grain')#c =lstColours[intCol])
            
            intCol +=1 
a = ax.get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
ax.legend(*d)
if strType == 'V':
    strYlabel = 'Excess volume per atom in $\AA^{3}$ per atom'   
elif strType == 'PE':
    strYlabel = 'Excess potential energy per atom in eV per atom'
plt.ylabel(strYlabel)
plt.xlabel('$\delta /r_0$')
plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax)))/10)
#plt.ylim([0.015, 0.045])
plt.tight_layout()
plt.show()


#%%
 
# %%
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
# %%
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
#%%
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
# %%
class AxisStore(object):
    def __init__(self, arrAxis: int, strType: str):
        self.___Axis = arrAxis
        self.__Type = strType
        self.__Values = dict() 
    def SetSigmaStore(self, inDirStore: DirStore, strKey: str):
        self.__Values[strKey] = inDirStore
    def GetSigmaStore(self, strKey: str):
        return self.__Values[strKey] 