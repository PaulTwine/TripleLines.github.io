import re
import numpy as np
import GeometryFunctions as gf
from scipy import spatial
class LAMMPSData(object):
    def __init__(self,strFilename: str):
        self.__dctTimeSteps = dict()
        lstNumberOfAtoms = []
        lstTimeSteps = []
        lstColumnNames = []
        lstBoundaryType = []
        self.__Dimensions = 3 # assume 3d unless file shows the problem is 2d
        with open(strFilename) as Dfile:
            while True:
                lstBounds = []
                try:
                    line = next(Dfile).strip()
                except StopIteration as EndOfFile:
                    break
                if "ITEM: TIMESTEP" != line:
                    raise Exception("Unexpected "+repr(line))
                timestep = int(next(Dfile).strip())
                lstTimeSteps.append(timestep)
                line = next(Dfile).strip()
                if "ITEM: NUMBER OF ATOMS" != line:
                    raise Exception("Unexpected "+repr(line))
                N = int(next(Dfile).strip())
                lstNumberOfAtoms.append(N)
                line = next(Dfile).strip()
                if "ITEM: BOX BOUNDS" != line[0:16]:
                    raise Exception("Unexpected "+repr(line))
                lstBoundaryType = line[17:].strip().split()
                lstBounds.append(list(map(float, next(Dfile).strip().split())))
                lstBounds.append(list(map(float, next(Dfile).strip().split())))
                if len(lstBoundaryType)%3 == 0:
                    lstBounds.append(list(map(float, next(Dfile).strip().split())))
                else:
                    self.__Dimensions = 2
                line = next(Dfile).strip()
                if "ITEM: ATOMS id" != line[0:14]:
                    raise Exception("Unexpected "+repr(line))
                lstColumnNames = line[11:].strip().split()
                intNumberOfColumns = len(lstColumnNames)
                objTimeStep = LAMMPSTimeStep(timestep, N,intNumberOfColumns)
                objTimeStep.SetColumnNames(lstColumnNames)
                objTimeStep.SetBoundBoxLabels(lstBoundaryType)
                objTimeStep.SetBoundBoxDimensions(lstBounds)
                for i in range(N):
                    line = next(Dfile).strip().split()
                    objTimeStep.SetRow(i,list(map(float,line)))
                self.__dctTimeSteps[str(timestep)] = objTimeStep            
            self.__lstTimeSteps = lstTimeSteps
            self.__lstNumberOfAtoms = lstNumberOfAtoms
    def GetTimeSteps(self):
        return self.__lstTimeSteps
    def GetAtomNumbers(self):
        return self.__lstNumberOfAtoms
    def GetTimeStep(self, strTimeStep: str):
        return self.__dctTimeSteps[strTimeStep]
    def GetTimeStepByIndex(self, intIndex : int):
        return self.__dctTimeSteps[str(self.__lstTimeSteps[intIndex])]
    def GetNumberOfDimensions(self)-> int:
        return self.__Dimensions 
        
            
class LAMMPSTimeStep(object):
    def __init__(self,fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int):
        self.__NumberOfAtoms = intNumberOfAtoms
        self.__NumberOfColumns = intNumberOfColumns
        self.__TimeStep = fltTimeStep
        self.__AtomData = np.zeros([intNumberOfAtoms,intNumberOfColumns])
        self.__ColumnNames = []
        self.__BoundingBoxLabel = []
        self.__BoundBoxDimensions = []
        self.__Dimensions = 3 #assume three dimensional unless specificed otherwise
    def SetRow(self, intRowNumber: int, lstRow: list):
        self.__AtomData[intRowNumber] = lstRow
    def GetRow(self,intRowNumber: int):
        return self.__AtomData[intRowNumber]
    def GetAtomData(self):
        return self.__AtomData
    def SetColumnNames(self, lstColumnNames):
        self.__ColumnNames = lstColumnNames
    def GetColumnNames(self): 
        return self.__ColumnNames
    def GetColumnByIndex(self, intStructureIndex: int):
        return self.__AtomData[:,intStructureIndex]
    def GetColumnByName(self, strColumnName: str):
        if self.__ColumnNames != []:
            intStructureIndex = self.__ColumnNames.index(strColumnName)
            self.GetColumnByIndex(intStructureIndex)
    def SetBoundBoxLabels(self, lstBoundBox: list):
        self.__BoundBoxLabels = lstBoundBox
        if lstBoundBox[0] == 'xy':
            self.__Cuboid = False
            self.__BoundaryTypes = lstBoundBox[self.__Dimensions:]
        else:
            self.__Cuboid = True
            self.__BoundaryTypes = lstBoundBox
    def GetBoundBoxLabels(self):
        return self.__BoundBoxLabels
    def SetBoundBoxDimensions(self, lstBoundBox):
        self.__BoundBoxDimensions = np.array(lstBoundBox)
        self.__Dimensions = len(lstBoundBox)
        arrCellVectors = np.zeros([self.__Dimensions, self.__Dimensions])
        lstOrigin = []
        for j in range(len(lstBoundBox)):
            lstOrigin.append(lstBoundBox[j][0])
            arrCellVectors[j,j] = lstBoundBox[j][1] - lstBoundBox[j][0]
        if len(lstBoundBox[0]) ==3: #then there are tiltfactors so include "xy" tilt
            arrCellVectors[1,0] = lstBoundBox[0][2]
            if self.__Dimensions == 3: #and there is also a z direction so include "xz" and "yz" tilts
                arrCellVectors[0,0] = arrCellVectors[0,0] -arrCellVectors[1,0] 
                arrCellVectors[2,0] = lstBoundBox[1][2]
                arrCellVectors[2,1] = lstBoundBox[2][2]
        self.__Origin = np.array(lstOrigin)
        self.__CellVectors  = arrCellVectors   
        self.__CellCentre = np.mean(arrCellVectors,axis=0)*self.__Dimensions/2+self.__Origin
        self.__CellBasis = np.zeros([self.__Dimensions,self.__Dimensions])
        for j, vctCell in enumerate(self.__CellVectors):
            self.__CellBasis[j] = vctCell 
        self.__BasisConversion = np.linalg.inv(self.__CellBasis)
    def GetNumberOfAtoms(self):
        return self.__NumberOfAtoms
    def GetNumberOfColumns(self):
        return self.__NumberOfColumns
    def GetCellVectors(self)->np.array:
        return self.__CellVectors
    def GetOrigin(self):
        return self.__Origin
    def GetNumberOfDimensions(self)->int:
        return self.__Dimensions
    def GetCellCentre(self):
        return self.__CellCentre
    def PeriodicEquivalents(self, inPositionVector: np.array)->np.array: #For POSITION vectors only for points within   
        arrVector = np.array([inPositionVector])                         #the simulation cell
        arrCellCoordinates = np.matmul(inPositionVector, self.__BasisConversion)
        for i,strBoundary in enumerate(self.__BoundaryTypes):
            if strBoundary == 'pp':
                 if  arrCellCoordinates[i] > 0.5:
                     arrVector = np.append(arrVector, np.subtract(arrVector,self.__CellVectors[i]),axis=0)
                 elif arrCellCoordinates[i] <= 0.5:
                     arrVector = np.append(arrVector, np.add(arrVector,self.__CellVectors[i]),axis=0)                  
        return arrVector
    def MoveToSimulationCell(self, inPositionVector: np.array)->np.array:
        #arrCellCoordinates = np.matmul(inPositionVector, self.__BasisConversion)
        # if np.min(arrCellCoordinates) < 0 or np.max(arrCellCoordinates) >= 1:
        #     rtnVector = np.zeros(self.__Dimensions)
        #     for j in range(self.__Dimensions):
        #         rtnVector = rtnVector + (arrCellCoordinates[j] % 1)*self.__CellVectors[j]
        #     return rtnVector
        # else:
        #     return inPositionVector
        return gf.WrapVectorInToSimulationCell(self.__CellBasis, self.__BasisConversion, inPositionVector)
    def PeriodicShiftCloser(self, inFixedPoint: np.array, inPointToShift: np.array)->np.array:
        arrPeriodicVectors = self.PeriodicEquivalents(inPointToShift)
        fltDistances = list(map(np.linalg.norm, np.subtract(arrPeriodicVectors, inFixedPoint)))
        return arrPeriodicVectors[np.argmin(fltDistances)]

class OVITOSPostProcess(object):
    def __init__(self,arrGrainQuaternions: np.array, objTimeStep: LAMMPSTimeStep, intLatticeType: int):
        self.__GrainOrientations = arrGrainQuaternions
        self.__NumberOfGrains = len(arrGrainQuaternions)
        self.__LAMMPSTimeStep = objTimeStep
        self.__Dimensions = objTimeStep.GetNumberOfDimensions()
        self.__LatticeStructure = intLatticeType #lattice structure type as defined by OVITOS
        self.__intStructureType = objTimeStep.GetColumnNames().index('StructureType')
        self.__intPositionX = objTimeStep.GetColumnNames().index('x')
        self.__intPositionY = objTimeStep.GetColumnNames().index('y')
        self.__intPositionZ = objTimeStep.GetColumnNames().index('z')
        self.__intQuarternionW = objTimeStep.GetColumnNames().index('OrientationW')
        self.__intQuarternionX = objTimeStep.GetColumnNames().index('OrientationX')
        self.__intQuarternionY = objTimeStep.GetColumnNames().index('OrientationY')
        self.__intQuarternionZ = objTimeStep.GetColumnNames().index('OrientationZ')
       # self.__PeriodicTranslations = objTimeStep.GetPeriodicTranslations()
        lstUnknownAtoms = []
        dctLatticeAtoms = dict()
        dctGrainPointsTree = dict()
        for j in range(self.__NumberOfGrains):
            dctLatticeAtoms[str(j)] = []    
        lstGBAtoms = []
        for j in range(objTimeStep.GetNumberOfAtoms()):
            arrCurrentRow = objTimeStep.GetRow(j)
            intGrainNumber = self.ReturnGrainIndex(arrCurrentRow)
            if (int(arrCurrentRow[self.__intStructureType]) == self.__LatticeStructure):    
                if  int(intGrainNumber) == -1:
                    lstUnknownAtoms.append(arrCurrentRow)
                else:
                    dctLatticeAtoms[str(intGrainNumber)].append(arrCurrentRow)
            else:
                lstGBAtoms.append(arrCurrentRow)
        self.__UnknownAtoms = lstUnknownAtoms
        self.__LatticeAtoms = dctLatticeAtoms
        self.__GBAtoms = lstGBAtoms
        for strGrainKey in dctLatticeAtoms.keys():
            dctGrainPointsTree[strGrainKey] = spatial.KDTree(list(zip(*self.PlotGrain(strGrainKey))))
        self.__dctGrainPointsTree = dctGrainPointsTree
    def ReturnUnknownAtoms(self):
        return np.array(self.__UnknownAtoms)
    def ReturnGrainIndex(self, lstAtomRow: list)->int: #returns -1 if the atoms orientation doesn't match any lattice
        fltTest = 0
        arrAtom = np.array([lstAtomRow[self.__intQuarternionW],lstAtomRow[self.__intQuarternionX],lstAtomRow[self.__intQuarternionY],lstAtomRow[self.__intQuarternionZ]])
        arrAtom = gf.NormaliseVector(arrAtom)
        blnFound = False
        j = 0
        intIndex = -1 #will return an error if lstatom doesn't belong to any grain
        while  (j < self.__NumberOfGrains and not blnFound): 
            objQuarternion = self.__GrainOrientations[j]
            fltTest = np.dot(objQuarternion,gf.QuaternionConjugate(arrAtom)) 
            if 0.999 < abs(fltTest) < 1.001:
                intIndex = j
                blnFound = True
            j = j + 1
        return intIndex
    def PlotTriplePoints(self):
        arrPoints = self.__TripleLine
        return arrPoints[:,:,0],arrPoints[:,:,1], arrPoints[:,:,2]
    def PlotGrain(self, strGrainNumber: str):
        return self.__PlotList(self.__LatticeAtoms[strGrainNumber])
    def PlotUnknownAtoms(self):
        return self.__PlotList(self.__UnknownAtoms)
    def PlotGBAtoms(self):
        return self.__PlotList(self.__GBAtoms)
    def PlotGBOnlyAtoms(self):
        return self.__PlotList(self.__GBOnlyAtoms)
    def PlotTripleLineAtoms(self):
        return self.__PlotList(self.__TripleLineAtoms)
    def PlotPoints(self, inArray: np.array)->np.array:
        return inArray[:,0],inArray[:,1], inArray[:,2]
    def PlotTripleLine(self):
        return self.PlotPoints(self.__TripleLine)
    def __PlotList(self, strList: list):
        arrPoints = np.array(strList)
        return arrPoints[:,self.__intPositionX], arrPoints[:,self.__intPositionY], arrPoints[:,self.__intPositionZ]
    def __GetCoordinates(self, strList: list):
        arrPoint = np.array(strList)
        return arrPoint[self.__intPositionX], arrPoint[self.__intPositionY], arrPoint[self.__intPositionZ]
    def FindClosestGrainPoint(self, arrPoint: np.array,strGrainKey: str)->np.array:
        arrPeriodicPoints = self.__LAMMPSTimeStep.PeriodicEquivalents(arrPoint)
        fltDistances, intIndices =  self.__dctGrainPointsTree[strGrainKey].query(arrPeriodicPoints)
        intMin = np.argmin(fltDistances)
        intDataIndex = intIndices[intMin]
        return self.__dctGrainPointsTree[strGrainKey].data[intDataIndex]
    def NumberOfGBAtoms(self)->int:
        return len(self.__GBAtoms)
    def FindTriplePoints(self):
        arrTriplePoints =np.zeros([self.NumberOfGBAtoms(),self.__NumberOfGrains,self.__Dimensions ])
        arrTripleLine = np.zeros([self.NumberOfGBAtoms(), self.__Dimensions])
        arrLocalGrainBoundaryWidth = np.zeros([self.NumberOfGBAtoms()])
        for n,GBAtom in enumerate(self.__GBAtoms):
            fltLengths = []
            for j,strGrainKey  in enumerate(self.__LatticeAtoms.keys()):
                arrGBAtom = self.__GetCoordinates(GBAtom)
                arrGrainPoint = self.FindClosestGrainPoint(arrGBAtom, strGrainKey)
                arrTriplePoints[n,j] = self.__LAMMPSTimeStep.PeriodicShiftCloser(arrGBAtom,arrGrainPoint)
            arrTripleLine[n] = gf.EquidistantPoint(*arrTriplePoints[n])
            for m in range(len(arrTriplePoints[n])):
                for k in range(m+1,len(arrTriplePoints[n])):
                    fltLengths.append(gf.RealDistance(arrTriplePoints[n,m],arrTriplePoints[n,k]))   
            arrLocalGrainBoundaryWidth[n] = min(fltLengths)
        lstIndicesToDelete = []
        lstTripleLineAtoms = []
        lstGBOnlyAtoms = []
        for j in range(len(arrTripleLine)):
            fltSpacing = gf.RealDistance(arrTriplePoints[j,0], arrTripleLine[j])
            fltDistance = gf.RealDistance(arrTripleLine[j], self.__GetCoordinates(self.__GBAtoms[j]))
            if fltSpacing > arrLocalGrainBoundaryWidth[j] or fltDistance > arrLocalGrainBoundaryWidth[j]:
                lstIndicesToDelete.append(j)
                lstGBOnlyAtoms.append(self.__GBAtoms[j])
            else:
                lstTripleLineAtoms.append(self.__GBAtoms[j])
           # else:
           #     arrTripleLine[j] = self.__LAMMPSTimeStep.MoveToSimulationCell(arrTripleLine[j])
            arrTripleLine[j] = self.__LAMMPSTimeStep.MoveToSimulationCell(arrTripleLine[j])
        arrTripleLine = np.delete(arrTripleLine,lstIndicesToDelete, axis = 0)
        arrTriplePoints = np.delete(arrTriplePoints,lstIndicesToDelete, axis = 0)
        arrLocalGrainBoundaryWidth = np.delete(arrLocalGrainBoundaryWidth,lstIndicesToDelete, axis = 0)
        self.__LocalGrainBoundaryWidth = arrLocalGrainBoundaryWidth
        self.__MeanGrainBoundaryWidth = np.mean(self.__LocalGrainBoundaryWidth) 
        self.__TriplePoints = np.unique(arrTriplePoints, axis=0)
        self.__TripleLine = np.unique(arrTripleLine, axis=0)
        self.__TripleLineAtoms = lstTripleLineAtoms
        self.__GBOnlyAtoms = lstGBOnlyAtoms
    def GetMeanGrainBoundaryWidth(self):
        return self.__MeanGrainBoundaryWidth
    def GetTripleLine(self):
        return self.__TripleLine
    def GetTripleLineAtoms(self):
        return np.array(self.__TripleLineAtoms)
    def GetGBAtoms(self):
        return np.array(self.__GBAtoms)
    def GetGBOnlyAtoms(self):
        return np.array(self.__GBOnlyAtoms)
    def MakePeriodicDistanceMatrix(self, inVector1: np.array, inVector2: np.array)->np.array:
        arrPeriodicDistance = np.zeros([len(inVector1), len(inVector2)])
        for j in range(len(inVector1)):
            for k in range(len(inVector2)):
                arrPeriodicDistance[j,k] = self.PeriodicMinimumDistance(inVector1[j],inVector2[k])
        return arrPeriodicDistance
    def PartitionTripleLines(self):
        arrDistanceMatrix = spatial.distance_matrix(self.__TripleLine, self.__TripleLine)
        setIndices = set(range(len(self.__TripleLine)))
        lstAllTripleLines = []
        while len(setIndices) > 0:            
            lstAllTripleLines.append(self.TripleLineList(setIndices.pop(),arrDistanceMatrix))
            setIndices = setIndices - set(lstAllTripleLines[-1])
        self.__TripleLineGroups = lstAllTripleLines
    def GetNumberOfTripleLines(self)->int:
        return len(self.__TripleLineGroups)
    def PlotNthTripleLine(self, intIndex: int):
        return self.PlotPoints(self.__TripleLine[np.array(self.__TripleLineGroups[intIndex])])
    def TripleLineList(self,inIndex: int, arrDistanceMatrix: np.array)->list:
        counter = 0
        lstTripleLineIndices = []
        lstIndices = [inIndex]
        while (set(lstTripleLineIndices) != set(lstIndices) and  counter < len(self.__TripleLine)):
            arrCurrentMatrix = arrDistanceMatrix[lstIndices]
            fltCurrentMean = np.mean(self.__LocalGrainBoundaryWidth[lstIndices])
            lstIndices = list(np.argwhere(arrCurrentMatrix < 2*fltCurrentMean)[:,1])
            lstTripleLineIndices.extend(lstIndices)
            lstTripleLineIndices = list(set(lstTripleLineIndices))
            counter += 1
        return lstTripleLineIndices
    def MergePeriodicTripleLines(self):
        lstMergedIndices = []
        lstRemainingIndices = list(range(len(self.__TripleLineGroups)))
        lstCurrentIndices = []
        counter = 0
        while (len(lstRemainingIndices)> 1):
            lstCurrentIndices = self.__TripleLineGroups[lstRemainingIndices[0]]
            lstRemainingIndices.remove(lstRemainingIndices[0])
            while (counter < len(lstRemainingIndices)):
                lstTripleLine = self.__TripleLineGroups[lstRemainingIndices[counter]]
                if self.CheckTripleLineEquivalence (lstCurrentIndices,lstTripleLine):
                    lstRemainingIndices.remove(lstRemainingIndices[counter])
                    lstCurrentIndices.extend(lstTripleLine)
                else:
                    counter += 1
            lstMergedIndices.append(lstCurrentIndices)
            counter = 0
        if len(lstRemainingIndices) == 1:
            lstMergedIndices.append(self.__TripleLineGroups[lstRemainingIndices[0]])
        self.__TripleLineGroups = lstMergedIndices
    def CheckTripleLineEquivalence(self,lstTripleLineOne, lstTripleLineTwo)->bool:
        blnFound = False
        counter = 0
        intLengthOne = len(lstTripleLineOne)
        intLengthTwo = len(lstTripleLineTwo)
        while not blnFound and counter < intLengthOne*intLengthTwo:
            i = lstTripleLineOne[np.mod(counter, intLengthOne)]
            j = lstTripleLineTwo[np.mod(counter - i, intLengthTwo)]
            if (self.PeriodicMinimumDistance(self.__TripleLine[i], self.__TripleLine[j]) < max(self.__LocalGrainBoundaryWidth[i],self.__LocalGrainBoundaryWidth[j])):
                blnFound = True
            counter += 1
        return blnFound
    def GetTriplePoints(self):
        return self.__TriplePoints
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        # arrVector1Periodic = self.__LAMMPSTimeStep.PeriodicEquivalents(inVector1)
        # arrDistances = np.zeros(len(arrVector1Periodic))
        # for j, vctCurrent in enumerate(arrVector1Periodic):
        #     arrDistances[j] = gf.RealDistance(vctCurrent, inVector2)
        # return np.min(arrDistances)
        arrVectorPeriodic = self.__LAMMPSTimeStep.PeriodicEquivalents(np.abs(inVector1-inVector2))
        return np.min(np.linalg.norm(arrVectorPeriodic, axis=1))
