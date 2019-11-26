import re
import numpy as np
import GeometryFunctions as gf
from scipy import spatial
from sklearn.cluster import AffinityPropagation
import hdbscan
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
    def GetRows(self, lstOfRows: list):
        return self.__AtomData[lstOfRows]
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
            return self.GetColumnByIndex(intStructureIndex)
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
        return gf.WrapVectorIntoSimulationCell(self.__CellBasis, self.__BasisConversion, inPositionVector)
    def PeriodicShiftCloser(self, inFixedPoint: np.array, inPointToShift: np.array)->np.array:
        arrPeriodicVectors = self.PeriodicEquivalents(inPointToShift)
        fltDistances = list(map(np.linalg.norm, np.subtract(arrPeriodicVectors, inFixedPoint)))
        return arrPeriodicVectors[np.argmin(fltDistances)]
    def StandardiseOrientationData(self):
        self.__AtomData[:, [self.GetColumnNames().index('OrientationX'),self.GetColumnNames().index('OrientationY'),self.GetColumnNames().index('OrientationZ'), self.GetColumnNames().index('OrientationW')]]=np.apply_along_axis(gf.FCCQuaternionEquivalence,1,self.GetOrientationData()) 
    def GetOrientationData(self)->np.array:
        return (self.__AtomData[:, [self.GetColumnNames().index('OrientationX'),self.GetColumnNames().index('OrientationY'),self.GetColumnNames().index('OrientationZ'), self.GetColumnNames().index('OrientationW')]])  
    def GetData(self, inDimensions: np.array, lstOfColumns):
        return np.where(self.__AtomData[:,lstOfColumns])

class OVITOSPostProcess(object):
    def __init__(self,arrGrainQuaternions: np.array, objTimeStep: LAMMPSTimeStep, intLatticeType: int):
        self.__GrainOrientations = arrGrainQuaternions
        self.__NumberOfGrains = len(arrGrainQuaternions)
        self.__LAMMPSTimeStep = objTimeStep
        self.__Dimensions = objTimeStep.GetNumberOfDimensions()
        self.__LatticeStructure = intLatticeType #lattice structure type as defined by OVITOS
        self.__intStructureType = int(objTimeStep.GetColumnNames().index('StructureType'))
        self.__intPositionX = int(objTimeStep.GetColumnNames().index('x'))
        self.__intPositionY = int(objTimeStep.GetColumnNames().index('y'))
        self.__intPositionZ = int(objTimeStep.GetColumnNames().index('z'))
        self.__intQuarternionW = objTimeStep.GetColumnNames().index('OrientationW')
        self.__intQuarternionX = objTimeStep.GetColumnNames().index('OrientationX')
        self.__intQuarternionY = objTimeStep.GetColumnNames().index('OrientationY')
        self.__intQuarternionZ = objTimeStep.GetColumnNames().index('OrientationZ')
       # self.__PeriodicTranslations = objTimeStep.GetPeriodicTranslations()
        lstUnknownAtoms = []
        lstLatticeAtoms = []
        # for j in range(self.__NumberOfGrains):
        #     dctLatticeAtoms[str(j)] = []    
        lstGBAtoms = []
        # for j in range(objTimeStep.GetNumberOfAtoms()):
        #     arrCurrentRow = objTimeStep.GetRow(j)
        #     #intGrainNumber = self.ReturnGrainIndex(arrCurrentRow)
            #if (int(arrCurrentRow[self.__intStructureType]) == self.__LatticeStructure):    
        #intAtomStructure = int(arrCurrentRow[self.__intStructureType])
        lstGBAtoms = list(np.where(objTimeStep.GetColumnByIndex(self.__intStructureType).astype(int) == 0)[0])
        lstLatticeAtoms =  list(np.where(objTimeStep.GetColumnByIndex(self.__intStructureType).astype(int) == intLatticeType)[0])
        lstUnknownAtoms = list((np.where(objTimeStep.GetColumnByIndex(self.__intStructureType).astype(int) != intLatticeType) and np.where(objTimeStep.GetColumnByIndex(self.__intStructureType).astype(int) != 0))[0])
        self.__LatticeAtoms = lstLatticeAtoms
        self.__GBAtoms = lstGBAtoms
        self.__GBTree =  spatial.KDTree(list(zip(*self.__PlotList(lstGBAtoms))))
        self.__LatticeTree = spatial.KDTree(list(zip(*self.__PlotList(lstLatticeAtoms))))
    def GetUnknownAtoms(self):
        return self.__LAMMPSTimeStep.GetRows(self.__UnknownAtoms)   
    def ReturnGrainIndex(self, lstAtomRow: list)->int: #returns -1 if the atoms orientation doesn't match any lattice
        fltTest = 0
        arrAtom = np.array([lstAtomRow[self.__intQuarternionX],lstAtomRow[self.__intQuarternionY],lstAtomRow[self.__intQuarternionZ],lstAtomRow[self.__intQuarternionW]])
        arrAtom = gf.FCCQuaternionEquivalence(arrAtom)
        #blnFound = False
        j = 0
        intIndex = -1 #will return an error if lstatom doesn't belong to any grain
        fltValue = 0.999
        while  (j < self.__NumberOfGrains): 
            #objQuaternion = np.sort(np.abs(gf.FCCQuaternionEquivalence(self.__GrainOrientations[j])))
            objQuaternion = self.__GrainOrientations[j]
            #fltTest = np.max(np.abs(np.matmul(gf.QuaternionConjugate(gf.FCCQuaternionEquivalence(arrAtom)),np.transpose(objQuarternion))))
            fltTest = np.abs(np.dot(arrAtom,objQuaternion))
           # fltTest2 = np.abs(np.dot(arrAtom,objQuaternion))
            if fltTest > fltValue:
                intIndex = j
                fltValue = fltTest
            j = j + 1
        return intIndex
    def PlotTriplePoints(self):
        arrPoints = self.__TripleLine
        return arrPoints[:,:,0],arrPoints[:,:,1], arrPoints[:,:,2]
    def PlotGrain(self, strGrainNumber: str):
        #return self.__PlotList(self.__LatticeAtoms[strGrainNumber])
        return self.__PlotList(self.__LatticeAtoms)
    def PlotUnknownAtoms(self):
        return self.__PlotList(self.__UnknownAtoms)
    def PlotGBAtoms(self):
        return self.__PlotList(self.__GBAtoms)
    def PlotGBOnlyAtoms(self):
        return self.__PlotList(self.__GBOnlyAtoms)
    def PlotTripleLineAtoms(self):
        return self.__PlotList(self.__TripleLineAtoms)
    def PlotDislocations(self):
        return self.__PlotList(self.__Dislocations)
    def PlotPoints(self, inArray: np.array)->np.array:
        return inArray[:,0],inArray[:,1], inArray[:,2]
    def PlotTripleLine(self):
        return self.PlotPoints(self.__TripleLine)
    def __PlotList(self, strList: list):
        arrPoints = self.__LAMMPSTimeStep.GetRows(strList)
        return arrPoints[:,self.__intPositionX], arrPoints[:,self.__intPositionY], arrPoints[:,self.__intPositionZ]
        #return arrPoints[:,[1,2,3]]
    def __GetCoordinates(self, strList: list):
        arrPoint = self.__LAMMPSTimeStep.GetRows(strList)
        return np.array([arrPoint[self.__intPositionX], arrPoint[self.__intPositionY], arrPoint[self.__intPositionZ]])
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
        return self.__LAMMPSTimeStep.GetRows(self.__GBAtoms)
    def GetGBOnlyAtoms(self):
        return self.__LAMMPSTimeStep.GetRows(self.__GBOnlyAtoms)
    def MakePeriodicDistanceMatrix(self, inVector1: np.array, inVector2: np.array)->np.array:
        arrPeriodicDistance = np.zeros([len(inVector1), len(inVector2)])
        for j in range(len(inVector1)):
            for k in range(j,len(inVector2)):
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
    def SortByAngles(self,inPoints: np.array, inStructure):#rotates around a point and counts the number of transitions from GB to grain
        intCount = 0 # pass an array of the form [xcoord, ycoord, Structure]
        fltOldValue = inStructure[0]
        arrValues = np.zeros([len(inStructure),2])
        arrValues[:,0] = np.arctan2(inPoints[:,1],inPoints[:,0])
        arrValues[:,1] = np.round(inStructure,0)
        arrValues = arrValues[arrValues[:,0].argsort()]
        for j in arrValues:
            fltNewValue = j[-1]
            if int(fltNewValue) != int(fltOldValue):
                intCount += 1
                fltOldValue = fltNewValue
        return intCount
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        # arrVector1Periodic = self.__LAMMPSTimeStep.PeriodicEquivalents(inVector1)
        # arrDistances = np.zeros(len(arrVector1Periodic))
        # for j, vctCurrent in enumerate(arrVector1Periodic):
        #     arrDistances[j] = gf.RealDistance(vctCurrent, inVector2)
        # return np.min(arrDistances)
        arrVectorPeriodic = self.__LAMMPSTimeStep.PeriodicEquivalents(np.abs(inVector1-inVector2))
        return np.min(np.linalg.norm(arrVectorPeriodic, axis=1))
    def ClassifyNonGrainAtoms(self):
        lstUnknownAtoms = []
        lstTripleLines = []
        lstDislocations = []
        lstGBAtoms = self.__GBAtoms
        # intSwaps = 0 #the number of times this has swapped from grain boundary to grain 
        # CurrentList = np.where(self.__LAMMPSTimeStep.GetColumnByIndex(self.__intStructureType) == 0)  and np.where(self.__LAMMPSTimeStep.GetColumnByIndex(self.__intPositionZ) < 10)[0]
        # for j in CurrentList:
        #     arrGBPoint = self.__LAMMPSTimeStep.GetRow(j)
        #     arrPoints = self.__LAMMPSTimeStep.GetAtomData() - arrGBPoint
        #     arrPoints = arrPoints[:,self.__intPositionX:self.__intPositionY+1] #centre the points with j at the origin
        #     lstOfIndices = np.where(np.linalg.norm(arrPoints, axis=1) > 12) and np.where(np.linalg.norm(arrPoints, axis=1) < 14)
        #     arrPoints = arrPoints[lstOfIndices]
        #     arrStructure = self.__LAMMPSTimeStep.GetColumnByIndex(self.__intStructureType)[lstOfIndices]
        #     intSwaps = self.SortByAngles(arrPoints, arrStructure)
        #     if intSwaps == 4:
        #         lstGBAtoms.append(j)
        #     elif intSwaps == 6:
        #         lstTripleLines.append(j)
        #     else:
        #         lstUnknownAtoms.append(j)
        for j in self.__GBAtoms:
            arrIndices = self.__LatticeTree.query_ball_point(self.__GetCoordinates(j), 2*4.05*np.sqrt(3))
            arrPoints = self.__LatticeTree.data[arrIndices]
            #af = AffinityPropagation(preference=-50).fit(arrPoints) 
            #intClusters = len(af.cluster_centers_indices_)
            #objCluster = hdbscan.HDBSCAN(metric='matching').fit(arrPoints) 
            #intClusters = objCluster.labels_.max()
            lstOfGrainIndices = []
            lstOfCurrentGrainIndices = []
            intGrains = 0
            setRemaingIndices = set(range(len(arrPoints)))
            arrDistanceMatrix = spatial.distance.cdist(arrPoints, arrPoints,'euclidean')
            while len(setRemaingIndices) > 0:
                lstOfCurrentGrainIndices =  self.FindGrainGroups(1.02*4.05/(np.sqrt(2)), arrPoints,arrDistanceMatrix, setRemaingIndices.pop())
                lstOfGrainIndices.extend(lstOfCurrentGrainIndices)
                setRemaingIndices = setRemaingIndices - set(lstOfGrainIndices)
                if len(lstOfCurrentGrainIndices) > 0:
                    intGrains += 1
            if (intGrains ==1):
                lstDislocations.append(j)
                lstGBAtoms.remove(j)
            elif intGrains == 3:
                lstTripleLines.append(j)
                lstGBAtoms.remove(j)
            elif intGrains > 3:
                lstUnknownAtoms.append(j)
                lstGBAtoms.remove(j)  
        self.__Dislocations = lstDislocations          
        self.__GBAtoms = lstGBAtoms
        self.__TripleLineAtoms = lstTripleLines
        self.__UnknownAtoms = lstUnknownAtoms
    def FindGrainGroups(self,fltDistance: float, arrPoints: np.array, inDistanceMatrix: np.array, intStart: int)->list:
         #setAllUsedRows = set(0)
         #setCurrentRows = set()
        lstCurrentRows = [intStart]
        lstAllUsedRows = []
       # arrDistanceMatrix = self.MakePeriodicDistanceMatrix(arrPoints,arrPoints)
        while (len(lstCurrentRows) > 0):
            lstCurrentRows = np.argwhere(np.any(inDistanceMatrix[lstCurrentRows] > 0, axis = 0) & np.any(inDistanceMatrix[lstCurrentRows]< fltDistance,axis=0))
            lstCurrentRows = list(set(lstCurrentRows[:,0]).difference(set(lstAllUsedRows)))
            lstAllUsedRows.extend(lstCurrentRows)
        return lstAllUsedRows
            