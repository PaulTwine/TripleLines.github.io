import re
import numpy as np
import GeometryFunctions as gf
import GeneralLattice as gl
from scipy import spatial, optimize, ndimage
#from sklearn.cluster import AffinityPropagation
from skimage.morphology import skeletonize, thin, medial_axis, remove_small_holes,label
#from scipy.ndimage import label

class LAMMPSData(object):
    def __init__(self,strFilename: str, intLatticeType: int):
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
                objTimeStep = LAMMPSAnalysis(timestep, N,intNumberOfColumns,lstColumnNames, lstBoundaryType, lstBounds,intLatticeType)
                objTimeStep.SetColumnNames(lstColumnNames)
                for i in range(N):
                    line = next(Dfile).strip().split()
                    objTimeStep.SetRow(i,list(map(float,line)))
                objTimeStep.CategoriseAtoms()
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
    def __init__(self,fltTimeStep: float,intNumberOfAtoms: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list):
        self.__Dimensions = 3 #assume three dimensional unless specificed otherwise
        self.__NumberOfAtoms = intNumberOfAtoms
        self.__NumberOfColumns = len(lstColumnNames)
        self.__TimeStep = fltTimeStep
        self.__AtomData = np.zeros([intNumberOfAtoms,self.__NumberOfColumns])
        self.__ColumnNames = lstColumnNames
        self.SetBoundBoxLabels(lstBoundaryType)
        self.SetBoundBoxDimensions(lstBounds)
    def SetRow(self, intRowNumber: int, lstRow: list):
        self.__AtomData[intRowNumber] = lstRow
    def GetRow(self,intRowNumber: int):
        return self.__AtomData[intRowNumber]
    def GetRows(self, lstOfRows: list):
        return self.__AtomData[lstOfRows,:]
    def GetAtomsByID(self, lstOfAtomIDs: list, intAtomColumn = 0):
        return self.__AtomData[np.isin(self.__AtomData[:,intAtomColumn],lstOfAtomIDs)]
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
        self.__UnitCellBasis = np.zeros([self.__Dimensions,self.__Dimensions])
        for j, vctCell in enumerate(self.__CellVectors):
            self.__CellBasis[j] = vctCell 
            self.__UnitCellBasis[j] = gf.NormaliseVector(vctCell)
        self.__BasisConversion = np.linalg.inv(self.__CellBasis)
        self.__UnitBasisConversion = np.linalg.inv(self.__UnitCellBasis)
    def GetBasisConversions(self):
        return self.__BasisConversion
    def GetUnitBasisConversions(self):
        return self.__UnitBasisConversion
    def GetCellBasis(self):
        return self.__CellBasis
    def GetUnitCellBasis(self):
        return self.__UnitCellBasis
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
    def PeriodicShiftAllCloser(self, inFixedPoint: np.array, inAllPointsToShift: np.array)->np.array:
        arrPoints = np.array(list(map(lambda x: self.PeriodicShiftCloser(inFixedPoint, x), inAllPointsToShift)))
        return arrPoints
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
    def GetBoundingBox(self):
        return np.array([self.GetCellBasis()[0,0]+self.GetCellBasis()[1,0]+self.GetCellBasis()[2,0],
        self.GetCellBasis()[1,1]+self.GetCellBasis()[2,1], self.GetCellBasis()[2,2]])


class LAMMPSPostProcess(LAMMPSTimeStep):
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int):
        LAMMPSTimeStep.__init__(self,fltTimeStep,intNumberOfAtoms, lstColumnNames, lstBoundaryType, lstBounds)
        self.__Dimensions = self.GetNumberOfDimensions()
        self._LatticeStructure = intLatticeType #lattice structure type as defined by OVITOS
        self._intStructureType = int(self.GetColumnNames().index('StructureType'))
        self._intPositionX = int(self.GetColumnNames().index('x'))
        self._intPositionY = int(self.GetColumnNames().index('y'))
        self._intPositionZ = int(self.GetColumnNames().index('z'))
        self._intPE = int(self.GetColumnNames().index('c_pe1'))
        self.CellHeight = np.linalg.norm(self.GetCellVectors()[2])
    def CategoriseAtoms(self):    
        lstOtherAtoms = list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == 0)[0])
        lstLatticeAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == self._LatticeStructure)[0])
        lstUnknownAtoms = list(np.where(np.isin(self.GetColumnByIndex(self._intStructureType).astype('int') ,[0,1],invert=True))[0])
        self.__LatticeAtoms = lstLatticeAtoms
        self.__NonLatticeAtoms = lstOtherAtoms + lstUnknownAtoms
        self.__OtherAtoms = lstOtherAtoms
        self.__NonLatticeTree =  spatial.KDTree(list(zip(*self.__PlotList(lstOtherAtoms+lstUnknownAtoms))))
        self.__LatticeTree = spatial.KDTree(list(zip(*self.__PlotList(lstLatticeAtoms))))
        self.__UnknownAtoms = lstUnknownAtoms
    def GetNonLatticeAtoms(self):
        return self.GetRows(self.__NonLatticeAtoms)
    def GetUnknownAtoms(self):
        return self.GetRows(self.__UnknownAtoms) 
    def GetLatticeAtoms(self):
        return self.GetRows(self.__LatticeAtoms)  
    def GetOtherAtoms(self):
        return self.GetRows(self.__OtherAtoms)
    def GetNumberOfNonLatticeAtoms(self):
        return len(self.__NonLatticeAtoms)
    def GetNumberOfOtherAtoms(self)->int:
        return len(self.GetRows(self.__OtherAtoms))
    def GetNumberOfLatticeAtoms(self)->int:
        return len(self.__LatticeAtoms)
    def PlotGrainAtoms(self, strGrainNumber: str):
        return self.__PlotList(self.__LatticeAtoms)
    def PlotUnknownAtoms(self):
        return self.__PlotList(self.__UnknownAtoms)
    def PlotPoints(self, inArray: np.array)->np.array:
        return inArray[:,0],inArray[:,1], inArray[:,2]
    def __PlotList(self, strList: list):
        arrPoints = self.GetRows(strList)
        return arrPoints[:,self._intPositionX], arrPoints[:,self._intPositionY], arrPoints[:,self._intPositionZ]
    def __GetCoordinate(self, intIndex: int):
        arrPoint = self.GetRow(intIndex)
        return arrPoint[self._intPositionX:self._intPositionZ+1]
    def __GetCoordinates(self, strList: list):
        arrPoints = self.GetRows(strList)
        return arrPoints[:,self._intPositionX:self._intPositionZ+1]
    def MakePeriodicDistanceMatrix(self, inVector1: np.array, inVector2: np.array)->np.array:
        arrPeriodicDistance = np.zeros([len(inVector1), len(inVector2)])
        for j in range(len(inVector1)):
            for k in range(j,len(inVector2)):
                arrPeriodicDistance[j,k] = self.PeriodicMinimumDistance(inVector1[j],inVector2[k])
        if np.shape(arrPeriodicDistance)[0] == np.shape(arrPeriodicDistance)[1]:
            return arrPeriodicDistance + arrPeriodicDistance.T - np.diag(arrPeriodicDistance.diagonal())
        else:
            return arrPeriodicDistance
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        arrVectorPeriodic = self.PeriodicEquivalents(np.abs(inVector1-inVector2))
        return np.min(np.linalg.norm(arrVectorPeriodic, axis=1))
    def FindNonGrainMean(self, inPoint: np.array, fltRadius: float): 
        lstPointsIndices = []
        lstPointsIndices = self.FindCylindricalAtoms(self.GetNonLatticeAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        lstPointsIndices = list(np.unique(lstPointsIndices))
        #arrPoints = self.GetRows(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
        arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
        #for j in range(len(arrPoints)):
        #    arrPoints[j] = self.PeriodicShiftCloser(inPoint, arrPoints[j])
        arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
        if len(arrPoints) ==0:
            return inPoint
        else:
           return np.median(arrPoints, axis=0)           
    def FindCylindricalAtoms(self,arrPoints, arrCentre: np.array, fltRadius: float, fltHeight: float, blnPeriodic =True)->list: #arrPoints are [atomId, x,y,z]
        lstIndices = []
        if blnPeriodic:
            arrCentres = self.PeriodicEquivalents(arrCentre)
            for j in arrCentres:
                lstIndices.extend(gf.CylindricalVolume(arrPoints[:,1:4],j,fltRadius,fltHeight))
        else:
            lstIndices.extend(gf.CylindricalVolume(arrPoints[:,1:4],arrCentre,fltRadius,fltHeight))
        lstIndices = list(np.unique(lstIndices))
        return list(arrPoints[lstIndices,0])
    def FindBoxAtoms(self, arrPoints: np.array, arrCentre: np.array, arrLength: np.array, arrWidth: np.array,
    arrHeight: np.array, blnPeriodic = True)->list:
        lstIndices = []
        if blnPeriodic: 
            arrCentres = self.PeriodicEquivalents(arrCentre)
            for j in arrCentres:
                lstIndices.extend(gf.ParallelopipedVolume(arrPoints[:,1:4],j, arrLength, arrWidth, arrHeight))
        else:
            lstIndices.extend(gf.ParallelopipedVolume(arrPoints[:,1:4],arrCentre, arrLength, arrWidth, arrHeight))
        lstIndices = list(np.unique(lstIndices))
        return list(arrPoints[lstIndices,0])
    def FindValuesInBox(self, arrPoints: np.array, arrCentre: np.array, arrLength: np.array, arrWidth: np.array, 
    arrHeight: np.array, intColumn: int):
        lstIDs = self.FindBoxAtoms(arrPoints, arrCentre, arrLength, arrWidth,arrHeight)
        return self.GetAtomsByID(lstIDs)[:,intColumn]
    def FindValuesInCylinder(self, arrPoints: np.array ,arrCentre: np.array, fltRadius: float, fltHeight: float, intColumn: int): 
        lstIDs = self.FindCylindricalAtoms(arrPoints, arrCentre, fltRadius, fltHeight)
        #for j in lstIndices:
        #    lstRows.extend(np.where(np.linalg.norm(self.GetAtomData()[:,1:4] - arrPoints[j],axis=1)==0)[0])
        #lstRows = list(np.unique(lstRows))
        return self.GetAtomsByID(lstIDs)[:, intColumn]

class LAMMPSAnalysis(LAMMPSPostProcess):
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int):
        LAMMPSPostProcess.__init__(self, fltTimeStep,intNumberOfAtoms, intNumberOfColumns, lstColumnNames, lstBoundaryType, lstBounds,intLatticeType)
        self.__GrainBoundaries = []
        self.__lstMergedGBs = []
        self.__lstMergedTripleLines = []
    def FindTripleLineEnergy(self, intTripleLine: int, fltIncrement: float, fltWidth: float):
        lstOfGBs = self.GetNeighbouringGrainBoundaries(intTripleLine)
       # lstEquivalentTripleLines = self.GetEquivalentTripleLines(intTripleLine)
       # lstOfTripleLines = list(range(self.GetNumberOfTripleLines))
        # for j in lstEquivalentTripleLines:
        #     lstOfTripleLines.remove(j) 
        # fltClosest = np.min(self.__TripleLineDistanceMatrix[intTripleLine,lstOfTripleLines])/2
        lstV = [3*[]]
        lstR = [3*[]]
        lstI = [3*[]]
        lstFit = [3*[]]
        for j in range(len(lstOfGBs)):
            lstR[j],lstV[j],lstI[j] = self.FindGrainStrip(intTripleLine,lstOfGBs[j], lstOfGBs[np.mod(j+1,3)],fltWidth,fltIncrement)
            popt, poptv = optimize.curve_fit(gf.LinearRule, lstR,lstV)
            for r in lstR[j]:
                lstFit[j].append(gf.LinearRule(r, *popt))
    def MergePeriodicTripleLines(self, fltDistanceTolerance: float):
        lstMergedIndices = []
        setIndices = set(range(self.GetNumberOfTripleLines()))
        lstCurrentIndices = []
        arrPeriodicDistanceMatrix = self.MakePeriodicDistanceMatrix(self.__TripleLines,self.__TripleLines)
        while len(setIndices) > 0:
            lstCurrentIndices = list(*np.where(arrPeriodicDistanceMatrix[setIndices.pop()] < fltDistanceTolerance))
            lstMergedIndices.append(lstCurrentIndices)
            setIndices = setIndices.difference(lstCurrentIndices)
        self.__lstMergedTripleLines = lstMergedIndices
        return lstMergedIndices
    def FindTripleLines(self,fltGridLength: float, fltSearchRadius: float, intMinCount: int):
        lstGrainBoundaries = []
        lstGrainBoundaryObjects = []
        fltMidHeight = self.CellHeight/2
        objQPoints = QuantisedRectangularPoints(self.GetNonLatticeAtoms()[:,self._intPositionX:self._intPositionY+1],self.GetUnitBasisConversions()[0:2,0:2],9,fltGridLength/4, intMinCount)
        self.__TripleLines = objQPoints.FindTriplePoints()
        self.__TripleLineDistanceMatrix = spatial.distance_matrix(self.__TripleLines[:,0:2], self.__TripleLines[:,0:2])
        self.__TripleLines[:,2] = fltMidHeight*np.ones(len(self.__TripleLines))
        lstGrainBoundaries = objQPoints.GetGrainBoundaries()
        for j,arrGB in enumerate(lstGrainBoundaries):
            arrFirstPoint = arrGB[0]
            for k in range(len(arrGB)):
                arrPoint = np.array([arrGB[k,0], arrGB[k,1],fltMidHeight])
                arrPoint[2] = fltMidHeight
                lstGrainBoundaries[j][k] = self.PeriodicShiftCloser(arrFirstPoint, arrPoint)
            lstGrainBoundaryObjects.append(gl.GrainBoundary(lstGrainBoundaries[j]))
            #self.NudgeGrainBoundary(lstGrainBoundaryObjects[j],fltGridLength)
        self.__GrainBoundaries = lstGrainBoundaryObjects
        for i  in range(len(self.__TripleLines)):
           # self.__TripleLines[i] = self.FindNonGrainMean(self.__TripleLines[i], fltSearchRadius)
            self.__TripleLines[i] = self.MoveTripleLine(i)
        self.__TripleLines[:,2] = fltMidHeight*np.ones(len(self.__TripleLines))
        return self.__TripleLines
    def NudgeGrainBoundary(self, objGrainBoundary: gl.GrainBoundary, fltLatticeParameter: float)->np.array:
        arrGBDirection = gf.NormaliseVector(objGrainBoundary.GetLinearDirection())
        arrAcross = gf.NormaliseVector(np.cross(arrGBDirection, np.array([0,0,1]))) 
        lstIDs = []
        for index, j in enumerate(objGrainBoundary.GetPoints()):
            lstIDs = self.FindBoxAtoms(self.GetNonLatticeAtoms()[:,0:4], j-arrGBDirection/2,arrGBDirection*fltLatticeParameter/2, 3*fltLatticeParameter*arrAcross,self.GetCellVectors()[2])
            arrPoints = self.GetAtomsByID(lstIDs)[:,1:4]
            arrPoints = self.PeriodicShiftAllCloser(j, arrPoints)
            if len(arrPoints) > 0:
                arrMean = np.mean(arrPoints, axis=0)
                arrShift = np.dot(arrMean-j, arrAcross)*arrAcross
                objGrainBoundary.ShiftPoint(index,arrShift)
    def MoveTripleLine(self,intTripleLine:int):
        lstGBs = self.GetNeighbouringGrainBoundaries(intTripleLine)
        lstPoints = []
        for j in lstGBs:
            fltDistances = []
            for k in self.GetGrainBoundaries(j).GetPoints():
                fltDistances.append(self.PeriodicMinimumDistance(k, self.GetTripleLines(intTripleLine)))
            lstPoints.append(self.GetGrainBoundaries(j).GetPoints(np.argmin(fltDistances)))
        arrPoints = self.PeriodicShiftAllCloser(self.GetTripleLines(intTripleLine),np.array(lstPoints))
        return gf.EquidistantPoint(*arrPoints)
    def GetGrainBoundaries(self, intValue = None):
        if intValue is None:
            return self.__GrainBoundaries
        else:
            return self.__GrainBoundaries[intValue]
    def GetNumberOfGrainBoundaries(self)->int:
        return len(self.__GrainBoundaries)
    def GetNumberOfTripleLines(self)->int:
        return len(self.__TripleLines)
    def GetTripleLines(self, intValue = None)->np.array:
        if intValue is None:
            return self.__TripleLines
        else:
            return self.__TripleLines[intValue]
    def SetTripleLine(self, intPosition: np.array, arrValue: np.array):
        self.__TripleLines[intPosition] = arrValue
    def GetNeighbouringGrainBoundaries(self, intTripleLine: int):
        lstDistances = [] #the closest distance 
        lstPositions = []
        arrTripleLine = self.__TripleLines[intTripleLine]
        for arrGB in self.__GrainBoundaries:
            lstTemporary = []
            for j in arrGB.GetPoints(): 
                lstTemporary.append(self.PeriodicMinimumDistance(j ,arrTripleLine))
            lstDistances.append(np.min(lstTemporary))
        intCounter = 0
        while len(lstPositions) < 3 and intCounter < self.GetNumberOfGrainBoundaries():
            lstValues = gf.FindNthSmallestPosition(lstDistances,intCounter)
            lstPositions.extend(lstValues)
            lstPositions = list(np.unique(lstPositions))
            intCounter += 1
        return lstPositions
    def GetGrainBoundaryDirection(self, intGrainBoundary:int, intTripleLine: int):
        fltStart = self.PeriodicMinimumDistance(self.__GrainBoundaries[intGrainBoundary].GetPoints(0), self.__TripleLines[intTripleLine])
        fltEnd = self.PeriodicMinimumDistance(self.__GrainBoundaries[intGrainBoundary].GetPoints(-1), self.__TripleLines[intTripleLine])
        vctDirection = self.__GrainBoundaries[intGrainBoundary].GetLinearDirection()
        if fltEnd < fltStart:
            vctDirection = -vctDirection
        return vctDirection
    def FindGBStrip(self, intGrainBoundaryNumber: int, fltProportion: float,  fltLength: float,fltWidth: float, fltIncrement:float, strValue = 'sum'):
        lstLength = []
        lstValues = []
        lstIndices = []
        objGrainBoundary = self.GetGrainBoundaries(intGrainBoundaryNumber)
        arrGBDirection = gf.NormaliseVector(objGrainBoundary.GetLinearDirection())
        arrCrossVector = gf.NormaliseVector(objGrainBoundary.GetAcrossVector())
        intGBNumber = int(np.round(fltProportion*(objGrainBoundary.GetNumberOfPoints())))
        arrCentre = objGrainBoundary.GetPoints(intGBNumber)
        intMax = np.round(fltLength/fltIncrement).astype('int')
        for j in range(1, intMax):
            l = fltIncrement*j
            lstLength.append(l)
            lstIndices.extend(self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrCentre,l*arrGBDirection, 
                                                           fltWidth*arrCrossVector,np.array([0,0,self.CellHeight])))
            lstIndices.extend(self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrCentre,-l*arrGBDirection, 
                                                           fltWidth*arrCrossVector,np.array([0,0,self.CellHeight])))                                               
            lstIndices = list(np.unique(lstIndices))
            if strValue == 'mean':
                lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
            elif strValue =='sum':
                lstValues.append(np.sum(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
        return lstLength, lstValues,lstIndices 
    def FindGrainStrip(self, intTripleLine: int,intGrainIndex1:int, intGrainIndex2: int, fltWidth: float,fltIncrement:float,strValue = 'mean',fltLength = None):
        lstRadii = []
        lstValues = []
        lstIndices = []
        if fltLength is None:
            fltClosest = np.sort(self.__TripleLineDistanceMatrix[intTripleLine])[1]/2
        else:
            fltClosest = fltLength
        intMax = np.floor(fltClosest/(fltIncrement)).astype('int')
        arrMeanVector1 = self.PeriodicShiftCloser(self.__TripleLines[intTripleLine], np.mean(self.GetGrainBoundaries(intGrainIndex1).GetPoints(),axis=0)) -self.__TripleLines[intTripleLine]
        arrMeanVector2 = self.PeriodicShiftCloser(self.__TripleLines[intTripleLine], np.mean(self.GetGrainBoundaries(intGrainIndex2).GetPoints(),axis=0)) -self.__TripleLines[intTripleLine]
        arrVector = gf.NormaliseVector(arrMeanVector1+arrMeanVector2)
        for j in range(1,intMax):
            r = fltIncrement*j
            lstRadii.append(r)
            lstIndices.extend(self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           self.__TripleLines[intTripleLine],r*arrVector, 
                                                           fltWidth*np.cross(arrVector,np.array([0,0,1])),np.array([0,0,self.CellHeight])))
            lstIndices = list(np.unique(lstIndices))
            if strValue == 'mean':
                lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
            elif strValue =='sum':
                lstValues.append(np.sum(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
        return lstRadii, lstValues,lstIndices 
    def FindThreeGrainStrips(self, intTripleLine: int,fltWidth: float, fltIncrement: float, strValue = 'mean',fltLength = None):
        lstNeighbouringGB = self.GetNeighbouringGrainBoundaries(intTripleLine)
        lstOfVectors = [] #unit vectors that bisect the grain boundary directions
        lstValues = []
        lstRadii = []
        lstIndices  = []
        if fltLength is None:
            fltClosest = np.sort(self.__TripleLineDistanceMatrix[intTripleLine])[1]/2
        else:
            fltClosest = fltLength
        intMax = np.floor(fltClosest/(fltIncrement)).astype('int')
        for intV in range(3):
            #lstOfVectors.append(self.GetGrainBoundaryDirection(lstNeighbouringGB[intV],intTripleLine))
            arrMeanVector = np.mean(self.PeriodicShiftAllCloser(self.__TripleLines[intTripleLine],self.GetGrainBoundaries(lstNeighbouringGB[intV]).GetPoints()),axis=0)
            lstOfVectors.append(gf.NormaliseVector(arrMeanVector - self.__TripleLines[intTripleLine]))
        for j in range(1,intMax):
            r = fltIncrement*j
            lstRadii.append(r)
            for kVector in range(len(lstOfVectors)):
                v = gf.NormaliseVector(lstOfVectors[np.mod(kVector,3)] + lstOfVectors[np.mod(kVector+1,3)])
                lstIndices.extend(self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           self.__TripleLines[intTripleLine],r*v, 
                                                           fltWidth*np.cross(v,np.array([0,0,1])),np.array([0,0,self.CellHeight])))
                lstIndices = list(np.unique(lstIndices))
            if strValue == 'mean':
                lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
            elif strValue =='sum':
                lstValues.append(np.sum(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
        return lstRadii, lstValues,lstIndices  
    def GetEquivalentTripleLines(self, intTripleLine: int)->list:
        for lstTripleLines in self.__lstMergedTripleLines:
            if intTripleLine in lstTripleLines:
                return lstTripleLines
    

    
class QuantisedRectangularPoints(object): #linear transform parallelograms into a rectangular parameter space
    def __init__(self, in2DPoints: np.array, inUnitBasisVectors: np.array, n: int, fltGridSize: float, intMinCount: int):
        self.__WrapperWidth = n #mininum count specifies how many nonlattice atoms occur in the float grid size before it is counted as a grain boundary grid or triple line
        self.__BasisVectors = inUnitBasisVectors
        self.__InverseMatrix =  np.linalg.inv(inUnitBasisVectors)
        self.__GridSize = fltGridSize
        arrPoints =  np.matmul(in2DPoints, self.__BasisVectors)*(1/fltGridSize)
        #arrPoints[:,0] = np.linalg.norm(inBasisVectors[0]/fltGridSize, axis=0)*arrPoints[:,0]
        #arrPoints[:,1] = np.linalg.norm(inBasisVectors[1]/fltGridSize, axis=0)*arrPoints[:,1]
        intMaxHeight = np.round(np.max(arrPoints[:,0])).astype('int')
        intMaxWidth = np.round(np.max(arrPoints[:,1])).astype('int')
        self.__ArrayGrid =  np.zeros([(intMaxHeight+1),intMaxWidth+1])
        arrPoints = np.round(arrPoints).astype('int')
        for j in arrPoints:
            self.__ArrayGrid[j[0],j[1]] += 1 #this array represents the simultion cell
        self.__ArrayGrid = (self.__ArrayGrid >= intMinCount).astype('int')
        self.__ArrayGrid = ndimage.binary_dilation(self.__ArrayGrid, np.ones([3,3]))
        self.__ExtendedArrayGrid = np.zeros([np.shape(self.__ArrayGrid)[0]+2*n,np.shape(self.__ArrayGrid)[1]+2*n])
        self.__ExtendedArrayGrid[n:-n, n:-n] = self.__ArrayGrid
        self.__ExtendedArrayGrid[0:n, n:-n] = self.__ArrayGrid[-n:,:]
        self.__ExtendedArrayGrid[-n:, n:-n] = self.__ArrayGrid[:n,:]
        self.__ExtendedArrayGrid[:,0:n] = self.__ExtendedArrayGrid[:,-2*n:-n]
        self.__ExtendedArrayGrid[:,-n:] = self.__ExtendedArrayGrid[:,n:2*n]
        self.__ExtendedSkeletonGrid = skeletonize(self.__ExtendedArrayGrid).astype('int')
        #self.__ExtendedSkeletonGrid = thin(self.__ExtendedArrayGrid).astype('int')
        #self.__ExtendedSkeletonGrid = medial_axis(self.__ExtendedArrayGrid).astype('int')
        self.__GrainValue = 0
        self.__GBValue = 1 #just fixed constants used in the array 
        self.__DislocationValue = 2
        self.__TripleLineValue = 3
        self.__TriplePoints = []
        self.__Dislocations = []
        self.__GrainBoundaryLabels = []
        self.__blnGrainBoundaries = False #this flag is set once FindGrainBoundaries() is called
    def GetArrayGrid(self):
        return self.__ArrayGrid
    def GetExtendedArrayGrid(self)->np.array:
        return self.__ExtendedArrayGrid
    def GetExtendedSkeletonPoints(self)->np.array:
        return self.__ExtendedSkeletonGrid  
    def GetDislocations(self)->np.array:
        return self.__Dislocations 
    def ClassifyGBPoints(self,m:int,blnFlagEndPoints = False)-> np.array:
        self.__ResetSkeletonGrid() #resets the array so all GBs, dislocations and triplies lines are 1 and grain is 0 
        arrTotal =np.zeros(4*m)
        intLow = int((m-1)/2)
        intHigh = int((m+1)/2)
        arrArgList = np.argwhere(self.__ExtendedSkeletonGrid==self.__GBValue)
        arrCurrent = np.zeros([m,m])
        for x in arrArgList: #loop through the array positions which have GB atoms
            arrCurrent = self.__ExtendedSkeletonGrid[x[0]-intLow:x[0]+intHigh,x[1]-intLow:x[1]+intHigh] #sweep out a m x m square of array positions 
            intSwaps = 0
            if np.shape(arrCurrent) == (m,m): #centre j. This check avoids boundary points
                intValue = arrCurrent[0,0]
                arrTotal[:m ] = arrCurrent[0,:]
                arrTotal[m:2*m] =  arrCurrent[:,-1]
                arrTotal[2*m:3*m] = arrCurrent[-1,::-1]
                arrTotal[3*m:4*m] = arrCurrent[-1::-1,0]
                for k in arrTotal:
                    if (k!= intValue): #the move has changed from grain (int 0) to grain boundary (int 1) or vice versa
                        intSwaps += 1
                        intValue = k
                if intSwaps == 6 and m ==3:
                    if not (arrCurrent[0].all() == self.__GBValue or arrCurrent[-1].all() == self.__GBValue or arrCurrent[:,0].all() == self.__GBValue or  arrCurrent[:,-1].all() ==self.__GBValue):
                        self.SetSkeletonValue(x,self.__TripleLineValue)
                elif intSwaps ==6:
                    self.SetSkeletonValue(x,self.__TripleLineValue)
                elif intSwaps < 4 and blnFlagEndPoints and m==3: #only flag end points for a search with 3x3
                    self.SetSkeletonValue(x,self.__DislocationValue)
        self.__Dislocations = np.argwhere(self.__ExtendedSkeletonGrid == self.__DislocationValue)
        return np.argwhere(self.__ExtendedSkeletonGrid == self.__TripleLineValue)
    def SetSkeletonValue(self,inArray:np.array, intValue: int):
        self.__ExtendedSkeletonGrid[inArray[0], inArray[1]] = intValue
    def __ConvertToCoordinates(self, inArrayPosition: np.array): #takes array position and return real 2D coordinates
        arrPoints = (inArrayPosition - np.ones([2])*self.__WrapperWidth)*self.__GridSize
        arrPoints = np.matmul(arrPoints, self.__InverseMatrix)
        rtnArray = np.zeros([len(arrPoints),3])
        rtnArray[:,0:2] = arrPoints
        return rtnArray
        #return np.matmul())*self.__GridSize,self.__BasisVectors)
    def __ResetSkeletonGrid(self):
        self.__ExtendedSkeletonGrid[self.__ExtendedSkeletonGrid != self.__GrainValue] = self.__GBValue
    def FindDislocations(self):
        self.ClassifyGBPoints(3,True)
        return self.__Dislocations
    def FindTriplePoints(self)->np.array:
        self.__TriplePoints = self.ClassifyGBPoints(3, False)
        return self.__ConvertToCoordinates(self.__TriplePoints)
    def FindGrainBoundaries(self)->np.array:
        self.ClassifyGBPoints(5,True)
        k = self.__WrapperWidth 
        arrSkeleton = np.copy(self.__ExtendedSkeletonGrid)
        arrSkeleton[:k,:] = 0
        arrSkeleton[-k:,:] = 0
        arrSkeleton[:,:k] = 0
        arrSkeleton[:,-k:] = 0
        arrSkeleton = (arrSkeleton == self.__GBValue).astype('int')
        self.__GrainBoundaryLabels = label(arrSkeleton)
        lstUniqueIDs = list(np.unique(self.__GrainBoundaryLabels))
        lstUniqueIDs.remove(self.__GrainValue)
        intCounter = 0
        while intCounter < len(lstUniqueIDs):
            intID = lstUniqueIDs[intCounter]
            if len(np.argwhere(self.__GrainBoundaryLabels == intID)) < 3:
                self.__GrainBoundaryLabels[self.__GrainBoundaryLabels == intID] = self.__GrainValue
                lstUniqueIDs.remove(intID)
            else:
                intCounter += 1
        self.__GrainBoundaryIDs = lstUniqueIDs
        self.__NumberOfGrainBoundaries = len(lstUniqueIDs)   
        self.__blnGrainBoundaries = True
        self.MergeGrainBoundaries() #merges periodically equivalent grain boundaries
        self.ExtendGrainBoundaries() #adds one more GB points between the end of the grain boundary and the triple line
    def GetGrainBoundaryLabels(self):
        if  not self.__blnGrainBoundaries:
            self.FindGrainBoundaries()
        return self.__GrainBoundaryLabels
    def GetNumberOfGrainBoundaries(self):
        if not self.__blnGrainBoundaries:
            self.FindGrainBoundaries()
        return len(self.__GrainBoundaryIDs)
    def GetGrainBoundaries(self)->np.array:
        lstGrainBoundaries = []
        if not self.__blnGrainBoundaries:
            self.FindGrainBoundaries()
        for j in self.__GrainBoundaryIDs:
            arrPoints = np.argwhere(self.__GrainBoundaryLabels == j) 
            lstGrainBoundaries.append(self.__ConvertToCoordinates(arrPoints))
        return lstGrainBoundaries
    def MergeGrainBoundaries(self):
        arrMod = np.array([np.shape(self.__ArrayGrid)])
        lstCurrentIDs = list(np.copy(self.__GrainBoundaryIDs))
        counter = 0
        while (counter < len(self.__GrainBoundaryIDs)):
            j = self.__GrainBoundaryIDs[counter]
            lstCurrentIDs.remove(j)
            arrPointsj = np.argwhere(self.__GrainBoundaryLabels == j)
            arrPointsj = np.fmod(arrPointsj, arrMod)
            for k in lstCurrentIDs:
                arrPointsk = np.argwhere(self.__GrainBoundaryLabels == k)
                arrPointsk = np.fmod(arrPointsk, arrMod)
                arrDistanceMatrix = spatial.distance_matrix(arrPointsj, arrPointsk)
                if np.min(arrDistanceMatrix) < 2: #the two grain boundaries periodically link
                    self.__GrainBoundaryLabels[self.__GrainBoundaryLabels == k] = j
                    self.__GrainBoundaryIDs.remove(k)
                    lstCurrentIDs.remove(k)
            counter +=1
    def ExtendGrainBoundaries(self):
        arrPoints = np.argwhere(self.__ExtendedSkeletonGrid == self.__GBValue)
        for j in self.__GrainBoundaryIDs:
            arrGBPoints = np.argwhere(self.__GrainBoundaryLabels ==j)
            for k in arrPoints:
                if np.any(np.linalg.norm(arrGBPoints-k,axis=1) < 2):
                    self.__GrainBoundaryLabels[k[0],k[1]] = j

      