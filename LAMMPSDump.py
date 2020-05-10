import re
import numpy as np
import GeometryFunctions as gf
import GeneralLattice as gl
from scipy import spatial, optimize, ndimage
from skimage.morphology import skeletonize, thin, medial_axis, remove_small_holes
from scipy.cluster.vq import kmeans,vq
from skimage.filters import gaussian
import shapely as sp
import geopandas as gpd
import copy

class LAMMPSData(object):
    def __init__(self,strFilename: str, intLatticeType: int, fltLatticeParameter):
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
                objTimeStep = LAMMPSAnalysis(timestep, N,intNumberOfColumns,lstColumnNames, lstBoundaryType, lstBounds,intLatticeType, fltLatticeParameter)
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
         #   for k in range(j,len(inVector2)):
            for k in range(len(inVector2)):
                arrPeriodicDistance[j,k] = self.PeriodicMinimumDistance(inVector1[j],inVector2[k])
        #if np.shape(arrPeriodicDistance)[0] == np.shape(arrPeriodicDistance)[1]:
        #    return arrPeriodicDistance + arrPeriodicDistance.T - np.diag(arrPeriodicDistance.diagonal())
        #else:
        return arrPeriodicDistance
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        #arrVectorPeriodic = self.PeriodicEquivalents(np.abs(inVector1-inVector2))
        inVector2 = self.PeriodicShiftCloser(inVector1, inVector2)
        return np.linalg.norm(inVector2-inVector1, axis=0)
    def FindNonGrainMediod(self, inPoint: np.array, fltRadius: float, bln2D= True):
        arrReturn = np.ones(3)*self.CellHeight/2
        lstPointsIndices = []
        lstPointsIndices = self.FindCylindricalAtoms(self.GetNonLatticeAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
        #arrPoints = self.GetRows(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
        #for j in range(len(arrPoints)):
        #    arrPoints[j] = self.PeriodicShiftCloser(inPoint, arrPoints[j])
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            arrPoint = gf.FindGeometricMediod(arrPoints, bln2D)
            arrReturn[0:2] = arrPoint
            return arrReturn  
        else:
            return None

    def FindNonGrainMean(self, inPoint: np.array, fltRadius: float): 
        lstPointsIndices = []
        lstPointsIndices = self.FindCylindricalAtoms(self.GetNonLatticeAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
        #arrPoints = self.GetRows(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
        #for j in range(len(arrPoints)):
        #    arrPoints[j] = self.PeriodicShiftCloser(inPoint, arrPoints[j])
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            return np.mean(arrPoints, axis=0)  
        else:
            return None
    def FindGrainMean(self, inPoint: np.array, fltRadius: float): 
        lstPointsIndices = []
        lstPointsIndices = self.FindCylindricalAtoms(self.GetLatticeAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            return np.mean(arrPoints, axis=0)  
        else:
            return inPoint         
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
    def FindBoxAtoms(self, arrPoints: np.array, arrStart: np.array, arrLength: np.array, arrWidth: np.array,
    arrHeight: np.array, blnPeriodic = True)->list:
        lstIndices = []
        if blnPeriodic: 
            arrStarts = self.PeriodicEquivalents(arrStart)
            for j in arrStarts:
                lstIndices.extend(gf.ParallelopipedVolume(arrPoints[:,1:4],j, arrLength, arrWidth, arrHeight))
        else:
            lstIndices.extend(gf.ParallelopipedVolume(arrPoints[:,1:4],arrStart, arrLength, arrWidth, arrHeight))
        lstIndices = list(np.unique(lstIndices))
        return list(arrPoints[lstIndices,0])
    def FindValuesInBox(self, arrPoints: np.array, arrCentre: np.array, arrLength: np.array, arrWidth: np.array, 
    arrHeight: np.array, intColumn: int):
        lstIDs = self.FindBoxAtoms(arrPoints, arrCentre, arrLength, arrWidth,arrHeight)
        return self.GetAtomsByID(lstIDs)[:,intColumn]
    def FindValuesInCylinder(self, arrPoints: np.array ,arrCentre: np.array, fltRadius: float, fltHeight: float, intColumn: int): 
        lstIDs = self.FindCylindricalAtoms(arrPoints, arrCentre, fltRadius, fltHeight)
        return self.GetAtomsByID(lstIDs)[:, intColumn]
    def FindCylindricalSegmentAtoms(self, arrPoints: np.array, arrCentre: np.array, arrVector1: np.array,
    arrVector2: np.array, fltRadius: float,fltHeight = None, blnPeriodic = True):
        if fltHeight is None:
            fltHeight = self.CellHeight
        lstIndices = []
        if blnPeriodic:
            arrStarts = self.PeriodicEquivalents(arrCentre)
            for j in arrStarts:
                lstIndices.extend(gf.ArcSegment(arrPoints[:,1:4],j, arrVector1, arrVector2, fltRadius, fltHeight))
        else:
            lstIndices.extend(gf.ArcSegment(arrPoints[:,1:4], arrCentre, arrVector1, arrVector2, fltRadius,fltHeight))
        return list(arrPoints[lstIndices,0])
class LAMMPSAnalysis(LAMMPSPostProcess):
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int, fltLatticeParameter: float):
        LAMMPSPostProcess.__init__(self, fltTimeStep,intNumberOfAtoms, intNumberOfColumns, lstColumnNames, lstBoundaryType, lstBounds,intLatticeType)
        self.__GrainBoundaries = dict()
        self.__UniqueGrainBoundaries = dict()
        self.__MergedTripleLines = dict()
        self.__UniqueTripleLines = dict() #periodically equivalent triple lines are merged into a single point
        self.__TripleLines = dict()
        self.__LatticeParameter = fltLatticeParameter
    def SetLatticeParameter(self, fltParameter: float):
        self.__LatticeParameter = fltParameter
    def __Reciprocal(self, r,a,b,c): 
        return a/(r+b)+c
    def FindGBAtoms(self,strGBID: str, fltWidth: float,fltSeparation:float, blnRemoveTripleLines = True):
        lstIndices = []
        lstRemove = []
        arrGB3d =  self.GetUniqueGrainBoundaries(strGBID).GetPoints(fltSeparation, True)
        fltLength = 0
        for k in range(1,len(arrGB3d)):
            arrLength = arrGB3d[k]-arrGB3d[k-1]
            fltLength += np.linalg.norm(arrLength)
            arrWidth = fltWidth*gf.NormaliseVector(np.cross(arrLength,np.array([0,0,1])))
            if k != 1:
                lstIndices.extend(self.FindCylindricalSegmentAtoms(self.GetAtomData()[:,0:4], arrGB3d[k], 
                                                    arrPreviousWidth, arrWidth,fltWidth/2,self.CellHeight))
                lstIndices.extend(self.FindCylindricalSegmentAtoms(self.GetAtomData()[:,0:4], arrGB3d[k], 
                                                    -arrPreviousWidth, -arrWidth,fltWidth/2,self.CellHeight))
            lstIndices.extend(self.FindBoxAtoms(self.GetAtomData()[:,0:4], arrGB3d[k-1], 
                                                               arrLength, arrWidth,self.GetCellVectors()[2,:]))
            arrPreviousWidth = arrWidth                                                               
        setIndices = set(lstIndices)
        if blnRemoveTripleLines:
            lstTJs = self.GetUniqueGrainBoundaries(strGBID).GetID()
            for j in lstTJs:
                lstRemove.extend(self.GetUniqueTripleLines(j).GetAtomIDs())
                fltLength -= self.GetUniqueTripleLines(j).GetRadius()
            setIndices = setIndices.difference(lstRemove)
        lstIndices = list(setIndices)
        return lstIndices, fltLength    
    def FindTripleLineEnergy(self, strTripleLineID: str, fltIncrement: float, fltWidth: float,fltMinimumLatticeValue = -3.3600000286, fltTolerance = 0.005):
        lstL = []
        lstV = []
        lstI = []
        lstOfVectors = []
        fltEnergy = 0
        arrCentre = self.GetUniqueTripleLines(strTripleLineID).GetCentre()
        fltLength = self.FindClosestTripleLine(strTripleLineID)
        for strGB in self.GetUniqueTripleLines(strTripleLineID).GetUniqueAdjacentGrainBoundaries():
            arrVector = self.__UniqueGrainBoundaries[strGB].GetVectorDirection(strTripleLineID, self.__LatticeParameter, bln3D=True) 
            lstOfVectors.append(gf.NormaliseVector(arrVector))
        arrVectors  = np.vstack(lstOfVectors)
        n = len(arrVectors)
        arrDisplacements = np.zeros([n,3])
        for k in range(n):
                v = gf.NormaliseVector(arrVectors[np.mod(k,n)] + arrVectors[np.mod(k+1,n)])
                lstL, lstV, lstI = self.FindStrip(arrCentre, v, fltWidth, fltIncrement, fltLength)
                intStart = len(lstV) - np.argmax(lstV[-1:0:-1])
                blnValueError = False
                popt = np.zeros([3]) #this is set incase the next step raises an error
                if len(lstL[intStart:]) > 2:
                    try:
                        popt = optimize.curve_fit(self.__Reciprocal, lstL[intStart:],lstV[intStart:])[0]
                    except RuntimeError:
                        blnValueError = True
                    while ((np.abs((popt[2] - fltMinimumLatticeValue)/fltMinimumLatticeValue) > fltTolerance) and (intStart < len(lstL)-3) and not(blnValueError)):
                        intStart += 1
                        try:
                            popt = optimize.curve_fit(self.__Reciprocal, lstL[intStart:],lstV[intStart:])[0]
                        except RuntimeError:
                            blnValueError = True
                    fltDistance = lstL[intStart]
                    arrDisplacements[k] = arrCentre + fltDistance*v 
        arrCentre = gf.EquidistantPoint(*arrDisplacements)
        fltRadius = np.linalg.norm(arrDisplacements[0]-arrCentre)
        self.__UniqueTripleLines[strTripleLineID].SetCentre(arrCentre)
        self.__UniqueTripleLines[strTripleLineID].SetRadius(fltRadius)
        lstTJIDs = self.FindCylindricalAtoms(self.GetAtomData()[:,0:4],arrCentre,fltRadius,self.CellHeight)
        intTJAtoms = len(lstTJIDs)
        if intTJAtoms >0:
            self.__UniqueTripleLines[strTripleLineID].SetAtomIDs(lstTJIDs)            
            arrTJValues = self.GetAtomsByID(lstTJIDs)[:, self._intPE]
            fltEnergy = np.mean(arrTJValues)
            return fltEnergy, fltRadius, intTJAtoms
        else:
            return 0,0,0

    def FindTripleLineEnergyPerVolume(self, strTripleLineID: str, fltIncrement: float, fltWidth: float,fltMinimumLatticeValue = -3.3600000286, fltTolerance = 0.01):
        fltMinimumLatticeValue = fltMinimumLatticeValue*4/(self.__LatticeParameter**3) 
        lstR = []
        lstV = []
        lstI = []
        fltEnergy = 0
        lstR,lstV,lstI = self.FindThreeGrainStrips(strTripleLineID,fltWidth,fltIncrement, 'vnolume')
        intStart = 1
        #fltMeanLatticeValue = np.mean(self.GetLatticeAtoms()[:,self._intPE]) 
        blnValueError = False
        popt = np.zeros([3]) #this is set incase the next step raises an error
        if len(lstR[intStart:]) > 2 and len(lstV[intStart:]) >2:
            try:
                popt = optimize.curve_fit(self.__Reciprocal, lstR[intStart:],lstV[intStart:])[0]
            except RuntimeError:
                blnValueError = True
           # while (np.abs((popt[0]/popt[1] +fltMeanLatticeValue)/fltMeanLatticeValue) > 0.001 and intStart < len(lstR)-3): #check to see if the fit is good if not move along one increment
            while ((np.abs((popt[2] - fltMinimumLatticeValue)/fltMinimumLatticeValue) > fltTolerance) and (intStart < len(lstR)-3) and not(blnValueError)):
                intStart += 1
                try:
                    popt = optimize.curve_fit(self.__Reciprocal, lstR[intStart:],lstV[intStart:])[0]
                except RuntimeError:
                    blnValueError = True
        if intStart >= len(lstR):
            return 0,0,0
        else:
            fltRadius = lstR[intStart]
            lstTJIDs = self.FindCylindricalAtoms(self.GetAtomData()[:,0:4],self.GetUniqueTripleLines(strTripleLineID).GetCentre(),fltRadius,self.CellHeight)
            arrTJValues = self.GetAtomsByID(lstTJIDs)[:, self._intPE]
            intTJAtoms = len(lstTJIDs)
            self.__UniqueTripleLines[strTripleLineID].SetAtomIDs(lstTJIDs)
            self.__UniqueTripleLines[strTripleLineID].SetRadius(fltRadius)
            if intTJAtoms > 0: 
                fltEnergy = np.sum(arrTJValues)/(np.pi*lstR[intStart]**2*self.CellHeight)
            return fltEnergy,fltRadius, intTJAtoms
    def FindTripleLineEnergyPerAtom(self, strTripleLineID: str, fltIncrement: float, fltWidth: float,fltMinimumLatticeValue = -3.3600000286, fltTolerance = 0.01):
        lstR = []
        lstV = []
        lstI = []
        lstN = []
        fltEnergy = 0
        lstR,lstV,lstI, lstN = self.FindThreeGrainStrips(strTripleLineID,fltWidth,fltIncrement, 'mean', blnReturnNumbers=True)
        intStart = len(lstV) - np.argmax(lstV[-1:0:-1]) #find the max value position counting backwards as the first max is used
        #fltMeanLatticeValue = np.mean(self.GetLatticeAtoms()[:,self._intPE]) 
        blnValueError = False
        popt = np.zeros([3]) #this is set incase the next step raises an error
        if len(lstR[intStart:]) > 2 and len(lstV[intStart:]) >2:
            try:
                popt = optimize.curve_fit(self.__Reciprocal, lstN[intStart:],lstV[intStart:])[0]
            except RuntimeError:
                blnValueError = True
           # while (np.abs((popt[0]/popt[1] +fltMeanLatticeValue)/fltMeanLatticeValue) > 0.001 and intStart < len(lstR)-3): #check to see if the fit is good if not move along one increment
            while ((np.abs((popt[2] - fltMinimumLatticeValue)/fltMinimumLatticeValue) > fltTolerance) and (intStart < len(lstR)-3) and not(blnValueError)):
                intStart += 1
                try:
                    popt = optimize.curve_fit(self.__Reciprocal, lstN[intStart:],lstV[intStart:])[0]
                except RuntimeError:
                    blnValueError = True
        if intStart >= len(lstR):
            return 0,0,0
        else:
            fltRadius = lstR[intStart]
            lstTJIDs = self.FindCylindricalAtoms(self.GetAtomData()[:,0:4],self.GetUniqueTripleLines(strTripleLineID).GetCentre(),fltRadius,self.CellHeight)
            arrTJValues = self.GetAtomsByID(lstTJIDs)[:, self._intPE]
            intTJAtoms = len(lstTJIDs)
            self.__UniqueTripleLines[strTripleLineID].SetAtomIDs(lstTJIDs)
            self.__UniqueTripleLines[strTripleLineID].SetRadius(fltRadius)
            if intTJAtoms > 0: 
                fltEnergy = np.mean(arrTJValues)
            return fltEnergy,fltRadius, intTJAtoms
    def GetUniqueTripleLineIDs(self):
        return sorted(list(self.__UniqueTripleLines.keys()),key = lambda x: int(x[3:]))
    def GetUniqueGrainBoundaryIDs(self):
        return sorted(list(self.__UniqueGrainBoundaries.keys()),key = lambda x: 10*x.split(',')[0][3:] + x.split(',')[1][3:])    
    def GetGrainBoundaryIDs(self):
        return sorted(list(self.__GrainBoundaries.keys()),key = lambda x: 10*x.split(',')[0][2:] + x.split(',')[1][2:])
    def GetUniqueTripleLines(self, strID = None)->gl.TripleLine:
        if strID is None:
            return self.__UniqueTripleLines
        else:
            return self.__UniqueTripleLines[strID]
    def FindTripleLines(self,fltGridLength: float, fltSearchRadius: float, intMinCount: int, intDilation = 2):
        fltMidHeight = self.CellHeight/2
        objQPoints = QuantisedRectangularPoints(self.GetNonLatticeAtoms()[:,self._intPositionX:self._intPositionY+1],self.GetUnitBasisConversions()[0:2,0:2],10,fltGridLength, intMinCount, intDilation)
        arrTripleLines = objQPoints.GetTriplePoints()   
        arrTripleLines[:,2] = fltMidHeight*np.ones(len(arrTripleLines))
        for i  in range(len(arrTripleLines)):
            #arrPoint = self.FindNonGrainMean(arrTripleLines[i], fltSearchRadius)
            arrPoint = self.FindNonGrainMediod(arrTripleLines[i], fltSearchRadius)
            if arrPoint is not None:
                arrTripleLines[i] = arrPoint 
            arrTripleLines[i] = self.MoveTripleLine(arrTripleLines[i],fltSearchRadius)
            arrTripleLines[i,2] = fltMidHeight
            objTripleLine = gl.TripleLine('TJ'+str(i),arrTripleLines[i], self.GetCellVectors()[2,:])
            for j in objQPoints.GetAdjacentTriplePoints(i):
                objTripleLine.SetAdjacentTripleLines('TJ'+str(j))
            for k in objQPoints.GetEquivalentTriplePoints(i):
                 objTripleLine.SetEquivalentTripleLines('TJ'+str(k))
            self.__TripleLines[objTripleLine.GetID()] =objTripleLine
        # here the ith row corrsponds to the TJi triple line
    def MergePeriodicTripleLines(self, fltRadius:float): #finds equivalent and adjacent triplelines and sets
        setKeys = set(self.GetTripleLineIDs())
        counter = 0
        while len(setKeys) > 0: 
            objTripleLine = self.__TripleLines[setKeys.pop()] #take care not to set any properties on objTripleLine
            lstEquivalentTripleLines =  objTripleLine.GetEquivalentTripleLines()
            #lstEquivalentTripleLines.append(objTripleLine.GetID())
            setKeys = setKeys.difference(lstEquivalentTripleLines)
            arrTripleLines = np.zeros([len(lstEquivalentTripleLines),3])
            for intCounter,strTripleLine in enumerate(lstEquivalentTripleLines):
                arrTripleLines[intCounter] = self.__TripleLines[strTripleLine].GetCentre()
            arrCentre = self.MoveToSimulationCell(np.mean(self.PeriodicShiftAllCloser(arrTripleLines[0], arrTripleLines),axis=0))
            objUniqueTripleLine = gl.UniqueTripleLine('UTJ' + str(counter),arrCentre, objTripleLine.GetAxis())
            objUniqueTripleLine.SetEquivalentTripleLines(lstEquivalentTripleLines, False)
            for j in lstEquivalentTripleLines:
                objUniqueTripleLine.SetAdjacentTripleLines(self.__TripleLines[j].GetAdjacentTripleLines())
            self.__UniqueTripleLines[objUniqueTripleLine.GetID()] = objUniqueTripleLine
            counter +=1
        setUniqueKeys = set(self.GetUniqueTripleLineIDs())
        while len(setUniqueKeys) > 0:
            objUniqueTripleLine1 = self.__UniqueTripleLines[setUniqueKeys.pop()]
            setAdjacentTripleLines1 = set(objUniqueTripleLine1.GetAdjacentTripleLines())
            for strKey in self.GetUniqueTripleLineIDs():
                if strKey != objUniqueTripleLine1.GetID():
                    objUniqueTripleLine2 = self.__UniqueTripleLines[strKey]
                    setEquivalentTripleLines2 = objUniqueTripleLine2.GetEquivalentTripleLines()
                    setIntersection = setAdjacentTripleLines1.intersection(setEquivalentTripleLines2)
                    if len(setIntersection) > 0:
                        objUniqueTripleLine1.SetUniqueAdjacentTripleLines(objUniqueTripleLine2.GetID())
        lstSortedUniqueIDs = self.GetUniqueTripleLineIDs()
        arrUniqueTripleLines =np.zeros([len(lstSortedUniqueIDs),3])
        for strID in lstSortedUniqueIDs:
            arrCentre = self.MoveTripleLine(self.GetUniqueTripleLines(strID).GetCentre(),fltRadius)
            self.GetUniqueTripleLines(strID).SetCentre(arrCentre)
            arrUniqueTripleLines[int(strID[3:])] = self.GetUniqueTripleLines(strID).GetCentre()
        self.__PeriodicTripleLineDistanceMatrix = self.MakePeriodicDistanceMatrix(arrUniqueTripleLines, arrUniqueTripleLines)
    def MakeGrainBoundaries(self):
        setTJIDs = set(self.GetTripleLineIDs())
        counter = 0
        while len(setTJIDs) > 0:
            strCurrentTJ = setTJIDs.pop()
            lstAdjacentTripleLines = self.GetTripleLines(strCurrentTJ).GetAdjacentTripleLines()
            for j in lstAdjacentTripleLines:
                arrMovedTripleLine = self.PeriodicShiftCloser(self.GetTripleLines(strCurrentTJ).GetCentre(),self.GetTripleLines(j).GetCentre())
                arrLength = arrMovedTripleLine - self.GetTripleLines(strCurrentTJ).GetCentre()
                arrWidth = 25*np.cross(gf.NormaliseVector(arrLength), np.array([0,0,1]))
                arrPoints = self.FindValuesInBox(self.GetNonLatticeAtoms()[:,0:4], 
                self.GetTripleLines(strCurrentTJ).GetCentre(),arrLength,arrWidth,self.GetCellVectors()[:,2],[1,2,3])
                arrPoints = self.PeriodicShiftAllCloser(self.GetTripleLines(strCurrentTJ).GetCentre(), arrPoints)
                lstGBID = [strCurrentTJ ,j]
                if int(strCurrentTJ[2:]) > int(j[2:]): #this overwrites the same grainboundary and sorts ID with lowest TJ number first
                    lstGBID.reverse()
                    objGrainBoundary = gl.GrainBoundaryCurve(arrMovedTripleLine,self.GetTripleLines(strCurrentTJ).GetCentre(), lstGBID, arrPoints, self.CellHeight/2)
                else:
                    objGrainBoundary = gl.GrainBoundaryCurve(self.GetTripleLines(strCurrentTJ).GetCentre(),arrMovedTripleLine, lstGBID, arrPoints,self.CellHeight/2)
                strGBID = str(lstGBID[0]) + ',' + str(lstGBID[1])
                self.__GrainBoundaries[strGBID] = objGrainBoundary
                self.__TripleLines[strCurrentTJ].SetAdjacentGrainBoundaries(strGBID)
                self.__TripleLines[j].SetAdjacentGrainBoundaries(strGBID)
                counter += 1
        self.MakeUniqueGrainBoundaries()
    def MakeUniqueGrainBoundaries(self, lstTripleLineID = None):
        if lstTripleLineID is None:
            lstUTJIDs = self.GetUniqueTripleLineIDs()
        else: 
            lstUTJIDs = lstTripleLineID
        counter = 0
        while counter < len(lstUTJIDs):
            strCurrentTJ = lstUTJIDs[counter]
            lstAdjacentTripleLines = self.GetUniqueTripleLines(strCurrentTJ).GetUniqueAdjacentTripleLines()
            for j in lstAdjacentTripleLines:
                arrMovedTripleLine = self.PeriodicShiftCloser(self.GetUniqueTripleLines(strCurrentTJ).GetCentre(),self.GetUniqueTripleLines(j).GetCentre())
                arrLength = arrMovedTripleLine - self.GetUniqueTripleLines(strCurrentTJ).GetCentre()
                arrWidth = 5*self.__LatticeParameter*np.cross(gf.NormaliseVector(arrLength), np.array([0,0,1]))
                arrPoints = self.FindValuesInBox(self.GetNonLatticeAtoms()[:,0:4], 
                self.GetUniqueTripleLines(strCurrentTJ).GetCentre(),arrLength,arrWidth,self.GetCellVectors()[:,2],[1,2,3])
                arrPoints = self.PeriodicShiftAllCloser(self.GetUniqueTripleLines(strCurrentTJ).GetCentre(), arrPoints)       
                lstGBID = [strCurrentTJ,j]
                strGBID = str(lstGBID[0]) + ',' + str(lstGBID[1])
                strGBIDReverse = str(lstGBID[1]) + ',' + str(lstGBID[0])
                if (strGBID not in self.GetUniqueGrainBoundaryIDs()) and (strGBIDReverse not in self.GetUniqueGrainBoundaryIDs()):
                    objGrainBoundary = gl.GrainBoundaryCurve(self.GetUniqueTripleLines(strCurrentTJ).GetCentre(),arrMovedTripleLine, lstGBID, arrPoints,self.CellHeight/2)
                    self.__UniqueGrainBoundaries[strGBID] = objGrainBoundary
                    self.__UniqueTripleLines[strCurrentTJ].SetUniqueAdjacentGrainBoundaries(strGBID)
                    self.__UniqueTripleLines[j].SetUniqueAdjacentGrainBoundaries(strGBID)
            counter += 1
    def TriangulateCentre(self, inPoints: np.array, fltRadius: float)->np.array:
        arr2DPoints = np.unique(np.round(inPoints[:,0:2],3), axis = 0)
        if len(arr2DPoints) > 5: #here the convex hull normally consists of three line sections across GBS and a further three line segements that connect the GB line segments making an irregular hexagon
            arrConvexHull = spatial.ConvexHull(arr2DPoints).vertices
            intLength = len(arrConvexHull)
            arrDistances = np.zeros(intLength)
            for j in range(intLength):
                    arrDistances[j] = np.linalg.norm(arr2DPoints[arrConvexHull[np.mod(j+1,intLength)]]-arr2DPoints[arrConvexHull[j]])
            arrVectors =np.ones([3,3])
            if np.sort(arrDistances)[-2] > 2*self.__LatticeParameter:
                for k in range(3):
                    intGreatest = gf.FindNthLargestPosition(arrDistances, k)[0]
                    arrVectors[k,0:2] = (arr2DPoints[arrConvexHull[intGreatest]]+arr2DPoints[arrConvexHull[intGreatest-1]])/2
                arrPoint = gf.EquidistantPoint(*arrVectors)
                arrPoint[2] = self.CellHeight/2
                return arrPoint
            else:
                return None
      #  elif len(inPoints) > 0:
      #      return np.mean(inPoints, axis=0)
        else:
            return None
    def ConvexCentre(self, inPoints: np.array, fltRadius: float)->np.array:
        arrReturn = self.CellHeight/2*np.ones([3])
        arr2DPoints = np.unique(np.round(inPoints[:,0:2],3), axis = 0)
        if len(arr2DPoints) > 5: #here the convex hull normally consists of three line sections across GBS and a further three line segements that connect the GB line segments making an irregular hexagon
            arrConvexHull = spatial.ConvexHull(arr2DPoints).vertices
            arrReturn[:2] = np.mean(arr2DPoints[arrConvexHull],axis=0)
            return arrReturn
      #  elif len(inPoints) > 0:
      #      return np.mean(inPoints, axis=0)
        else:
            return None
    def MoveTripleLine(self, arrTripleLine, fltRadius)->np.array:
        arrPoint = self.FindNonGrainMediod(arrTripleLine, fltRadius)
        if arrPoint is None:
            arrPoint = arrTripleLine
        else:
            arrNextPoint = arrPoint
            arrNextPoint[2] = self.CellHeight/2
            fltRadius = self.__LatticeParameter
            arrNextPoint = self.FindNonGrainMediod(arrNextPoint,fltRadius)
        if arrNextPoint is None:
            return arrPoint
        else:
            arrNextPoint[2] = self.CellHeight/2
            return arrNextPoint
    def FindClosestGrainPoint(self, inPoint: np.array, fltRadius: float)->np.array:
        lstDistances = []
        arrPoints = self.FindValuesInCylinder(self.GetNonLatticeAtoms()[:,0:4],inPoint, fltRadius,self.CellHeight,[1,2,3])
       # arrMovedPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints
        for j in arrPoints:
            lstDistances.append(self.PeriodicMinimumDistance(inPoint, j))
        if len(lstDistances) > 0:
            intPosition = np.argmin(lstDistances)
            arrReturn =  self.PeriodicShiftCloser(inPoint,arrPoints[intPosition])
            return arrReturn
        else:
            return None
    def GetGrainBoundaries(self, strID = None):
        if strID is None:
            return self.__GrainBoundaries
        else:
            return self.__GrainBoundaries[strID]
    def GetUniqueGrainBoundaries(self, strID = None):
        if strID is None:
            return self.__UniqueGrainBoundaries
        else:
            return self.__UniqueGrainBoundaries[strID]        
    def GetNumberOfGrainBoundaries(self)->int:
        return len(self.__GrainBoundaries)
    def GetNumberOfTripleLines(self)->int:
        return len(self.__TripleLines)
    def GetTripleLineIDs(self):
        return sorted(list(self.__TripleLines.keys()),key = lambda x: int(x[2:]))
    def GetTripleLines(self, strID = None)->gl.TripleLine:
        if strID is None:
            return self.__TripleLines
        else:
            return self.__TripleLines[strID]
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
            lstIndices = list(np.unique(lstIndices))
            if strValue == 'mean':
                lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
            elif strValue =='sum':
                lstValues.append(np.sum(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
        return lstLength, lstValues,lstIndices
    def FindClosestTripleLine(self, strTripleLineID: str)->float:
        if strTripleLineID[:1] == 'U':
            intTripleLine = int(strTripleLineID[3:])
        elif strTripleLineID[:1] == 'T':
            intTripleLine = int(strTripleLineID[2:])
        fltDistance = np.sort(self.__PeriodicTripleLineDistanceMatrix[intTripleLine])[1]
        return fltDistance
    def FindStrip(self, arrStart: np.array, arrVector: np.array,fltWidth: float, fltIncrement: float, fltLength: float):
        lstLength = []
        lstIndices  = []
        lstI = []
        lstValues = []
        intMax = np.floor(fltLength/(fltIncrement)).astype('int')
        intMin = 1
        for j in range(intMin,intMax+1):
            l = fltIncrement*j
            lstLength.append(l)
            v = gf.NormaliseVector(arrVector)
            arrWidth  = fltWidth*np.cross(v,np.array([0,0,1]))
            lstI = self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrStart,l*v, 
                                                           arrWidth,np.array([0,0,self.CellHeight]))
            if len(lstI) >0:
                lstIndices.extend(lstI)
                lstIndices = list(np.unique(lstIndices))
                lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
            else:
                lstValues.append(0)
        return lstLength, lstValues, lstIndices   
    def FindThreeGrainStrips(self, strTripleLineID: int,fltWidth: float, fltIncrement: float, strValue = 'mean',fltLength = None, blnReturnNumbers = False):
        lstOfVectors = [] #unit vectors that bisect the grain boundary directions
        lstValues = []
        lstRadii = []
        lstIndices  = []
        lstI = []
        lstVolume = []
        lstN = []
        if fltLength is None:
           # fltClosest = np.sort(self.__TripleLineDistanceMatrix[intTripleLine,list(setTripleLines)])[1]/2
           fltClosest = self.FindClosestTripleLine(strTripleLineID)/2
        else:
            fltClosest = fltLength
        intMax = np.floor(fltClosest/(fltIncrement)).astype('int')
        #intMin = np.ceil((self.__LatticeParameter/np.sqrt(2))/fltIncrement).astype('int')
        intMin = 1
        for strGB in self.__UniqueTripleLines[strTripleLineID].GetUniqueAdjacentGrainBoundaries():
            arrVector = self.__UniqueGrainBoundaries[strGB].GetVectorDirection(strTripleLineID, self.__LatticeParameter, bln3D=True) 
            lstOfVectors.append(gf.NormaliseVector(arrVector))
        for j in range(intMin,intMax+1):
            lstPolygons = []
            r = fltIncrement*j
            lstRadii.append(r)
            n = len(lstOfVectors)
            for kVector in range(n):
                v = gf.NormaliseVector(lstOfVectors[np.mod(kVector,n)] + lstOfVectors[np.mod(kVector+1,n)])
                arrCentre  = self.GetUniqueTripleLines(strTripleLineID).GetCentre()
                arrWidth  = fltWidth*np.cross(v,np.array([0,0,1]))
                lstI = self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrCentre,r*v, 
                                                           arrWidth,np.array([0,0,self.CellHeight]))
                lstIndices.extend(lstI)
                objPlg = sp.geometry.Polygon([arrWidth[0:2]/2, arrWidth[0:2]/2 +r*v[0:2],-arrWidth[0:2]/2 + r*v[0:2],-arrWidth[0:2]/2,arrWidth[0:2]/2])
                lstPolygons.append(objPlg)
                lstIndices = list(np.unique(lstIndices))
            u =  sp.ops.unary_union(lstPolygons)
            if u.is_valid:
                fltVolume = u.area*self.CellHeight
            else:
                raise('Error invalid polygon base')   
            if strValue == 'mean':
                if len(lstIndices) >0:
                    lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
                   # lstVR.append(u.area/(3*fltWidth)) #not used anymore
                    lstN.append(len(lstIndices))
                else:
                    lstValues.append(0)
            elif strValue =='volume':
                if len(lstIndices) >0:
                    lstValues.append(np.sum(self.GetAtomsByID(lstIndices)[:,self._intPE]/fltVolume))
                    lstVolume.append(r)
                else:
                    lstValues.append(0)
        if strValue == 'mean':
            if blnReturnNumbers:
                return lstRadii, lstValues, lstIndices, lstN
            else:        
                return lstRadii, lstValues,lstIndices
        elif strValue =='volume':
            return lstRadii, lstValues, lstIndices, lstN  
 

class QuantisedRectangularPoints(object): #linear transform parallelograms into a rectangular parameter space
    def __init__(self, in2DPoints: np.array, inUnitBasisVectors: np.array, n: int, fltGridSize: float, intMinCount: int, intDilation = 2, blnDebug = False):
        self.__WrapperWidth = n #mininum count specifies how many nonlattice atoms occur in the float grid size before it is counted as a grain boundary grid or triple line
        self.__BasisVectors = inUnitBasisVectors
        self.__InverseMatrix =  np.linalg.inv(inUnitBasisVectors)
        self.__GridSize = fltGridSize
        self.__WrapperVector = np.array([n,n])
        arrPoints =  np.matmul(in2DPoints, self.__BasisVectors)*(1/fltGridSize)
        intMaxHeight = np.round(np.max(arrPoints[:,0])).astype('int')
        intMaxWidth = np.round(np.max(arrPoints[:,1])).astype('int')
        self.__ArrayGrid =  np.zeros([(intMaxHeight+1),intMaxWidth+1])
        arrPoints = np.round(arrPoints).astype('int')
        for j in arrPoints:
            self.__ArrayGrid[j[0],j[1]] += 1 #this array represents the simultion cell
        self.__ArrayGrid = (self.__ArrayGrid >= intMinCount).astype('int')
        self.__ArrayGrid = ndimage.binary_dilation(self.__ArrayGrid, np.ones([intDilation,intDilation]))
        self.__ArrayGrid = remove_small_holes(self.__ArrayGrid, 4)
        self.__ExtendedArrayGrid = np.zeros([np.shape(self.__ArrayGrid)[0]+2*n,np.shape(self.__ArrayGrid)[1]+2*n])
        self.__ExtendedArrayGrid[n:-n, n:-n] = self.__ArrayGrid
        self.__ExtendedArrayGrid[0:n, n:-n] = self.__ArrayGrid[-n:,:]
        self.__ExtendedArrayGrid[-n:, n:-n] = self.__ArrayGrid[:n,:]
        self.__ExtendedArrayGrid[:,0:n] = self.__ExtendedArrayGrid[:,-2*n:-n]
        self.__ExtendedArrayGrid[:,-n:] = self.__ExtendedArrayGrid[:,n:2*n]
        self.__ExtendedArrayGrid = gaussian(self.__ExtendedArrayGrid, sigma=intDilation/2)
        self.__ExtendedArrayGrid = np.round(self.__ExtendedArrayGrid,0).astype('int')
        self.__ExtendedArrayGrid = (self.__ExtendedArrayGrid.astype('bool')).astype('int')
        self.__ExtendedSkeletonGrid = skeletonize(self.__ExtendedArrayGrid).astype('int')
        self.__GrainValue = 0
        self.__GBValue = 1 #just fixed constants used in the array 
        self.__DislocationValue = 2
        self.__TripleLineValue = 3
        self.__TriplePoints = []
        self.__Dislocations = []
        self.__GrainBoundaryLabels = []
        self.__GrainBoundaryIDs = []
        self.__blnGrainBoundaries = False #this flag is set once FindGrainBoundaries() is called 
        if not(blnDebug):
            self.FindTriplePoints() #when debugging comment these out and call the methods one at a time in this order
            self.FindGrainBoundaries()
            self.MakeAdjacencyMatrix()
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
                    # if not (arrCurrent[0].all() == self.__GBValue or arrCurrent[-1].all() == self.__GBValue or arrCurrent[:,0].all() == self.__GBValue or  arrCurrent[:,-1].all() ==self.__GBValue):
                        if self.CheckEachSide(arrCurrent):
                            self.SetSkeletonValue(x,self.__TripleLineValue)
                elif intSwaps ==6:
                    self.SetSkeletonValue(x,self.__TripleLineValue)
                elif intSwaps < 4 and blnFlagEndPoints and m==3: #only flag end points for a search with 3x3
                   # if not self.OnEdge(x,3): #only set dislocation end points inside the ArrayGrid
                    self.SetSkeletonValue(x,self.__DislocationValue)
        self.__Dislocations = np.argwhere(self.__ExtendedSkeletonGrid == self.__DislocationValue)
        return np.argwhere(self.__ExtendedSkeletonGrid == self.__TripleLineValue)
    def CheckEachSide(self, inArray: np.array)->bool: #checks to make sure there isn't a line of points along one side of the grid
        blnReturn = True
        if inArray[0].all() != self.__GrainValue or inArray[-1].all() != self.__GrainValue or inArray[:,0].all() != self.__GrainValue or inArray[:,-1].all() != self.__GrainValue:
            blnReturn = False
        return blnReturn
    def SetSkeletonValue(self,inArray:np.array, intValue: int):
        self.__ExtendedSkeletonGrid[inArray[0], inArray[1]] = intValue
    def GetSkeletonValue(self, inArray:np.array)->int:
        return self.__ExtendedSkeletonGrid[inArray[0], inArray[1]]
    def __ConvertToCoordinates(self, inArrayPosition: np.array): #takes array position and return real 2D coordinates
        arrPoints = (inArrayPosition - np.ones([2])*self.__WrapperWidth)*self.__GridSize
        arrPoints = np.matmul(arrPoints, self.__InverseMatrix)
        rtnArray = np.zeros([len(arrPoints),3])
        rtnArray[:,0:2] = arrPoints
        return rtnArray
    def __ResetSkeletonGrid(self):
        self.__ExtendedSkeletonGrid[self.__ExtendedSkeletonGrid != self.__GrainValue] = self.__GBValue
    def OnEdge(self, inPoint: np.array, intTolerance: int)->bool:
        blnReturn = False
        if inPoint[0] <= intTolerance or inPoint[0] >= np.shape(self.__ExtendedArrayGrid)[0] -intTolerance or inPoint[1] <= intTolerance or inPoint[1] >= np.shape(self.__ExtendedArrayGrid)[1] -intTolerance:
            blnReturn = True
        return blnReturn
    def FindDislocations(self):
        return self.__Dislocations
    def ClearWrapper(self, blnUsingTripleLines = True): 
        k = self.__WrapperWidth
        self.__ExtendedSkeletonGrid[:k, :] = self.__GrainValue
        self.__ExtendedSkeletonGrid[k:, :] = self.__GrainValue
        self.__ExtendedSkeletonGrid[:, :k] = self.__GrainValue
        self.__ExtendedSkeletonGrid[:, k:] = self.__GrainValue
    def FindTriplePoints(self):
        self.__TriplePoints = self.ClassifyGBPoints(3, True)
        k = np.ceil(self.__WrapperWidth/2).astype('int')
        lstDeleteTJs = []
        for j, arrTriplePoint in enumerate(self.__TriplePoints): #errors in skeletonize sometimes produces fake TJS at the edge of the extended skeleton grid. 
            if self.OnEdge(arrTriplePoint, k):
                self.SetSkeletonValue(arrTriplePoint, self.__GBValue)
                lstDeleteTJs.append(j)
        self.__TriplePoints = np.delete(self.__TriplePoints, lstDeleteTJs, axis= 0)               
    def GetTriplePoints(self)->np.array:
        return self.__ConvertToCoordinates(self.__TriplePoints)
    def FindGrainBoundaries(self):
        intStart = np.max(self.__ExtendedSkeletonGrid)+1
        #self.ClearWrapper(True)
        for k in range(len(self.__TriplePoints)):
            self.MakeGrainBoundaries(k, intStart)
            intStart =  np.max(self.__ExtendedSkeletonGrid)+1
        arrPoints = np.argwhere(self.__ExtendedSkeletonGrid == self.__TripleLineValue)
        for j in arrPoints: #this removes any erroneous triple lines which can now be detected as they aren't adjacent to a grain boundary
            arrBox = self.__ExtendedSkeletonGrid[j[0]-1:j[0]+2,j[1]-1:j[1]+2]
            if np.max(np.unique(arrBox)) <= self.__TripleLineValue:
                self.SetSkeletonValue(j, self.__GrainValue)
                arrPoints = np.delete(arrPoints, j, axis=0)
        self.__TriplePoints = arrPoints
        #self.MergeGrainBoundaries() #removed as this caused problems with finding adjacent triple lines
        #self.ClearWrapper()
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
        arrMod = np.array([np.shape(self.__ArrayGrid)])
        if not self.__blnGrainBoundaries:
            self.FindGrainBoundaries()
        for j in self.__GrainBoundaryIDs:
            arrPoints = np.argwhere(self.__ExtendedSkeletonGrid == j)
            if len(arrPoints) > 3:
                arrPoints = np.unique(np.mod(arrPoints,arrMod), axis=0)
                lstGrainBoundaries.append(self.__ConvertToCoordinates(arrPoints))
        return lstGrainBoundaries
    def MergeGrainBoundaries(self):
        arrMod = np.array([np.shape(self.__ArrayGrid)])
        lstCurrentIDs = list(np.copy(self.__GrainBoundaryIDs))
        counterj = 0
        while (counterj < len(self.__GrainBoundaryIDs)):
            j = self.__GrainBoundaryIDs[counterj]
            lstCurrentIDs.remove(j)
            arrPointsj = np.argwhere(self.__ExtendedSkeletonGrid == j)
            arrDistanceMatrixJ = spatial.distance_matrix(self.__TriplePoints, arrPointsj)
            arrPositionsJ = np.argwhere(arrDistanceMatrixJ < 2)[:,1]
            arrPointsj =np.delete(arrPointsj, arrPositionsJ, axis = 0)
            arrPointsjMod = np.mod(arrPointsj, arrMod)
            counterk = 0
            while (counterk < len(lstCurrentIDs) and len(arrPointsjMod) > 0):
                blnEmptyArray = False
                k = lstCurrentIDs[counterk]
                arrPointsk = np.argwhere(self.__ExtendedSkeletonGrid == k)
                arrDistanceMatrixK = spatial.distance_matrix(self.__TriplePoints, arrPointsk)
                arrPositionsK = np.argwhere(arrDistanceMatrixK < 2)[:,1]
                arrPointsk =np.delete(arrPointsk, arrPositionsK, axis = 0)
                arrPointskMod = np.mod(arrPointsk, arrMod)
                arrDistanceMatrix = spatial.distance_matrix(arrPointsjMod, arrPointskMod)
                if arrDistanceMatrix.size > 0:
                    fltMin = np.min(arrDistanceMatrix)
                    #arrEquivalent = np.argwhere(arrDistanceMatrix == 0) #as points adjacent to triple points have been
                else:#removed then diagonal connectivity is permissible
                    blnEmptyArray = True                      
                if fltMin < 2 and not(blnEmptyArray): #the two grain boundaries periodically link
                    #if len(arrEquivalent) > 0:
                       # for arrPositions in arrEquivalent:
                            #if np.linalg.norm(arrPointsj[arrPositions[0]]- arrPointsk[arrPositions[1]]) > 0:
                            #    if arrPointsj[arrPositions[0]]-self.__WrapperVector 
                       #     if not(self.InGrid(arrPointsk[arrPositions[1]])):
                       #         self.SetSkeletonValue(arrPointsk[arrPositions[1]],self.__GrainValue)
                    self.__ExtendedSkeletonGrid[self.__ExtendedSkeletonGrid == k] = j
                    self.__GrainBoundaryIDs.remove(k)
                    lstCurrentIDs.remove(k)
                else:
                    counterk +=1
            counterj +=1
    def InGrid(self, inPoint: np.array)->bool:
        blnReturn = True
        arrPoint = inPoint - self.__WrapperVector
        ZeroMax, OneMax = np.shape(self.__ArrayGrid)
        if arrPoint[0] < 0 or arrPoint[1] < 0 or arrPoint[1] > ZeroMax or arrPoint[1] > OneMax:
            blnReturn = False
        return blnReturn
    def MakeGrainBoundaries(self, intTriplePoint: int,intValue: int):
        x = self.__TriplePoints[intTriplePoint]
        arrCurrent = np.copy(self.__ExtendedSkeletonGrid[x[0]-1:x[0]+2,x[1]-1:x[1]+2])
        arrPoints =np.argwhere(arrCurrent == self.__GBValue) + x - np.array([1,1])
        for index1 in range(len(arrPoints)): #checks whether there is more than one point corresponding to the same GB
            if self.GetSkeletonValue(arrPoints[index1]) == self.__GBValue: 
                self.SetSkeletonValue(arrPoints[index1], intValue+index1)
                for index2 in range(index1,len(arrPoints)):
                    if np.linalg.norm(arrPoints[index1]-arrPoints[index2],axis=0) ==1:
                        if self.GetSkeletonValue(arrPoints[index2]) == self.__GBValue:
                            self.SetSkeletonValue(arrPoints[index2], intValue+index1)
        arrCurrent = np.copy(self.__ExtendedSkeletonGrid[x[0]-1:x[0]+2,x[1]-1:x[1]+2])
        lstStartPoints = []
        for k in range(intValue, np.max(arrCurrent)+1):
            arrPositions = np.argwhere(arrCurrent == k)
            if len(arrPositions) == 1:
                lstStartPoints.append(arrPositions[0]+x-np.array([1,1]))
            elif len(arrPositions) > 1:
                for k in arrPositions:
                    if np.linalg.norm(k-np.array([1,1])) > 1:
                        lstStartPoints.append(k+x-np.array([1,1]))
        for j in lstStartPoints: #normally 3 points buy maybe one or two
            self.SetSkeletonValue(j, intValue)
            blnEnd = False
            arrNewPoint = j
            counter = 0
            blnFirstTime = True
            while not(blnEnd) and counter < 10000: #counter is just here incase there is a failure to converge
                arrCurrent = np.copy(self.__ExtendedSkeletonGrid[arrNewPoint[0]-1:arrNewPoint[0]+2,arrNewPoint[1]-1:arrNewPoint[1]+2])
                arrBoxPoint = np.argwhere(arrCurrent == self.__GBValue)
                arrTriplePoint = np.argwhere(arrCurrent == self.__TripleLineValue)
                arrDislocationPoint = np.argwhere(arrCurrent == self.__DislocationValue)
                if len(arrBoxPoint) ==1 and blnFirstTime:
                    arrNewPoint = arrNewPoint+arrBoxPoint[0] - np.array([1,1])
                    self.SetSkeletonValue(arrNewPoint, intValue)
                    blnFirstTime = False
                elif len(arrBoxPoint) ==1 and len(arrTriplePoint) == 0 and len(arrDislocationPoint)  == 0:
                    arrNewPoint = arrNewPoint+arrBoxPoint[0] - np.array([1,1])
                    self.SetSkeletonValue(arrNewPoint, intValue) 
                elif len(arrBoxPoint) ==1 and (len(arrTriplePoint) > 0 or len(arrDislocationPoint) > 0):
                    if np.linalg.norm(arrBoxPoint[0]-arrNewPoint) ==1:
                        arrNewPoint = arrNewPoint+arrBoxPoint[0] - np.array([1,1])
                        self.SetSkeletonValue(arrNewPoint, intValue)
                        if len(arrDislocationPoint) > 0:
                            self.SetSkeletonValue(arrDislocationPoint[0]+arrNewPoint -np.array([1,1]), intValue)
                    else:     
                        blnEnd = True
                elif len(arrDislocationPoint) > 0:
                    self.SetSkeletonValue(arrDislocationPoint[0]+arrNewPoint -np.array([1,1]), intValue)
                    blnEnd = True
                else:     
                        blnEnd = True
                counter +=1
            self.__GrainBoundaryIDs.append(intValue)
            intValue +=1 
    def MakeAdjacencyMatrix(self):
        self.__AdjacencyMatrix = np.zeros([len(self.__TriplePoints),len(self.__TriplePoints)])        
        for j in self.__GrainBoundaryIDs:
            lstAdjacentTJs = []
            lstDeleteTJs = []
            arrPoints = np.argwhere(self.__ExtendedSkeletonGrid == j)
            for intTJ in range(len(self.__TriplePoints)):
                if np.min(np.linalg.norm(arrPoints - self.__TriplePoints[intTJ], axis= 1)) < 2:
                    lstAdjacentTJs.append(intTJ)
            if len(lstAdjacentTJs) == 2:
                self.__AdjacencyMatrix[lstAdjacentTJs[0],lstAdjacentTJs[1]] =1
                self.__AdjacencyMatrix[lstAdjacentTJs[1],lstAdjacentTJs[0]] =1
        lstDeleteTJs =    np.where(self.__AdjacencyMatrix.any(axis=1) == 0)
        if len(lstDeleteTJs) > 0 :
            lstDeleteTJs = list(lstDeleteTJs[0])
            self.__TriplePoints = np.delete(self.__TriplePoints, lstDeleteTJs, axis = 0)
            self.__AdjacencyMatrix = np.delete(self.__AdjacencyMatrix, lstDeleteTJs, axis = 0)
            self.__AdjacencyMatrix = np.delete(self.__AdjacencyMatrix, lstDeleteTJs, axis = 1)
        return self.__AdjacencyMatrix
    def GetAdjacentTriplePoints(self, intTriplePoint: int)->np.array:
        lstAdjacentTriplePoints = list(np.where(self.__AdjacencyMatrix[intTriplePoint] == 1)[0])
        return lstAdjacentTriplePoints
    def GetEquivalentTriplePoints(self, intTriplePoint: int, intTolerance =2)->np.array:
        lstEquivalentTripleLines = []
        arrMod = np.array(np.shape(self.__ArrayGrid))
        arrCurrentTriplePoint  = self.__TriplePoints[intTriplePoint]
        for intPosition,j in enumerate(self.__TriplePoints):
            arrDisplacement = np.mod(arrCurrentTriplePoint - j, arrMod)
            for j in range(len(arrDisplacement)):
                if arrDisplacement[j] > arrMod[j]/2:
                    arrDisplacement[j] = arrDisplacement[j] - arrMod[j]
            if np.linalg.norm(arrDisplacement,axis=0) < intTolerance:
                lstEquivalentTripleLines.append(int(intPosition))
        lstEquivalentTripleLines.sort()
        return lstEquivalentTripleLines