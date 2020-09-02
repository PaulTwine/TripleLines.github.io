import re
import numpy as np
import GeometryFunctions as gf
import GeneralLattice as gl
from scipy import spatial, optimize, ndimage,stats
from skimage.morphology import skeletonize, thin, medial_axis, remove_small_holes, remove_small_objects
from scipy.cluster.vq import kmeans,vq
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from skimage.filters import gaussian
from skimage import measure
from sklearn.cluster import DBSCAN
import shapely as sp
import copy
import warnings
from functools import reduce
from sklearn.metrics import mean_squared_error

class LAMMPSData(object):
    def __init__(self,strFilename: str, intLatticeType: int, fltLatticeParameter: float, objAnalysis: object):
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
                objTimeStep = objAnalysis(timestep, N,intNumberOfColumns,lstColumnNames, lstBoundaryType, lstBounds,intLatticeType, fltLatticeParameter)
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
    def GetNumberOfColumns(self):
        return len(self.__ColumnNames)
    def SetColumnByIndex(self, arrColumn:np.array, intColumnIndex: int):
        self.__AtomData[:, intColumnIndex] = arrColumn
    def GetColumnByIDs(self,lstOfAtomIDs: list, intColumn: int):
        return self.__AtomData[np.isin(self.__AtomData[:,0], lstOfAtomIDs), intColumn]     
    def SetColumnByIDs(self,lstOfAtomIDs: list, intColumn: int, arrValues: np.array):
        self.__AtomData[np.isin(self.__AtomData[:,0], lstOfAtomIDs), intColumn] = arrValues
    def SetRow(self, intRowNumber: int, lstRow: list):
        self.__AtomData[intRowNumber] = lstRow
    def AddColumn(self, arrColumn: np.array, strColumnName):
        self.__AtomData = np.append(self.__AtomData, arrColumn, axis=1)
        self.__ColumnNames.append(strColumnName)
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
        if 'StructureType' in self.GetColumnNames():
            self._intStructureType = int(self.GetColumnNames().index('StructureType'))
        else:
            warnings.warn('Error missing atom structure types in dump file.')
        if 'c_v1' in self.GetColumnNames():
            self._intVolume = int(self.GetColumnNames().index('c_v1'))
        else:
            warnings.warn('Per atom volume data is missing.')
        if 'c_pe1' in self.GetColumnNames():
            self._intPE = int(self.GetColumnNames().index('c_pe1'))
        else:
            warnings.warn('Per atom potential energy is missing.')
        self._intPositionX = int(self.GetColumnNames().index('x'))
        self._intPositionY = int(self.GetColumnNames().index('y'))
        self._intPositionZ = int(self.GetColumnNames().index('z'))
        self.CellHeight = np.linalg.norm(self.GetCellVectors()[2])
        self.__fltGrainTolerance = 1.96
        self.__DefectiveAtoms = []
        self.__NonDefectiveAtoms = []
        self.FindPlaneNormalVectors()
    def FindPlaneNormalVectors(self):
        n= self.__Dimensions
        arrVectors = np.zeros([n,n])
        arrLimits = np.zeros([n])
        for j in range(n):
            arrVectors[j] = gf.NormaliseVector(np.cross(self.GetUnitCellBasis()[j],self.GetUnitCellBasis()[np.mod(j+1,n)]))
            arrLimits[j] = np.linalg.norm(self.GetCellVectors()[:,np.mod(j+2,n)]) 
        self.__PlaneNormalVectors = arrVectors
        self.__PlaneNormalLimits = arrLimits
    def GetPlaneNormalVectors(self):
        return self.__PlaneNormalVectors
    def CategoriseAtoms(self, fltTolerance = None):    
        lstOtherAtoms = list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == 0)[0])
        lstPTMAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == self._LatticeStructure)[0])
        lstUnknownAtoms = list(np.where(np.isin(self.GetColumnByIndex(self._intStructureType).astype('int') ,[0,self._LatticeStructure],invert=True))[0])
        self.__PTMAtoms = lstPTMAtoms
        self.__NonPTMAtoms = lstOtherAtoms + lstUnknownAtoms
        self.__OtherAtoms = lstOtherAtoms
        self.__UnknownAtoms = lstUnknownAtoms
        self.FindDefectiveAtoms(fltTolerance)
        self.FindNonDefectiveAtoms(fltTolerance)
        self.__LatticeAtoms = list(set(self.__NonDefectiveAtoms) & set(self.__PTMAtoms))
        self.__NonLatticeAtoms = list(np.where(np.isin(self.GetColumnByIndex(self._intStructureType).astype('int'), self.__LatticeAtoms, invert=True))[0])
    def GetLatticeAtoms(self):
        return self.__LatticeAtoms
    def GetNonLatticeAtoms(self):
        return self.__NonLatticeAtoms  
    def FindDefectiveAtoms(self, fltTolerance = None):
        if fltTolerance is None:
            fltStdLatticeValue = np.std(self.GetPTMAtoms()[:,self._intPE])
            fltTolerance = self.__fltGrainTolerance*fltStdLatticeValue #95% limit assuming Normal distribution
        fltMeanLatticeValue = np.mean(self.GetPTMAtoms()[:,self._intPE])
        lstDefectiveAtoms = np.where((self.GetColumnByIndex(self._intPE) > fltMeanLatticeValue +fltTolerance) | (self.GetColumnByIndex(self._intPE) < fltMeanLatticeValue - fltTolerance))[0]
        self.__DefectiveAtoms = list(self.GetAtomData()[lstDefectiveAtoms,0].astype('int'))
        return self.GetRows(lstDefectiveAtoms)
    def FindNonDefectiveAtoms(self,fltTolerance = None):
        if fltTolerance is None:
            fltStdLatticeValue = np.std(self.GetPTMAtoms()[:,self._intPE])
            fltTolerance = self.__fltGrainTolerance*fltStdLatticeValue #95% limit assuming Normal distribution
        fltMeanLatticeValue = np.mean(self.GetPTMAtoms()[:,self._intPE])
        lstNonDefectiveAtoms = np.where((self.GetColumnByIndex(self._intPE) <= fltMeanLatticeValue +fltTolerance) & (self.GetColumnByIndex(self._intPE) >= fltMeanLatticeValue  - fltTolerance))[0]
        self.__NonDefectiveAtoms = list(self.GetAtomData()[lstNonDefectiveAtoms,0].astype('int'))
        return self.GetRows(lstNonDefectiveAtoms)
    def GetOtherAtomIDs(self):
        return list(self.GetAtomData()[self.__OtherAtoms,0].astype('int'))
    def GetPTMAtomIDs(self):
        return list(self.GetAtomData()[self.__PTMAtoms,0].astype('int'))
    def GetNonPTMAtomIDs(self):
        return list(self.GetAtomData()[self.__PTMAtoms,0].astype('int'))
    def GetDefectiveAtomIDs(self):
        return self.__DefectiveAtoms
    def GetNonDefectiveAtomIDs(self):
        return self.__NonDefectiveAtoms
    def GetNonDefectiveAtoms(self):
        if len(self.__NonDefectiveAtoms) ==0:
            self.FindNonDefectiveAtoms()
        return self.GetAtomsByID(self.__NonDefectiveAtoms)
    def GetDefectiveAtoms(self):
        if len(self.__DefectiveAtoms) ==0:
            self.FindDefectiveAtoms()
        return self.GetAtomsByID(self.__DefectiveAtoms)
    def GetNonPTMAtoms(self):
        return self.GetRows(self.__NonPTMAtoms)
    def GetUnknownAtoms(self):
        return self.GetRows(self.__UnknownAtoms) 
    def GetPTMAtoms(self):
        return self.GetRows(self.__PTMAtoms)  
    def GetOtherAtoms(self):
        return self.GetRows(self.__OtherAtoms)
    def GetNumberOfNonPTMAtoms(self):
        return len(self.__NonPTMAtoms)
    def GetNumberOfOtherAtoms(self)->int:
        return len(self.GetRows(self.__OtherAtoms))
    def GetNumberOfPTMAtoms(self)->int:
        return len(self.__PTMAtoms)
    def PlotGrainAtoms(self, strGrainNumber: str):
        return self.__PlotList(self.__PTMAtoms)
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
            for k in range(len(inVector2)):
                arrPeriodicDistance[j,k] = self.PeriodicMinimumDistance(inVector1[j],inVector2[k])
        return arrPeriodicDistance
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        inVector2 = self.PeriodicShiftCloser(inVector1, inVector2)
        return np.linalg.norm(inVector2-inVector1, axis=0)
    def FindNonGrainMediod(self, inPoint: np.array, fltRadius: float, bln2D= True, region = 'cylinder'):
        arrReturn = np.ones(3)*self.CellHeight/2
        lstPointsIndices = []
        if region == 'cylinder':
            lstPointsIndices = self.FindCylindricalAtoms(self.GetDefectiveAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        elif region == 'sphere':
            lstPointsIndices = self.FindSphericalAtoms(self.GetDefectiveAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            arrPoint = gf.FindGeometricMediod(arrPoints, bln2D)
            if bln2D:
                return arrPoint[0:2]
            else: 
                return arrPoint  
        else:
            return None
    def FindNonGrainMean(self, inPoint: np.array, fltRadius: float): 
        lstPointsIndices = []
        lstPointsIndices = self.FindCylindricalAtoms(self.GetDefectiveAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            return np.mean(arrPoints, axis=0)  
        else:
            return None
    def FindGrainMean(self, inPoint: np.array, fltRadius: float): 
        lstPointsIndices = []
        lstPointsIndices = self.FindCylindricalAtoms(self.GetPTMAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            return np.mean(arrPoints, axis=0)  
        else:
            return inPoint  
    def SphereLiesInCell(self, arrCentre: np.array, fltRadius: float)->bool:
        arrProjections = np.matmul(arrCentre, np.transpose(self.__PlaneNormalVectors))
        blnInside = False
        if np.all(arrProjections > fltRadius) and np.all(self.__PlaneNormalLimits -arrProjections > fltRadius):
            blnInside = True
        return blnInside        
    def FindSphericalAtoms(self,arrPoints, arrCentre: np.array, fltRadius: float, blnPeriodic =True)->list:
        lstIndices = []
        if blnPeriodic:
            blnPeriodic = not(self.SphereLiesInCell(arrCentre, fltRadius))
        if blnPeriodic:
            arrCentres = self.PeriodicEquivalents(arrCentre)
            for j in arrCentres:
                lstIndices.extend(gf.SphericalVolume(arrPoints[:,1:4],j,fltRadius))
        else:
            lstIndices.extend(gf.SphericalVolume(arrPoints[:,1:4],arrCentre,fltRadius))
        lstIndices = list(np.unique(lstIndices))
        return list(arrPoints[lstIndices,0])
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
               
class LAMMPSAnalysis3D(LAMMPSPostProcess):
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int, fltLatticeParameter: float):
        LAMMPSPostProcess.__init__(self, fltTimeStep,intNumberOfAtoms, intNumberOfColumns, lstColumnNames, lstBoundaryType, lstBounds,intLatticeType)
        self.__GrainBoundaries = dict()
        self.__JunctionLines = dict()
        self.__Grains = dict()
        self.__LatticeParameter = fltLatticeParameter
        self.__GrainLabels = []
        self.__JunctionLineIDs = []
    def SetLatticeParameter(self, fltParameter: float):
        self.__LatticeParameter = fltParameter
    def LabelAtomsByGrain(self):
        self.__QuantisedCuboidPoints = QuantisedCuboidPoints(self.GetDefectiveAtoms()[:,1:4],self.GetUnitBasisConversions(),self.GetCellVectors(),self.__LatticeParameter*np.ones(3),10)
        lstGrainAtoms = self.GetLatticeAtoms()
        lstNonGrainAtoms = self.GetNonLatticeAtoms()
        lstGrainNumbers = self.__QuantisedCuboidPoints.ReturnGrains(self.GetAtomsByID(lstGrainAtoms)[:,1:4])
        self.AppendGrainNumbers(lstGrainNumbers, lstGrainAtoms)
        self.__QuantisedCuboidPoints.FindJunctionLines()
        self.__JunctionLineIDs = self.__QuantisedCuboidPoints.GetJunctionLineIDs()
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i] = gl.GeneralJunctionLine(self.__QuantisedCuboidPoints.GetJunctionLinePoints(i),i)
            self.__JunctionLines[i].SetAdjacentGrains(self.__QuantisedCuboidPoints.GetAdjacentGrains(i, 'JunctionLine'))
            self.__JunctionLines[i].SetAdjacentGrainBoundaries(self.__QuantisedCuboidPoints.GetAdjacentGrainBoundaries(i))
            self.__JunctionLines[i].SetPeriodicDirections(self.__QuantisedCuboidPoints.GetPeriodicExtensions(i,'JunctionLine'))
        self.__GrainBoundaryIDs = self.__QuantisedCuboidPoints.GetGrainBoundaryIDs()
        for k in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[k] = gl.GeneralGrainBoundary(self.__QuantisedCuboidPoints.GetGrainBoundaryPoints(k),k)
            self.__GrainBoundaries[k].SetAdjacentGrains(self.__QuantisedCuboidPoints.GetAdjacentGrains(k, 'GrainBoundary'))
            self.__GrainBoundaries[k].SetAdjacentJunctionLines(self.__QuantisedCuboidPoints.GetAdjacentJunctionLines(k))
            self.__GrainBoundaries[k].SetPeriodicDirections(self.__QuantisedCuboidPoints.GetPeriodicExtensions(k,'GrainBoundary'))
            for l in self.__GrainBoundaries[k].GetMeshPoints():
                lstSurroundingAtoms = list(self.FindSphericalAtoms(self.GetAtomsByID(lstNonGrainAtoms)[:,0:4],l, 3*self.__LatticeParameter))
                self.__GrainBoundaries[k].AddAtomIDs(lstSurroundingAtoms)
        self.CheckBoundaries()   
    def AppendGrainNumbers(self, lstGrainNumbers: list, lstGrainAtoms = None):
        if 'GrainNumber' not in self.GetColumnNames():
            self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]), 'GrainNumber')
        arrGrainNumbers = np.array([lstGrainNumbers])
        self.__GrainLabels = list(np.unique(lstGrainNumbers))
        np.reshape(arrGrainNumbers, (len(lstGrainNumbers),1))
        self.__intGrainNumber = self.GetNumberOfColumns()-1 #column number for the grains
        if lstGrainAtoms is None:
            self.SetColumnByIndex(arrGrainNumbers, self.__intGrainNumber)
        else:
            self.SetColumnByIDs(lstGrainAtoms, self.__intGrainNumber, arrGrainNumbers)
    def AssignPE(self):
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i].SetPE(np.sum(self.GetColumnByIDs(self.__JunctionLines[i].GetAtomIDs(),self._intPE)))
        for j in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[j].SetPE(np.sum(self.GetColumnByIDs(self.__GrainBoundaries[j].GetAtomIDs(),self._intPE)))
    def AssignVolumes(self):
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i].SetVolume(np.sum(self.GetColumnByIDs(self.__JunctionLines[i].GetAtomIDs(),self._intVolume)))
        for j in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[j].SetVolume(np.sum(self.GetColumnByIDs(self.__GrainBoundaries[j].GetAtomIDs(),self._intVolume)))
    def MakeGrainTrees(self):
        lstGrainLabels = self.__GrainLabels
        if 0 in lstGrainLabels:
            lstGrainLabels.remove(0)
        for k in self.__GrainLabels:
            lstIDs = self.GetGrainAtomIDs(k)
            self.__Grains[k] = spatial.KDTree(self.GetAtomsByID(lstIDs)[:,1:4])
    def CheckBoundaries(self):
        lstGrains = self.__GrainLabels
        if 0 in lstGrains:
            lstGrains.remove(0)
        lstNextAtoms = list(set(self.GetLatticeAtoms()).intersection(set(self.GetGrainAtomIDs(0))))
        lstAtoms = []
        self.MakeGrainTrees()
        while len(set(lstNextAtoms).difference(lstAtoms)) > 0:
            lstAtoms = lstNextAtoms
            arrAtoms = np.array(lstAtoms)
            arrPoints = self.GetAtomsByID(lstAtoms)[:,1:4]
            clustering = DBSCAN(1.01*self.__LatticeParameter/np.sqrt(2), 1).fit(arrPoints)
            arrValues = clustering.labels_
            lstUniqueValues = np.unique(arrValues)
            for i in lstUniqueValues:
                intMax = 0
                intGrain = 0
                arrCluster = arrPoints[arrValues == i]
                arrClusterIDs = arrAtoms[arrValues == i]
                for k in lstGrains:
                    arrDistances = self.__Grains[k].query(arrCluster, k=12, distance_upper_bound= 1.01*self.__LatticeParameter/np.sqrt(2))[0]
                    intCurrent = len(arrDistances[arrDistances < 1.01*self.__LatticeParameter/np.sqrt(2)])
                    if intCurrent > intMax:
                        intMax = intCurrent
                        intGrain = k
                    elif intCurrent == intMax and intMax > 0:
                        intGrain = 0
                if intGrain > 0:
                    self.SetColumnByIDs(list(arrClusterIDs), self.__intGrainNumber, intGrain*np.ones(len(arrClusterIDs)))
            lstNextAtoms = list(set(self.GetLatticeAtoms()).intersection(set(self.GetGrainAtomIDs(0))))
            self.MakeGrainTrees()
        if len(lstNextAtoms) > 0:
            warnings.warn(str(len(lstNextAtoms)) + ' grain atom(s) have been assigned a grain number of -1 \n' + str(lstNextAtoms))
            self.SetColumnByIDs(lstNextAtoms, self.__intGrainNumber, -1**np.ones(len(lstNextAtoms)))
    def FinaliseGrainBoundaries(self):
        lstGBIDs =  list(np.copy(self.__GrainBoundaryIDs))
        while len(lstGBIDs) > 0:
            lstOverlapAtoms = []
            intGBID = lstGBIDs.pop()
            lstAtomIDs = self.__GrainBoundaries[intGBID].GetAtomIDs()
            lstAdjacentGrainBoundaries = self.GetAdjacentGrainBoundaries(intGBID)
            for i in lstAdjacentGrainBoundaries:
                lstOverlapAtoms = list(set(lstAtomIDs) & set(self.__GrainBoundaries[i].GetAtomIDs()))
                if len(lstOverlapAtoms) > 0:
                    self.__GrainBoundaries[intGBID].RemoveAtomIDs(lstOverlapAtoms)
                    self.__GrainBoundaries[i].RemoveAtomIDs(lstOverlapAtoms)
                    arrAtoms = self.GetAtomsByID(lstOverlapAtoms)[:,0:4]
                    for j in arrAtoms:
                        arrPosition = j[1:4]
                        intAtomID = j[0].astype('int')
                        lstClosestGrains = []
                        lstGrainDistances = []
                        lstGrainLabels = []
                        for k in self.__GrainLabels:
                            arrPeriodicVariants = self.PeriodicEquivalents(arrPosition)
                            lstDistances,lstIndices = self.__Grains[k].query(arrPeriodicVariants,1)
                            lstGrainDistances.append(min(lstDistances))
                            lstGrainLabels.append(k)
                        lstIndices = np.argsort(lstGrainDistances)
                        lstClosestGrains.append(lstGrainLabels[lstIndices[0]])
                        lstClosestGrains.append(lstGrainLabels[lstIndices[1]])
                        lstClosestGrains = sorted(lstClosestGrains)
                        if lstClosestGrains == self.__GrainBoundaries[intGBID].GetAdjacentGrains():
                            self.__GrainBoundaries[intGBID].AddAtomIDs([intAtomID])
                        elif lstClosestGrains == self.__GrainBoundaries[i].GetAdjacentGrains():
                            self.__GrainBoundaries[i].AddAtomIDs([intAtomID])
    def FindJunctionLines(self):
        for i in self.__JunctionLineIDs:
            lstJLAtomsIDs = []
            lstCloseAtoms = []
            for s in self.__JunctionLines[i].GetAdjacentGrainBoundaries():
                lstCloseAtoms.append(set(self.__GrainBoundaries[s].GetAtomIDs()))
            lstOverlapAtoms = list(reduce(lambda x,y: x & y,lstCloseAtoms))
            for t in self.__JunctionLines[i].GetAdjacentGrainBoundaries():
                self.__GrainBoundaries[t].RemoveAtomIDs(lstOverlapAtoms)
            arrAtoms = self.GetAtomsByID(lstOverlapAtoms)
            for j in arrAtoms:
                arrPosition = j[1:4]
                intID = j[0].astype('int')
                lstVectors = []
                lstGrainDistances = []
                lstGrainLabels = []
                lstClosestGrains = []
                for k in self.__GrainLabels:
                    arrPeriodicVariants = self.PeriodicEquivalents(arrPosition)
                    lstDistances,lstIndices = self.__Grains[k].query(arrPeriodicVariants,1)
                    intIndex = np.argmin(lstDistances)
                    lstGrainDistances.append(lstDistances[intIndex])
                    lstGrainLabels.append(k)
                    lstVectors.append(self.PeriodicShiftCloser(arrPosition,self.__Grains[k].data[lstIndices[intIndex]]))
                lstIndices = np.argsort(lstGrainDistances)
                lstClosestGrains.append(lstGrainLabels[lstIndices[0]])
                lstClosestGrains.append(lstGrainLabels[lstIndices[1]])
                lstClosestGrains = sorted(lstClosestGrains)
                fltGBLength = np.linalg.norm(lstVectors[lstIndices[0]]-lstVectors[lstIndices[1]])
                if fltGBLength < self.__LatticeParameter/2:
                    warnings.warn('Estimated GB length is only ' + str(fltGBLength) + ' Anstroms')
                if  np.all(np.array(lstGrainDistances) < fltGBLength):
                    lstJLAtomsIDs.append(intID)
                else:
                    intCounter = 0
                    blnFound = False
                    while (intCounter < len(self.__JunctionLines[i].GetAdjacentGrainBoundaries()) and not(blnFound)):
                        intGrainBoundary = self.__JunctionLines[i].GetAdjacentGrainBoundaries()[intCounter]
                        if lstClosestGrains == self.__GrainBoundaries[intGrainBoundary].GetAdjacentGrains():
                            self.__GrainBoundaries[intGrainBoundary].AddAtomIDs([intID])
                            blnFound = True
                        else:
                            intCounter += 1
                self.__JunctionLines[i].SetAtomIDs(lstJLAtomsIDs)        
    def GetGrainBoundaryAtomIDs(self, inGrainBoundaries = None):
        lstGrainBoundaryIDs = []
        if inGrainBoundaries is None:
            for j in self.__GrainBoundaryIDs:
                lstGrainBoundaryIDs.extend(self.__GrainBoundaries[j].GetAtomIDs())
            return lstGrainBoundaryIDs
        elif isinstance(inGrainBoundaries, list):
            for k in inGrainBoundaries:
                lstGrainBoundaryIDs.extend(self.__GrainBoundaries[k].GetAtomIDs())
            return lstGrainBoundaryIDs
        elif isinstance(inGrainBoundaries, int):
            return self.__GrainBoundaries[inGrainBoundaries].GetAtomIDs()
    def GetJunctionLineAtomIDs(self, intJunctionLine = None):
        if intJunctionLine is None:
            lstJunctionLineIDs = []
            for j in self.__JunctionLineIDs:
                lstJunctionLineIDs.extend(self.__JunctionLines[j].GetAtomIDs())
            return lstJunctionLineIDs
        else:
            return self.__JunctionLines[intJunctionLine].GetAtomIDs()
    def GetGrainAtomIDs(self, intGrainNumber: int):
        lstGrainAtoms = list(np.where(self.GetColumnByName('GrainNumber').astype('int') == intGrainNumber)[0])
        return self.GetAtomData()[lstGrainAtoms,0].astype('int')
    def GetGrainLabels(self):
        return self.__GrainLabels
    def GetJunctionLineMeshPoints(self, intJunctionLine = None):
        if intJunctionLine is None:
            lstJunctionLinePoints = []
            for j in self.__JunctionLineIDs:
                lstJunctionLinePoints.append(self.__JunctionLines[j].GetMeshPoints())
            return np.vstack(lstJunctionLinePoints)
        else:
            return self.__JunctionLines[intJunctionLine].GetMeshPoints()
    def GetJunctionLineAdjustedMeshPoints(self, intJunctionLine = None):
        if intJunctionLine is None:
            lstJunctionLinePoints = []
            for j in self.__JunctionLineIDs:
                lstJunctionLinePoints.append(self.__JunctionLines[j].GetAdjustedMeshPoints())
            return np.vstack(lstJunctionLinePoints)
        else:
            return self.__JunctionLines[intJunctionLine].GetMeshPoints()
    def GetGrainBoundaryAdjustedMeshPoints(self, intGrainBoundary = None):
        if intGrainBoundary is None:
            lstGrainBoundaryIDs = []
            for j in self.__GrainBoundaryIDs:
                lstGrainBoundaryIDs.append(self.__GrainBoundaries[j].GetAdjustedMeshPoints())
            return np.vstack(lstGrainBoundaryIDs)
        else:
            return self.__GrainBoundaries[intGrainBoundary].GetAdjustedMeshPoints()
    def GetGrainBoundaryMeshPoints(self, intGrainBoundary = None):
        if intGrainBoundary is None:
            lstGrainBoundaryIDs = []
            for j in self.__GrainBoundaryIDs:
                lstGrainBoundaryIDs.append(self.__GrainBoundaries[j].GetMeshPoints())
            return np.vstack(lstGrainBoundaryIDs)
        else:
            return self.__GrainBoundaries[intGrainBoundary].GetMeshPoints()
    def GetGrainBoundary(self, intGrainBoundary):
        return self.__GrainBoundaries[intGrainBoundary]
    def GetJunctionLine(self, intJunctionLine):
        return self.__JunctionLines[intJunctionLine]
    def GetJunctionLineIDs(self):
        return self.__JunctionLineIDs
    def GetGrainBoundaryIDs(self):
        return self.__GrainBoundaryIDs
    def GetAdjacentGrainBoundaries(self, intGrainBoundary: int):
        lstAdjacentGrainBoundaries = []
        lstJunctionLines = self.__GrainBoundaries[intGrainBoundary].GetAdjacentJunctionLines()
        for k in lstJunctionLines:
            lstAdjacentGrainBoundaries.extend(self.__JunctionLines[k].GetAdjacentGrainBoundaries())
        lstAdjacentGrainBoundaries =  list(np.unique(lstAdjacentGrainBoundaries))
        if intGrainBoundary in lstAdjacentGrainBoundaries:
            lstAdjacentGrainBoundaries.remove(intGrainBoundary)
        return lstAdjacentGrainBoundaries
    def WriteDefectData(self, strFileName: str):
        with open(strFileName, 'w') as fdata:
            for i in self.__JunctionLineIDs:
                fdata.write('Junction Line \n')
                fdata.write('{} \n'.format(i))
                fdata.write('Mesh Points \n')
                fdata.write('{} \n'.format(self.__JunctionLines[i].GetMeshPoints().tolist()))
                fdata.write('Adjacent Grains \n')
                fdata.write('{} \n'.format(self.__JunctionLines[i].GetAdjacentGrains()))
                fdata.write('Adjacent Grain Boundaries \n')
                fdata.write('{} \n'.format(self.__JunctionLines[i].GetAdjacentGrainBoundaries()))
                fdata.write('Periodic Directions \n')
                fdata.write('{} \n'.format(self.__JunctionLines[i].GetPeriodicDirections()))
                fdata.write('Atom IDs \n')
                fdata.write('{} \n'.format(self.__JunctionLines[i].GetAtomIDs()))
                fdata.write('Volume \n')
                fdata.write('{} \n'.format(self.__JunctionLines[i].GetVolume()))
                fdata.write('PE \n')
                fdata.write('{} \n'.format(self.__JunctionLines[i].GetPE()))
            for k in self.__GrainBoundaryIDs:
                fdata.write('Grain Boundary \n')
                fdata.write('{} \n'.format(k))
                fdata.write('Mesh Points \n')
                fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetMeshPoints().tolist()))
                fdata.write('Adjacent Grains \n')
                fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetAdjacentGrains()))
                fdata.write('Adjacent Junction Lines \n')
                fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetAdjacentJunctionLines()))
                fdata.write('Periodic Directions \n')
                fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetPeriodicDirections()))
                fdata.write('Atom IDs \n')
                fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetAtomIDs()))
                fdata.write('Volume \n')
                fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetVolume()))
            fdata.write('Grain Numbers \n')
            fdata.write('{}'.format(self.GetColumnByIndex(self.__intGrainNumber).tolist()))
    def ReadInDefectData(self, strFilename: str):
            with open(strFilename) as fdata:
                while True:
                    try:
                        line = next(fdata).strip()
                    except StopIteration as EndOfFile:
                        break
                    if line == "Junction Line":
                        intJL = int(next(fdata).strip())
                        line = next(fdata).strip()
                        if line == "Mesh Points":
                            line = next(fdata).strip()    
                            arrMeshPoints = np.array(eval(line))
                        objJunctionLine = gl.GeneralJunctionLine(arrMeshPoints, intJL)
                        line = next(fdata).strip()
                        if line == "Adjacent Grains":
                            line = next(fdata).strip()    
                            objJunctionLine.SetAdjacentGrains(eval(line))
                        line = next(fdata).strip()
                        if line == "Adjacent Grain Boundaries":
                            line = next(fdata).strip()
                            objJunctionLine.SetAdjacentGrainBoundaries(eval(line))
                        line = next(fdata).strip()
                        if line == "Periodic Directions":
                            line = next(fdata).strip()
                            objJunctionLine.SetPeriodicDirections(eval(line))
                        line = next(fdata).strip()
                        if line == "Atom IDs":
                            line = next(fdata).strip()
                            objJunctionLine.SetAtomIDs(eval(line))
                        line = next(fdata).strip()
                        if line == "Volume":
                            line = next(fdata).strip()
                            objJunctionLine.SetVolume(eval(line))
                        self.__JunctionLines[intJL] = objJunctionLine
                    elif line == "Grain Boundary":
                        intGB = int(next(fdata).strip())
                        line = next(fdata).strip()
                        if line == "Mesh Points":
                            line = next(fdata).strip()    
                            arrMeshPoints = np.array(eval(line))
                        objGrainBoundary = gl.GeneralGrainBoundary(arrMeshPoints, intGB)
                        line = next(fdata).strip()
                        if line == "Adjacent Grains":
                            line = next(fdata).strip()    
                            objGrainBoundary.SetAdjacentGrains(eval(line))
                        line = next(fdata).strip()
                        if line == "Adjacent Junction Lines":
                            line = next(fdata).strip()
                            objGrainBoundary.SetAdjacentJunctionLines(eval(line))
                        line = next(fdata).strip()
                        if line == "Periodic Directions":
                            line = next(fdata).strip()
                            objGrainBoundary.SetPeriodicDirections(eval(line))
                        line = next(fdata).strip()
                        if line == "Atom IDs":
                            line = next(fdata).strip()
                            objGrainBoundary.SetAtomIDs(eval(line))
                        line = next(fdata).strip()
                        if line == "Volume":
                            line = next(fdata).strip()
                            objGrainBoundary.SetVolume(eval(line))
                        self.__GrainBoundaries[intGB] = objGrainBoundary
                    elif line == "Grain Numbers": 
                        line = next(fdata).strip()
                        self.AppendGrainNumbers(eval(line))
            self.MakeGrainTrees()
            self.__JunctionLineIDs = list(self.__JunctionLines.keys())
            self.__GrainBoundaryIDs = list(self.__GrainBoundaries.keys())

class QuantisedCuboidPoints(object):
    def __init__(self, in3DPoints: np.array, inBasisConversion: np.array, inCellVectors: np.array, arrGridDimensions: np.array, intWrapper = None):
        arrCuboidCellVectors = np.matmul(inCellVectors,inBasisConversion)
        arrModArray = np.zeros([3])
        for i in range(len(arrCuboidCellVectors)): #calculate a scaling factor that splits into exact integer multiples
            intValue = np.round(np.linalg.norm(arrCuboidCellVectors[i])/arrGridDimensions[i]).astype('int')
            arrModArray[i] = intValue
            arrGridDimensions[i] = np.linalg.norm(arrCuboidCellVectors[i])/intValue
        self.__BasisConversion = inBasisConversion
        self.__InverseBasisConversion = np.linalg.inv(inBasisConversion)
        self.__GridDimensions = arrGridDimensions
        self.__SizeParameter = np.min(arrGridDimensions)
        arrModArray = arrModArray.astype('int')
        self.__InverseScaling = np.diag(arrGridDimensions)
        self.__Scaling = np.linalg.inv(self.__InverseScaling)
        arrCuboidPoints = np.matmul(in3DPoints, inBasisConversion)
        arrCuboidPoints = np.matmul(arrCuboidPoints, self.__Scaling)    
        arrValues =  np.zeros([arrModArray[0],arrModArray[1],arrModArray[2]])
        self.__JunctionLinesArray = np.copy(arrValues)
        self.__GrainBoundariesArray = np.copy(arrValues)
        self.__JunctionLines = []
        nx,ny,nz = np.shape(arrValues)
        self.__ModArray = arrModArray
        arrCoordinates = (np.linspace(0,nx-1,nx),np.linspace(0,ny-1,ny),
                             np.linspace(0,nz-1,nz))
        for m in range(len(arrCuboidPoints)):
            k = arrCuboidPoints[m]
            k = np.round(k,0).astype('int')
            k = np.mod(k,self.__ModArray)
            arrValues[k[0],k[1], k[2]] += 1
        objInterpolate = RegularGridInterpolator(arrCoordinates, arrValues, method = 'linear')
        self.__DefectPositions = arrValues.astype('bool').astype('int')
        self.__Coordinates = gf.CreateCuboidPoints(np.array([[0,nx-1],[0,ny-1],[0,nz-1]]))
        arrOut = objInterpolate(self.__Coordinates)
        arrOut = np.reshape(arrOut,arrModArray)
        arrOut = gaussian(arrOut,1, mode='wrap',multichannel = False)
        arrOut = arrOut > np.mean(arrOut)
        self.__BinaryArray = arrOut.astype('bool').astype('int')
        self.__BinaryArray = remove_small_holes(self.__BinaryArray.astype('bool'), 4).astype('int')
        self.__Grains  = measure.label(arrOut == 0).astype('int')
        lstGrainLabels = list(np.unique(self.__Grains))
        lstGrainLabels.remove(0)
        self.__GrainLabels = lstGrainLabels
        if intWrapper is not(None):
            self.__blnPeriodic = True
            self.__WrapperWidth = int(intWrapper)
            self.CheckPeriodicGrains() #check to see if any grains connect over the simulation cell boundary
            self.MergeEquivalentGrains() #if two grains are periodically linked then merge them into one
           # self.ExtendArrayPeriodically(intWrapper)#extend the cuboid by intwrapper using a periodic copy
            self.ExpandGrains() #expand all the grains until the grain boundaries are dissolved
    def MergeEquivalentGrains(self):
        for j in self.__EquivalentGrains:
            if len(j) > 0:
                for k in j[1:]:
                    self.__Grains[self.__Grains == k] = j[0]
        lstValues = list(np.unique(self.__Grains))
        lstValues.remove(0)
        self.__GrainLabels = lstValues        
    def ExtendArrayPeriodically(self, intWrapperWidth: int):
        n = intWrapperWidth
        self.__ExtendedGrains = np.zeros([self.__ModArray[0]+2*n,self.__ModArray[1]+2*n, self.__ModArray[2]+2*n])
        self.__ExtendedGrains[n:-n, n:-n, n:-n] = self.__Grains
        self.__ExtendedGrains[0:n, n:-n, n:-n] = self.__Grains[-n:,:,:]
        self.__ExtendedGrains[-n:, n:-n, n:-n] = self.__Grains[:n,:,:]
        self.__ExtendedGrains[:,0:n,:] = self.__ExtendedGrains[:,-2*n:-n,:]
        self.__ExtendedGrains[:,-n:,:] = self.__ExtendedGrains[:,n:2*n,:]
        self.__ExtendedGrains[:,0:n,:] = self.__ExtendedGrains[:,-2*n:-n,:]
        self.__ExtendedGrains[:,-n:,:] = self.__ExtendedGrains[:,n:2*n,:]
        self.__ExtendedGrains[:,:,0:n] = self.__ExtendedGrains[:,:,-2*n:-n]
        self.__ExtendedGrains[:,:,-n:] = self.__ExtendedGrains[:,:,n:2*n]
        self.__ExtendedGrains = self.__ExtendedGrains.astype('int') 
    def CheckPeriodicGrains(self):
        lstEquivalentGrains = []
        for intCurrentGrain in self.__GrainLabels:
            lstMatchedGrains = [intCurrentGrain]
            for j in range(3):
                arrCellBoundaryPoints = self.CellBoundaryPoints(j, self.GetGrainPoints(intCurrentGrain),True)
                arrCellBoundaryPoints[:,j] = arrCellBoundaryPoints[:,j] + self.__ModArray[j] -1
                lstMatchedGrains.extend(list(np.unique(self.__Grains[arrCellBoundaryPoints[:,0],arrCellBoundaryPoints[:,1],arrCellBoundaryPoints[:,2]])))
                arrCellBoundaryPoints = self.CellBoundaryPoints(j, self.GetGrainPoints(intCurrentGrain),False)
                arrCellBoundaryPoints[:,j] = np.zeros(len(arrCellBoundaryPoints[:,j]))
                lstMatchedGrains.extend(list(np.unique(self.__Grains[arrCellBoundaryPoints[:,0],arrCellBoundaryPoints[:,1],arrCellBoundaryPoints[:,2]])))
            lstMatchedGrains = list(np.unique(lstMatchedGrains))
            if 0 in lstMatchedGrains:
                lstMatchedGrains.remove(0)
            lstEquivalentGrains.append(lstMatchedGrains)
        self.__EquivalentGrains = lstEquivalentGrains
        return lstEquivalentGrains
    def CellBoundaryPoints(self, intAxis: int, inPoints: np.array, blnZeroFace = True)->np.array: #returns boundary points
        lstIndices = []
        if blnZeroFace:
            lstIndices = np.where(inPoints[:,intAxis] == 0)[0] 
        else:
            lstIndices  = np.where(inPoints[:,intAxis] == self.__ModArray[intAxis]-1)[0]
        if len(lstIndices) > 0: 
            lstIndices = np.unique(lstIndices)
        return inPoints[lstIndices]
    def ReturnGrains(self, inPoints: np.array, blnExpanded = False)->list:
        inPoints = np.matmul(inPoints, self.__Scaling)
        inPoints = np.matmul(inPoints, self.__BasisConversion)
        inPoints = np.round(inPoints, 0).astype('int')
        inPoints = np.mod(inPoints, self.__ModArray)
        #print(np.argwhere(self.__ExpandedGrains == 0)) debugging only
        if blnExpanded:
            return list(self.__ExpandedGrains[inPoints[:,0],inPoints[:,1],inPoints[:,2]])
        else:
            arrZeros = np.argwhere(self.__BinaryArray == 0)
            self.__ReturnGrains = np.copy(self.__ExpandedGrains)
            for j in arrZeros:
                self.__ReturnGrains[j[0],j[1],j[2]] = 0
            return list(self.__ReturnGrains[inPoints[:,0],inPoints[:,1],inPoints[:,2]])                     
    def FindJunctionLines(self):
        lstGrainBoundaryList = []
        for j in self.__Coordinates:
            j = j.astype('int')
            arrBox  = self.__ExpandedGrains[gf.WrapAroundSlice(np.array([[j[0],j[0]+2],[j[1],j[1]+2],[j[2],j[2]+2]]),self.__ModArray)]
            lstValues = list(np.unique(arrBox))
            if len(lstValues) > 2:
                self.__JunctionLinesArray[j[0],j[1],j[2]] = len(lstValues)
            elif len(lstValues) ==2:
                if lstValues not in lstGrainBoundaryList:
                    lstGrainBoundaryList.append(lstValues)
                self.__GrainBoundariesArray[j[0],j[1],j[2]] = 1 + lstGrainBoundaryList.index(lstValues)
        self.__JunctionLinesArray = measure.label(self.__JunctionLinesArray).astype('int')
        self.CheckPeriodicJunctionLines()
        self.__GrainBoundariesArray = measure.label(self.__GrainBoundariesArray).astype('int')
        self.CheckPeriodicGrainBoundaries() 
    def CheckPeriodicJunctionLines(self):
        arrJLPoints = np.argwhere(self.__JunctionLinesArray > 0)
        setEquivalentJunctionLines = set()
        for j in range(3):
            lstIndicesLower = np.where(arrJLPoints[:,j] == 0)[0]
            arrPointsLower = arrJLPoints[lstIndicesLower]
            lstIndicesUpper = np.where(arrJLPoints[:,j] == self.__ModArray[j]-1)[0]
            arrPointsUpper  = arrJLPoints[lstIndicesUpper]
            arrPointsUpper[:,j] = arrPointsUpper[:,j] - np.ones(len(arrPointsUpper))*self.__ModArray[j]
            arrDistanceMatrix = spatial.distance_matrix(arrPointsLower, arrPointsUpper)
            arrClosePoints = np.argwhere(arrDistanceMatrix < 2) 
            for j in arrClosePoints:
                tupPairs = (self.__JunctionLinesArray[tuple(zip(arrPointsLower[j[0]]))][0], self.__JunctionLinesArray[tuple(zip(arrPointsUpper[j[1]]))][0])
                if tupPairs[0] != tupPairs[1]:
                    setEquivalentJunctionLines.add(tupPairs)
        for k in setEquivalentJunctionLines: #merge the periodic values
            self.__JunctionLinesArray[self.__JunctionLinesArray == k[1]] = k[0] 
        lstValues = list(np.unique(self.__JunctionLinesArray)) #renumber the array sequentially for convenience starting at 1
        if 0 in lstValues:
            lstValues.remove(0)
        for intIndex, intValue in enumerate(lstValues):
            self.__JunctionLinesArray[self.__JunctionLinesArray == intValue] = intIndex+1 
        lstValues = list(np.unique(self.__JunctionLinesArray))
        if 0 in lstValues:
            lstValues.remove(0)
        self.__JunctionLineIDs = lstValues             
    def CheckPeriodicGrainBoundaries(self):
        arrGBPoints = np.argwhere(self.__GrainBoundariesArray > 0)
        setEquivalentGrainBoundaries = set()
        for j in range(3):
            lstIndicesLower = np.where(arrGBPoints[:,j] == 0)[0]
            arrPointsLower = arrGBPoints[lstIndicesLower]
            lstIndicesUpper = np.where(arrGBPoints[:,j] == self.__ModArray[j]-1)[0]
            arrPointsUpper  = arrGBPoints[lstIndicesUpper]
            arrPointsUpper[:,j] = arrPointsUpper[:,j] - np.ones(len(arrPointsUpper))*self.__ModArray[j]
            arrDistanceMatrix = spatial.distance_matrix(arrPointsLower, arrPointsUpper)
            arrClosePoints = np.argwhere(arrDistanceMatrix < 2) #returns the indices of the other grain boundaries 
            for j in arrClosePoints:
                tupPairs = (self.__GrainBoundariesArray[tuple(zip(arrPointsLower[j[0]]))][0], self.__GrainBoundariesArray[tuple(zip(arrPointsUpper[j[1]]))][0])
                if tupPairs[0] != tupPairs[1]:
                    setJunctionLines = set(self.GetAdjacentJunctionLines(tupPairs[0]))
                    lstJunctionLines = self.GetAdjacentJunctionLines(tupPairs[1])
                    setJunctionLines = setJunctionLines.intersection(lstJunctionLines)
                    if len(setJunctionLines) >= 1:
                        if len(setJunctionLines) > 1:
                            warnings.warn("Two grain boundaries sharing more than one junction line")
                        lstGrainBoundaries = self.GetAdjacentGrainBoundaries(setJunctionLines.pop())
                        if (tupPairs[0] not in lstGrainBoundaries) and (tupPairs[1] not in lstGrainBoundaries): 
                            setEquivalentGrainBoundaries.add(tupPairs)
                    elif len(setJunctionLines) == 0:
                        setEquivalentGrainBoundaries.add(tupPairs)
        for k in setEquivalentGrainBoundaries: #merge the periodic values
            self.__GrainBoundariesArray[self.__GrainBoundariesArray == k[1]] = k[0] 
        lstValues = list(np.unique(self.__GrainBoundariesArray)) #renumber the array sequentially for convenience starting at 1
        if 0 in lstValues:
            lstValues.remove(0)
        for intIndex, intValue in enumerate(lstValues):
            self.__GrainBoundariesArray[self.__GrainBoundariesArray == intValue] = intIndex+1 
        lstValues = list(np.unique(self.__GrainBoundariesArray))
        if 0 in lstValues:
            lstValues.remove(0)
        self.__GrainBoundaryIDs = lstValues
    def ExpandGrains(self, n=3): 
        self.__ExpandedGrains = np.copy(self.__Grains)
        arrGBPoints = np.argwhere(self.__Grains == 0)
        arrGBPoints = arrGBPoints.astype('int')
        arrIndices = gf.CreateCuboidPoints(np.array([[-n,n],[-n,n],[-n,n]]))
        arrIndices = np.matmul(np.matmul(arrIndices, self.__InverseBasisConversion),self.__InverseScaling)
        arrDistances = np.reshape(np.array(list(map(np.linalg.norm, arrIndices))),(2*n+1,2*n+1,2*n+1))
        if len(arrGBPoints) > 0 :
            for j in arrGBPoints:
                if np.all(np.mod(j -n*np.ones(3), self.__ModArray) == j-n*np.ones(3)) and np.all(np.mod(j +(n+1)*np.ones(3), self.__ModArray) == j+(n+1)*np.ones(3)):
                        arrBox = self.__Grains[j[0]-n:j[0]+n+1,j[1]-n:j[1]+n+1, j[2]-n:j[2]+n+1]
                else:
                    arrBox = self.__Grains[gf.WrapAroundSlice(np.array([[j[0]-n,j[0]+n+1],[j[1]-n,j[1]+n+1], [j[2]-n,j[2]+n+1]]),self.__ModArray)]
                    arrBox = np.reshape(arrBox,(2*n+1,2*n+1,2*n+1))
                arrGrainPoints = np.where(arrBox != 0) 
                if np.size(arrGrainPoints) > 0:
                    arrGrainDistances = arrDistances[arrGrainPoints]
                    arrClosestPoint = np.where(arrGrainDistances == np.min(arrGrainDistances))
                    arrGrainPoints = np.transpose(np.array(arrGrainPoints))
                    arrPosition = arrGrainPoints[arrClosestPoint]
                    if len(arrPosition) == 1:    
                        self.__ExpandedGrains[j[0],j[1],j[2]] = arrBox[arrPosition[0,0],arrPosition[0,1],arrPosition[0,2]]
        else:
            warnings.warn("Unable to find any defective regions")
        arrGBPoints = np.argwhere(self.__ExpandedGrains == 0)
        n = 1
        counter = 0
        while (counter < len(arrGBPoints)): 
            l = arrGBPoints[counter]
            arrBox = self.__ExpandedGrains[gf.WrapAroundSlice(np.array([[l[0]-n,l[0]+n+1],[l[1]-n,l[1]+n+1], [l[2]-n,l[2]+n+1]]),self.__ModArray)]
            arrBox = arrBox[arrBox != 0]
            if len(arrBox) > 0:
                arrValues, arrCounts = np.unique(arrBox, return_counts = True)
                intGrain = arrValues[np.argmax(arrCounts)]
                if len(np.argwhere(arrValues == intGrain)) ==1 or n > 5:
                    self.__ExpandedGrains[l[0],l[1],l[2]] = intGrain
                    counter += 1
                else:
                    n += 1
        intNumberOfDefects = len(np.argwhere(self._QuantisedCuboidPoints__ExpandedGrains == 0))
        if intNumberOfDefects > 0:
            warnings.warn('Error expanding grains. The ExpandedGrains array still has ' + str(intNumberOfDefects) + ' defect(s)')
    def GetExpandedGrains(self):
        return self.__ExpandedGrains
    def GetGrainBoundaryPoints(self, intGrainBoundaryID = None):
        if intGrainBoundaryID is None:
            return np.matmul(np.matmul(np.argwhere(self.__GrainBoundariesArray.astype('int') != 0) +np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion) 
        elif intGrainBoundaryID in self.__GrainBoundaryIDs: 
            return np.matmul(np.matmul(np.argwhere(self.__GrainBoundariesArray.astype('int') == intGrainBoundaryID)+np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion)
        else:
            warnings.warn(str(intGrainBoundaryID) + ' is an invalid grain boundary ID')
    def GetJunctionLinePoints(self, intJunctionLineID = None)->np.array:
        if intJunctionLineID is None:
            return np.matmul(np.matmul(np.argwhere(self.__JunctionLinesArray.astype('int') != 0) +np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion) 
        elif intJunctionLineID in self.__JunctionLineIDs: 
            return np.matmul(np.matmul(np.argwhere(self.__JunctionLinesArray.astype('int') == intJunctionLineID)+np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion)
        else:
            warnings.warn(str(intJunctionLineID) + ' is an invalid junction line ID')
    def GetJunctionLineIDs(self)->list:
        return self.__JunctionLineIDs
    def GetGrainBoundaryIDs(self)->list:
        return self.__GrainBoundaryIDs
    def GetGrainPoints(self, intGrainNumber):
        return np.argwhere(self.__Grains == intGrainNumber)
    def GetExpandedGrainPoints(self, intGrainNumber: int):
        return np.argwhere(self.__ExpandedGrains == intGrainNumber)
    def GetExtendedGrainPoints(self, intGrainNumber: int):
        return np.argwhere(self.__ExtendedGrains == intGrainNumber)
    def GetExtendedGrains(self):
        return self.__ExtendedGrains
    def GetBinaryArray(self):
        return self.__BinaryArray
    def GetGrains(self):
        return self.__Grains
    def GetGrainLabels(self):
        return self.__GrainLabels
    def GetAdjacentGrains(self, intID: int, strType: str)->list:
        if strType == 'JunctionLine':
            arrPoints = np.argwhere(self.__JunctionLinesArray == intID)
        elif strType =='GrainBoundary':   
            arrPoints = np.argwhere(self.__GrainBoundariesArray == intID)
        if len(arrPoints) > 0:
            l = arrPoints[0]
            arrBox = self.__ExpandedGrains[gf.WrapAroundSlice(np.array([[l[0],l[0]+2],[l[1],l[1]+2], [l[2],l[2]+2]]),self.__ModArray)]
            return list(np.unique(arrBox))
        else:
            warnings.warn('Invalid ' +str(strType) + ' ID')
    def GetAdjacentJunctionLines(self, intGrainBoundary: int)->list:
        arrPoints  = np.argwhere(self.__GrainBoundariesArray == intGrainBoundary)
        lstValues = []
        if len(arrPoints) > 0:
            for l in arrPoints:
                lstValues.extend(self.__JunctionLinesArray[gf.WrapAroundSlice(np.array([[l[0]-1,l[0]+2],[l[1]-1,l[1]+2], [l[2]-1,l[2]+2]]),self.__ModArray)])
        lstValues = list(np.unique(lstValues))
        if 0 in lstValues:
            lstValues.remove(0)
        return lstValues
    def GetAdjacentGrainBoundaries(self, intJunctionLine)->list:
        arrPoints  = np.argwhere(self.__JunctionLinesArray == intJunctionLine)
        lstValues = []
        if len(arrPoints) > 0:
            for l in arrPoints:
                lstValues.extend(self.__GrainBoundariesArray[gf.WrapAroundSlice(np.array([[l[0]-1,l[0]+2],[l[1]-1,l[1]+2], [l[2]-1,l[2]+2]]),self.__ModArray)])
        lstValues = list(np.unique(lstValues))
        if 0 in lstValues:
            lstValues.remove(0)
        return lstValues
    def GetPeriodicExtensions(self, intIndex, strType): #returns index of the periodic cell vector
        lstPeriodicDirections = []
        if strType == 'JunctionLine':
            arrPoints = np.argwhere(self.__JunctionLinesArray == intIndex)
        elif strType == 'GrainBoundary':
            arrPoints = np.argwhere(self.__GrainBoundariesArray == intIndex)
        for j in range(3):
            if np.any(arrPoints[:,j] == 0) and np.any(arrPoints[:,j] == self.__ModArray[j] - 1):
                lstPeriodicDirections.append(j)
        return lstPeriodicDirections
