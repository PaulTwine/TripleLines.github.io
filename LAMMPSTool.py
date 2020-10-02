import re
import numpy as np
import GeometryFunctions as gf
import GeneralLattice as gl
from scipy import spatial, optimize, ndimage, stats 
from skimage.morphology import skeletonize, thin, medial_axis, remove_small_holes, remove_small_objects, skeletonize_3d, binary_dilation
from scipy.cluster.vq import kmeans,vq
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from skimage.filters import gaussian
from skimage import measure
from sklearn.cluster import DBSCAN
#import hdbscan
#import shapely as sp
import copy
import warnings
from functools import reduce
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances

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
        self.__BoundaryTypes = []
        self.__NumberOfAtoms = intNumberOfAtoms
        self.__NumberOfColumns = len(lstColumnNames)
        self.__TimeStep = fltTimeStep
        self.__AtomData = np.zeros([intNumberOfAtoms,self.__NumberOfColumns])
        self.__ColumnNames = lstColumnNames
        self.SetBoundBoxLabels(lstBoundaryType)
        self.SetBoundBoxDimensions(lstBounds)
    def GetBoundaryTypes(self):
        return self.__BoundaryTypes
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
    def PeriodicEquivalents(self, inPositionVector: np.array)->np.array: #Moved to GeometryFunctions.py for access from other   
        # arrVector = np.array([inPositionVector])                         classes
        # arrCellCoordinates = np.matmul(inPositionVector, self.__BasisConversion)
        # for i,strBoundary in enumerate(self.__BoundaryTypes):
        #     if strBoundary == 'pp':
        #          if  arrCellCoordinates[i] > 0.5:
        #              arrVector = np.append(arrVector, np.subtract(arrVector,self.__CellVectors[i]),axis=0)
        #          elif arrCellCoordinates[i] <= 0.5:
        #              arrVector = np.append(arrVector, np.add(arrVector,self.__CellVectors[i]),axis=0)                  
        # return arrVector
        return gf.PeriodicEquivalents(inPositionVector,  self.__CellVectors,self.__BasisConversion, self.__BoundaryTypes)
    def MoveToSimulationCell(self, inPositionVector: np.array)->np.array:
        return gf.WrapVectorIntoSimulationCell(self.__CellBasis, self.__BasisConversion, inPositionVector)
    def PeriodicShiftAllCloser(self, inFixedPoint: np.array, inAllPointsToShift: np.array)->np.array:
        #arrPoints = np.array(list(map(lambda x: self.PeriodicShiftCloser(inFixedPoint, x), inAllPointsToShift)))
        return gf.PeriodicShiftAllCloser(inFixedPoint,inAllPointsToShift, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
    def PeriodicShiftCloser(self, inFixedPoint: np.array, inPointToShift: np.array)->np.array:
        #arrPeriodicVectors = self.PeriodicEquivalents(inPointToShift)
        #fltDistances = list(map(np.linalg.norm, np.subtract(arrPeriodicVectors, inFixedPoint)))
        #return arrPeriodicVectors[np.argmin(fltDistances)]
        return gf.PeriodicShiftCloser(inFixedPoint, inPointToShift, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
    def MakePeriodicDistanceMatrix(self, inVectors1: np.array, inVectors2: np.array)->np.array:
        return gf.MakePeriodicDistanceMatrix(inVectors1, inVectors2, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
        # arrPeriodicDistance = np.zeros([len(inVector1), len(inVector2)])
        # for j in range(len(inVector1)):
        #     for k in range(len(inVector2)):
        #         arrPeriodicDistance[j,k] = self.PeriodicMinimumDistance(inVector1[j],inVector2[k])
        # return arrPeriodicDistance
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        return gf.PeriodicMinimumDistance(inVector1, inVector2, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
        #inVector2 = self.PeriodicShiftCloser(inVector1, inVector2)
        #return np.linalg.norm(inVector2-inVector1, axis=0)
    def StandardiseOrientationData(self):
        self.__AtomData[:, [self.GetColumnNames().index('OrientationX'),self.GetColumnNames().index('OrientationY'),self.GetColumnNames().index('OrientationZ'), self.GetColumnNames().index('OrientationW')]]=np.apply_along_axis(gf.FCCQuaternionEquivalence,1,self.GetOrientationData()) 
    def GetOrientationData(self)->np.array:
        return (self.__AtomData[:, [self.GetColumnNames().index('OrientationX'),self.GetColumnNames().index('OrientationY'),self.GetColumnNames().index('OrientationZ'), self.GetColumnNames().index('OrientationW')]])  
    def GetData(self, inDimensions: np.array, lstOfColumns):
        return np.where(self.__AtomData[:,lstOfColumns])
    def GetBoundingBox(self):
        return np.array([self.GetCellBasis()[0,0]+self.GetCellBasis()[1,0]+self.GetCellBasis()[2,0],
        self.GetCellBasis()[1,1]+self.GetCellBasis()[2,1], self.GetCellBasis()[2,2]])
    def GetTimeStep(self):
        return self.__TimeStep

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
        self.__fltGrainTolerance = 3.14
        self.__DefectiveAtomIDs = []
        self.__NonDefectiveAtomIDs = []
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
        self.__DefectiveAtomIDs = []
        self.__NonDefectiveAtomIDs = []
        lstOtherAtoms = list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == 0)[0])
        lstPTMAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == self._LatticeStructure)[0])
        lstNonPTMAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') != self._LatticeStructure)[0])
        self.__PTMAtomIDs = list(self.GetAtomData()[lstPTMAtoms,0].astype('int'))
        self.__NonPTMAtomIDs = list(self.GetAtomData()[lstNonPTMAtoms,0].astype('int'))
        self.__OtherAtomIDs = list(self.GetAtomData()[lstOtherAtoms,0].astype('int'))
        self.FindDefectiveAtoms(fltTolerance)
        self.__LatticeAtomIDs = list(set(self.__NonDefectiveAtomIDs) & set(self.__PTMAtomIDs))
        setAllLatticeAtomIDs = set(list(self.GetAtomData()[:,0]))
        self.__NonLatticeAtomIDs = list(setAllLatticeAtomIDs.difference(self.__LatticeAtomIDs))
    def GetLatticeAtomIDs(self):
        return self.__LatticeAtomIDs
    def GetNonLatticeAtomIDs(self):
        return self.__NonLatticeAtomIDs  
    def FindDefectiveAtoms(self, fltTolerance = None):
        if fltTolerance is None:
            fltStdLatticeValue = np.std(self.GetPTMAtoms()[:,self._intPE])
            fltTolerance = self.__fltGrainTolerance*fltStdLatticeValue #95% limit assuming Normal distribution
        fltMeanLatticeValue = np.mean(self.GetPTMAtoms()[:,self._intPE])
        lstNonDefectiveAtoms = np.where((self.GetColumnByIndex(self._intPE) < fltMeanLatticeValue +fltTolerance) & (self.GetColumnByIndex(self._intPE) > fltMeanLatticeValue - fltTolerance))[0]
        self.__NonDefectiveAtomIDs = list(self.GetAtomData()[lstNonDefectiveAtoms,0].astype('int'))
        setAllLatticeAtomIDs = set(list(self.GetAtomData()[:,0]))
        self.__DefectiveAtomIDs = setAllLatticeAtomIDs.difference(self.__NonDefectiveAtomIDs)
    def GetOtherAtomIDs(self):
        return self.__OtherAtomIDs
    def GetPTMAtomIDs(self):
        return self.__PTMAtomIDs
    def GetNonPTMAtomIDs(self):
        return self.__NonPTMAtomIDs
    def GetDefectiveAtomIDs(self):
        return self.__DefectiveAtomIDs
    def GetNonDefectiveAtomIDs(self):
        return self.__NonDefectiveAtomIDs
    def GetNonDefectiveAtoms(self):
        if len(self.__NonDefectiveAtomIDs) ==0:
            self.FindDefectiveAtoms()
        return self.GetAtomsByID(self.__NonDefectiveAtomIDs)
    def GetDefectiveAtoms(self):
        if len(self.__DefectiveAtomIDs) ==0:
            self.FindDefectiveAtoms()
        return self.GetAtomsByID(self.__DefectiveAtomIDs)
    def GetNonPTMAtoms(self):
        return self.GetAtomsByID(self.__NonPTMAtomIDs)
    def GetPTMAtoms(self):
        return self.GetAtomsByID(self.__PTMAtomIDs)  
    def GetOtherAtoms(self):
        return self.GetAtomsByID(self.__OtherAtomIDs)
    def GetNumberOfNonPTMAtoms(self):
        return len(self.__NonPTMAtomIDs)
    def GetNumberOfOtherAtoms(self)->int:
        return len(self.GetRows(self.__OtherAtomIDs))
    def GetNumberOfPTMAtoms(self)->int:
        return len(self.__PTMAtomIDs)
    def PlotGrainAtoms(self, strGrainNumber: str):
        return self.__PlotList(self.__PTMAtomIDs)
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
        self.__GrainBoundaryIDs = [] 
        self.__blnPEAssigned = False
        self.__blnVolumeAssigned = False
        self.__blnAdjustedMeshPointsAssigned = False
    def SetLatticeParameter(self, fltParameter: float):
        self.__LatticeParameter = fltParameter
    def LabelAtomsByGrain(self, fltRadius = None):
        if fltRadius is None:
            fltRadius = 3*self.__LatticeParameter
        objQuantisedCuboidPoints = QuantisedCuboidPoints(self.GetAtomsByID(self.GetNonLatticeAtomIDs())[:,1:4],self.GetUnitBasisConversions(),self.GetCellVectors(),self.__LatticeParameter*np.ones(3),10)
        lstGrainAtoms = self.GetLatticeAtomIDs()
        lstNonGrainAtoms = self.GetNonLatticeAtomIDs()
        lstGrainNumbers = objQuantisedCuboidPoints.ReturnGrains(self.GetAtomsByID(lstGrainAtoms)[:,1:4])
        self.AppendGrainNumbers(lstGrainNumbers, lstGrainAtoms)
        objQuantisedCuboidPoints.FindJunctionLines()
        self.__JunctionLineIDs = objQuantisedCuboidPoints.GetJunctionLineIDs()
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i] = gl.GeneralJunctionLine(objQuantisedCuboidPoints.GetJunctionLinePoints(i),i)
            self.__JunctionLines[i].SetWrappedMeshPoints(self.MoveToSimulationCell(objQuantisedCuboidPoints.GetJunctionLinePoints(i)))
            self.__JunctionLines[i].SetAdjacentGrains(objQuantisedCuboidPoints.GetAdjacentGrains(i, 'JunctionLine'))
            self.__JunctionLines[i].SetAdjacentGrainBoundaries(objQuantisedCuboidPoints.GetAdjacentGrainBoundaries(i))
            self.__JunctionLines[i].SetPeriodicDirections(objQuantisedCuboidPoints.GetPeriodicExtensions(i,'JunctionLine'))
        self.__GrainBoundaryIDs = objQuantisedCuboidPoints.GetGrainBoundaryIDs()
        for k in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[k] = gl.GeneralGrainBoundary(objQuantisedCuboidPoints.GetGrainBoundaryPoints(k),k)
            self.__GrainBoundaries[k].SetWrappedMeshPoints(self.MoveToSimulationCell(objQuantisedCuboidPoints.GetGrainBoundaryPoints(k)))
            self.__GrainBoundaries[k].SetAdjacentGrains(objQuantisedCuboidPoints.GetAdjacentGrains(k, 'GrainBoundary'))
            self.__GrainBoundaries[k].SetAdjacentJunctionLines(objQuantisedCuboidPoints.GetAdjacentJunctionLines(k))
            self.__GrainBoundaries[k].SetPeriodicDirections(objQuantisedCuboidPoints.GetPeriodicExtensions(k,'GrainBoundary'))
            for l in self.__GrainBoundaries[k].GetWrappedMeshPoints():
                lstSurroundingAtoms = list(self.FindSphericalAtoms(self.GetAtomsByID(lstNonGrainAtoms)[:,0:4],l, fltRadius))
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
            self.__JunctionLines[i].SetTotalPE(np.sum(self.GetColumnByIDs(self.__JunctionLines[i].GetAtomIDs(),self._intPE)))
        for j in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[j].SetTotalPE(np.sum(self.GetColumnByIDs(self.__GrainBoundaries[j].GetAtomIDs(),self._intPE)))
        self.__blnPEAssigned = True
    def AssignVolumes(self):
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i].SetVolume(np.sum(self.GetColumnByIDs(self.__JunctionLines[i].GetAtomIDs(),self._intVolume)))
        for j in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[j].SetVolume(np.sum(self.GetColumnByIDs(self.__GrainBoundaries[j].GetAtomIDs(),self._intVolume)))
        self.__blnVolumeAssigned = True
    def AssignAdjustedMeshPoints(self):
        for i in self.__JunctionLineIDs:
            arrMeshPoints = self.__JunctionLines[i].GetMeshPoints()
            arrAdjustedMeshPoints = np.array(list(map(lambda x: np.mean(self.GetAtomsByID(self.FindSphericalAtoms(self.GetAtomsByID(self.GetNonLatticeAtomIDs())[:,0:4],x,3*self.__LatticeParameter, True))[:,1:4],axis=0), arrMeshPoints)))
            self.__JunctionLines[i].SetAdjustedMeshPoints(arrAdjustedMeshPoints)
        for j in self.__GrainBoundaryIDs:
            arrMeshPoints = self.__GrainBoundaries[j].GetMeshPoints()
            arrAdjustedMeshPoints = np.array(list(map(lambda x: np.mean(self.GetAtomsByID(self.FindSphericalAtoms(self.GetAtomsByID(self.GetNonLatticeAtomIDs())[:,0:4],x,3*self.__LatticeParameter, True))[:,1:4],axis=0), arrMeshPoints)))
            self.__GrainBoundaries[j].SetAdjustedMeshPoints(arrAdjustedMeshPoints)
        self.__blnAdjustedMeshPointsAssigned = True
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
        lstNextAtoms = list(set(self.GetLatticeAtomIDs()).intersection(set(self.GetGrainAtomIDs(0))))
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
            lstNextAtoms = list(set(self.GetLatticeAtomIDs()).intersection(set(self.GetGrainAtomIDs(0))))
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
                for k in self.__JunctionLines[i].GetAdjacentGrains():
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
            fdata.write('Time Step \n')
            fdata.write('{} \n'.format(self.GetTimeStep()))
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
                if self.__blnVolumeAssigned: #optional data writing should be put here with a boolean flag
                    fdata.write('Volume \n')
                    fdata.write('{} \n'.format(self.__JunctionLines[i].GetVolume()))
                if self.__blnPEAssigned:
                    fdata.write('PE \n')
                    fdata.write('{} \n'.format(self.__JunctionLines[i].GetTotalPE()))
                if self.__blnAdjustedMeshPointsAssigned:
                    fdata.write('Adjusted Mesh Points \n')
                    fdata.write('{} \n'.format(self.__JunctionLines[i].GetAdjustedMeshPoints().tolist()))
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
                if self.__blnVolumeAssigned: #optional data writing should be put here with a boolean flag
                    fdata.write('Volume \n')
                    fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetVolume()))
                if self.__blnPEAssigned:
                    fdata.write('PE \n')
                    fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetTotalPE()))
                if self.__blnAdjustedMeshPointsAssigned:
                    fdata.write('Adjusted Mesh Points \n')
                    fdata.write('{} \n'.format(self.__GrainBoundaries[k].GetAdjustedMeshPoints().tolist()))
            fdata.write('Grain Numbers \n')
            fdata.write('{}'.format(self.GetColumnByIndex(self.__intGrainNumber).astype('int').tolist()))
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
                        line = next(fdata).strip()
                        if line == "PE":
                            line = next(fdata).strip()
                            objJunctionLine.SetTotalPE(eval(line))
                        if line == "Adjusted Mesh Points":
                            line = next(fdata).strip()
                            objJunctionLine.SetAdjustedMeshPoints(eval(line))
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
                        line = next(fdata).strip()
                        if line == "PE":
                            line = next(fdata).strip()
                            objGrainBoundary.SetTotalPE(eval(line))
                        if line == "Adjusted Mesh Points":
                            line = next(fdata).strip()
                            objGrainBoundary.SetAdjustedMeshPoints(eval(line))
                        self.__GrainBoundaries[intGB] = objGrainBoundary
                    elif line == "Grain Numbers": 
                        line = next(fdata).strip()
                        self.AppendGrainNumbers(eval(line))
            self.MakeGrainTrees()
            self.__JunctionLineIDs = list(self.__JunctionLines.keys())
            self.__GrainBoundaryIDs = list(self.__GrainBoundaries.keys())

class LAMMPSSummary(object):
    def __init__(self):
        self.__dctDefects = dict()
        self.__GlobalJunctionLines = dict()
        self.__GlobalGrainBoundaries = dict()
        self.__CellVectors = gf.StandardBasisVectors(3)
        self.__BasisConversion = gf.StandardBasisVectors(3)
        self.__BoundaryTypes = ['pp','pp','pp']
    def GetDefectObject(self, fltTimeStep):
        return self.__dctDefects[fltTimeStep]
    def GetTimeSteps(self):
        return sorted(list(self.__dctDefects.keys()))
    def GetTimeStepPosition(self, fltTimeStep)->int:
        return list(self.__dctDefects.keys()).index(fltTimeStep)
    def ReadInData(self, strFilename: str, blnCorrelateDefects = False):
        with open(strFilename) as fdata:
            while True:
                try:
                    line = next(fdata).strip()
                except StopIteration as EndOfFile:
                    break
                if line == "Time Step":
                    intTimeStep = int(next(fdata).strip())
                    objDefect = gl.DefectObject(intTimeStep)
                    line = next(fdata).strip()
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
                    line = next(fdata).strip()
                    if line == "PE":
                        line = next(fdata).strip()
                        objJunctionLine.SetTotalPE(eval(line))
                    if line == "Adjusted Mesh Points":
                        line = next(fdata).strip()
                        objJunctionLine.SetAdjustedMeshPoints(eval(line))
                    objDefect.AddJunctionLine(objJunctionLine)
                if line == "Grain Boundary":
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
                    line = next(fdata).strip()
                    if line == "PE":
                        line = next(fdata).strip()
                        objGrainBoundary.SetTotalPE(eval(line))
                    if line == "Adjusted Mesh Points":
                        line = next(fdata).strip()
                        objGrainBoundary.SetAdjustedMeshPoints(eval(line))
                    objDefect.AddGrainBoundary(objGrainBoundary)
        if len(list(self.__dctDefects.keys())) == 0 and blnCorrelateDefects: 
            for i in objDefect.GetGrainBoundaryIDs():
                objDefect.GetGrainBoundary(i).SetGlobalID(i)
                objDefect.AddGlobalGrainBoundary(objDefect.GetGrainBoundary(i))
            for j in objDefect.GetJunctionLineIDs():
                objDefect.GetJunctionLine(j).SetGlobalID(j)
                objDefect.AddGlobalJunctionLine(objDefect.GetJunctionLine(j))    
        elif blnCorrelateDefects:
            intLastTimeStep = self.GetTimeSteps()[-1]
            lstPreviousGlobalGBIDs = self.__dctDefects[intLastTimeStep].GetGlobalGrainBoundaryIDs()
            lstCurrentGBIDs = objDefect.GetGrainBoundaryIDs()
            if len(lstPreviousGlobalGBIDs) != len(lstCurrentGBIDs):
                warnings.warn('Number of grain boundaries changed from ' + str(len(lstPreviousGlobalGBIDs)) + ' to ' + str(len(lstCurrentGBIDs)) + ' at time step ' + str(intTimeStep))
            lstGBIDs = lstPreviousGlobalGBIDs
            lstMapToGlobalGB = []
            while len(lstCurrentGBIDs) > 0:
                intGBID = lstCurrentGBIDs.pop(0)
                intPreviousID = self.CorrelateMeshPoints(objDefect.GetGrainBoundary(intGBID).GetMeshPoints(), lstGBIDs, 'Grain Boundary')
                lstMapToGlobalGB.append(intPreviousID)
                objDefect.GetGrainBoundary(intGBID).SetGlobalID(intPreviousID)
                objDefect.AddGlobalGrainBoundary(objDefect.GetGrainBoundary(intGBID))
                lstGBIDs.remove(intPreviousID)
            lstPreviousGlobalJLIDs = self.__dctDefects[intLastTimeStep].GetGlobalJunctionLineIDs()
            lstCurrentJLIDs = objDefect.GetJunctionLineIDs()
            if len(lstPreviousGlobalJLIDs) != len(lstCurrentJLIDs):
                warnings.warn('Number of junction lines changed from ' + str(len(lstPreviousGlobalJLIDs)) + ' to ' + str(len(lstCurrentJLIDs)) + ' at time step ' + str(intTimeStep))
            lstJLIDs = np.copy(lstPreviousGlobalJLIDs).tolist()
            while len(lstCurrentJLIDs) > 0:
                intJLID = lstCurrentJLIDs.pop(0)
                intPreviousID = self.CorrelateMeshPoints(objDefect.GetJunctionLine(intJLID).GetMeshPoints(), lstJLIDs, 'Junction Line')
                objDefect.GetJunctionLine(intJLID).SetGlobalID(intPreviousID)
                objDefect.AddGlobalJunctionLine(objDefect.GetJunctionLine(intJLID))
                lstJLIDs.remove(intPreviousID)
        self.__dctDefects[intTimeStep] = objDefect
    def SetCellVectors(self, inCellVectors: np.array):
        self.__CellVectors = inCellVectors
    def SetBasisConversion(self,inBasisConversion: np.array):
        self.__BasisConversion = inBasisConversion
    def SetBoundaryTypes(self, inList):
        self.__BoundaryTypes
    def ConvertPeriodicDirections(self, inPeriodicDirections)->list:
        lstPeriodicity = ['f','f','f'] #assume fixed boundary types
        for j in inPeriodicDirections:
            lstPeriodicity[j] = 'pp'
        return lstPeriodicity
    def CorrelateMeshPoints(self,arrMeshPoints: np.array, lstPreviousGlobalIDs: list, strType: str)->int: #checks to see which previous set of mesh points lie closest to the current mesh point
        lstDistances = []
        arrMean = np.mean(arrMeshPoints, axis= 0) 
        for j in lstPreviousGlobalIDs:
            if strType == 'Grain Boundary':
                arrPreviousMeshPoints =  self.__dctDefects[self.GetTimeSteps()[-1]].GetGlobalGrainBoundary(j).GetMeshPoints()
            elif strType == 'Junction Line':
                arrPreviousMeshPoints =  self.__dctDefects[self.GetTimeSteps()[-1]].GetGlobalJunctionLine(j).GetMeshPoints()
            arrPreviousMean = np.mean(arrPreviousMeshPoints, axis= 0)
            arrPeriodicShift = gf.PeriodicEquivalentMovement(arrMean, arrPreviousMean, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)[2]
            arrPreviousMeshPoints = arrPreviousMeshPoints - arrPeriodicShift
            flt1 = spatial.distance.directed_hausdorff(arrMeshPoints,arrPreviousMeshPoints)[0]
            flt2 = spatial.distance.directed_hausdorff(arrPreviousMeshPoints,arrMeshPoints)[0]
            lstDistances.append(max(flt1,flt2))
        fltMin = min(lstDistances)
        if fltMin > 8.1:
            warnings.warn('Possible error as the time correlated Hausdorff distance is ' + str(min(lstDistances)) + ' at time step ' + str(self.GetTimeSteps()[-1]) + ' for ' + str(strType) + ' ' + str(j))
        return lstPreviousGlobalIDs[np.argmin(lstDistances)]   

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
        self.__Grains = np.copy(arrValues)
        self.__JunctionLines = []
        nx,ny,nz = np.shape(arrValues)
        self.__ModArray = arrModArray
        self.__Iterations = 0
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
        intConnections = 0
        intIterations = 0
        arrTotal = np.zeros(self.__ModArray)
        while intConnections != 1:
            arrOut = ndimage.filters.gaussian_filter(arrOut, 1, mode = 'wrap')
            arrOut = (arrOut > np.mean(arrOut))
            arrOut = arrOut.astype('bool').astype('int') # convert to binary
            arrOut, intConnections = ndimage.measurements.label(arrOut, np.ones([3,3,3]))
            arrOut = arrOut.astype('bool').astype('float')
            intIterations += 1
        if intIterations > 1:
            warnings.warn('A gaussian filter has been applied ' +  str(intIterations) + ' time(s) to form a connected defective array.')
        self.__Iterations = intIterations
        arrPoints = np.argwhere(arrOut == 0)
        for j in arrPoints:
            n=1
            if self.BoxIsNotWrapped(j, n):
                arrBox = arrOut[j[0]-n:j[0]+n+1,j[1]-n:j[1]+n+1, j[2]-n:j[2]+n+1]
            else:
                arrBox = arrOut[gf.WrapAroundSlice(np.array([[j[0]-n,j[0]+n+1],[j[1]-n,j[1]+n+1], [j[2]-n,j[2]+n+1]]),self.__ModArray)]
                arrBox = np.reshape(arrBox,(2*n+1,2*n+1,2*n+1))
            arrNeighbours = np.argwhere(arrBox != 0) - np.ones(3)
            if len(arrNeighbours) > 0:
                arrSums = np.concatenate(list(map(lambda x: x + arrNeighbours, arrNeighbours)))
                arrSums = arrSums[np.all(arrSums == 0, axis=1)]
                if len(arrSums) >0:
                    arrOut[j[0],j[1],j[2]] = 1
        self.__BinaryArray = arrOut.astype('bool').astype('int')
        self.__InvertBinary = np.invert(self.__BinaryArray.astype('bool')).astype('int')
        self.__Grains, intGrainLabels  = ndimage.measurements.label(self.__InvertBinary, np.ones([3,3,3]))
        if intGrainLabels <= 1:
            warnings.warn('Only ' + str(intGrainLabels) + ' grain(s) detected')
        self.__Grains = np.asarray(self.__Grains)
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
    def GetDefectPositions(self):
        return self.__DefectPositions
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
        if len(np.argwhere(self.__ExpandedGrains == 0)) > 0:
            warnings.warn('Expanded grain method has not removed all grain boundary atoms')
        if blnExpanded:
            return list(self.__ExpandedGrains[inPoints[:,0],inPoints[:,1],inPoints[:,2]])
        else:
            arrZeros = np.argwhere(self.__BinaryArray == 0)
            self.__ReturnGrains = np.copy(self.__ExpandedGrains)
            for j in arrZeros:
                self.__ReturnGrains[j[0],j[1],j[2]] = 0
            return list(self.__ReturnGrains[inPoints[:,0],inPoints[:,1],inPoints[:,2]])                     
    def FindJunctionLines(self):
        arrGrainBoundaries = np.zeros(self.__ModArray) #temporary array 
        self.__GrainBoundariesArray = np.zeros(self.__ModArray)
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
                arrGrainBoundaries[j[0],j[1],j[2]] = 1 + lstGrainBoundaryList.index(lstValues)
        self.__JunctionLinesArray, intJLs = ndimage.measurements.label(self.__JunctionLinesArray, np.ones([3,3,3]))
        arrJLPoints = np.argwhere(self.__JunctionLinesArray > 0)
        self.CheckPeriodicity(arrJLPoints, 'JunctionLine')
        lstValues = list(np.unique(self.__JunctionLinesArray)) #renumber the array sequentially for convenience starting at 1
        if 0 in lstValues:
            lstValues.remove(0)
        for intIndex, intValue in enumerate(lstValues):
            self.__JunctionLinesArray[self.__JunctionLinesArray == intValue] = intIndex+1 
        lstValues = list(np.unique(self.__JunctionLinesArray))
        if 0 in lstValues:
            lstValues.remove(0)
        self.__JunctionLineIDs = lstValues
        self.FindPeriodicGrainBoundaries(arrGrainBoundaries)
        self.__GrainBoundariesArray = self.__GrainBoundariesArray.astype('int')
        lstValues = list(np.unique(self.__GrainBoundariesArray)) #renumber the array sequentially for convenience starting at 1
        if 0 in lstValues:
            lstValues.remove(0)
        for intIndex, intValue in enumerate(lstValues):
            self.__GrainBoundariesArray[self.__GrainBoundariesArray == intValue] = intIndex+1 
        lstValues = list(np.unique(self.__GrainBoundariesArray))
        if 0 in lstValues:
            lstValues.remove(0)
        self.__GrainBoundaryIDs = lstValues
    def FindPeriodicGrainBoundaries(self,inGBArray: np.array):
        intMaxGBNumber = 0
        lstValues = list(np.unique(inGBArray))
        if 0 in lstValues:
            lstValues.remove(0)
        for j in lstValues:
            arrPoints = np.argwhere(inGBArray == j)
            arrReturn, intGBs = ndimage.measurements.label(inGBArray == j, np.ones([3,3,3]))
            arrReturn[arrReturn > 0 ] += intMaxGBNumber
            self.__GrainBoundariesArray += arrReturn
            self.CheckPeriodicity(arrPoints, 'GrainBoundary')
            intMaxGBNumber += intGBs
    def CheckPeriodicity(self, inPoints: np.array, strType: str, inArray = None):
        if strType == 'JunctionLine':
            arrCurrent = self.__JunctionLinesArray
        elif strType == 'GrainBoundary':
            arrCurrent = self.__GrainBoundariesArray
        elif strType == 'PassArray':
            arrCurrent = inArray
        for j in range(3):
            lstIndicesLower = np.where(inPoints[:,j] == 0)[0]
            arrPointsLower = inPoints[lstIndicesLower]
            lstIndicesUpper = np.where(inPoints[:,j] == self.__ModArray[j]-1)[0]
            arrPointsUpper  = inPoints[lstIndicesUpper]
            arrPointsUpper[:,j] = arrPointsUpper[:,j] - np.ones(len(arrPointsUpper))*self.__ModArray[j]
            arrDistanceMatrix = spatial.distance_matrix(arrPointsLower, arrPointsUpper)
            arrClosePoints = np.argwhere(arrDistanceMatrix < 2) #allows 3d diagonal connectivity
            for j in arrClosePoints:
                tupPairs = (arrCurrent[tuple(zip(arrPointsLower[j[0]]))][0], arrCurrent[tuple(zip(arrPointsUpper[j[1]]))][0])
                if tupPairs[0] != tupPairs[1]:
                    arrCurrent[arrCurrent == max(tupPairs)] = min(tupPairs)
    def BoxIsNotWrapped(self, inPoint: np.array, n: int)->bool:
        if np.all(np.mod(inPoint -n*np.ones(3), self.__ModArray) == inPoint-n*np.ones(3)) and np.all(np.mod(inPoint +(n+1)*np.ones(3), self.__ModArray) == inPoint+(n+1)*np.ones(3)):
            return True
        else:
            return False
    def ExpandGrains(self, n=3):
        n += 2*self.__Iterations 
        self.__ExpandedGrains = np.copy(self.__Grains)
        arrGBPoints = np.argwhere(self.__Grains == 0)
        arrGBPoints = arrGBPoints.astype('int')
        arrIndices = gf.CreateCuboidPoints(np.array([[-n,n],[-n,n],[-n,n]]))
        arrIndices = np.matmul(np.matmul(arrIndices, self.__InverseBasisConversion),self.__InverseScaling)
        arrDistances = np.reshape(np.array(list(map(np.linalg.norm, arrIndices))),(2*n+1,2*n+1,2*n+1))
        if len(arrGBPoints) > 0 :
            for j in arrGBPoints:
                #if np.all(np.mod(j -n*np.ones(3), self.__ModArray) == j-n*np.ones(3)) and np.all(np.mod(j +(n+1)*np.ones(3), self.__ModArray) == j+(n+1)*np.ones(3)):
                if self.BoxIsNotWrapped(j, n):
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
        n = 3
        counter = 0
        while (counter < len(arrGBPoints)): 
            l = arrGBPoints[counter]
            if self.BoxIsNotWrapped(l,n):
                arrBox = self.__ExpandedGrains[l[0]-n:l[0]+n+1,l[1]-n:l[1]+n+1, l[2]-n:l[2]+n+1]
            else:
                arrBox = self.__ExpandedGrains[gf.WrapAroundSlice(np.array([[l[0]-n,l[0]+n+1],[l[1]-n,l[1]+n+1], [l[2]-n,l[2]+n+1]]),self.__ModArray)]
            arrBox = arrBox[arrBox != 0]
            if len(arrBox) > 0:
                arrValues, arrCounts = np.unique(arrBox, return_counts = True)
                intGrain = arrValues[np.argmax(arrCounts)]
                if len(np.argwhere(arrValues == intGrain)) ==1 or n >  np.round(min(self.__ModArray)/2,0).astype('int'):
                    self.__ExpandedGrains[l[0],l[1],l[2]] = intGrain
                    counter += 1
                else:
                    n += 1
        intNumberOfDefects = len(np.argwhere(self.__ExpandedGrains == 0))
        if intNumberOfDefects > 0:
            warnings.warn('Error expanding grains. The ExpandedGrains array still has ' + str(intNumberOfDefects) + ' defect(s)')
    def GetExpandedGrains(self):
        return self.__ExpandedGrains
    def GetGrainBoundaryPoints(self, intGrainBoundaryID = None):
        if intGrainBoundaryID is None:
            return np.matmul(np.matmul(self.MergeMeshPoints(np.argwhere(self.__GrainBoundariesArray.astype('int') != 0)) +np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion) 
        elif intGrainBoundaryID in self.__GrainBoundaryIDs: 
            return np.matmul(np.matmul(self.MergeMeshPoints(np.argwhere(self.__GrainBoundariesArray.astype('int') == intGrainBoundaryID))+np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion)
        else:
            warnings.warn(str(intGrainBoundaryID) + ' is an invalid grain boundary ID')
    def MergeMeshPoints(self, inGridPoints: np.array):#This merges grain boundaries or junction lines so they form one group of points when they were
        lstPoints = []   #previously split over the simulation cell boundaries
        arrDistanceMatrix = pairwise_distances(inGridPoints)
        clustering = DBSCAN(eps=2, metric = 'precomputed', min_samples = 1).fit(arrDistanceMatrix)
        arrLabels = clustering.labels_
        intLabels = len(np.unique(arrLabels))
        if intLabels > 1: 
            arrConnected = np.ones(3) #assumes the defect is connected in all three directions
            lstPoints = []
            arrTotal = np.copy(inGridPoints)
            for j in range(3):
                arrTranslation = np.zeros(3)
                if list(np.unique(inGridPoints[:,j])) != list(range(self.__ModArray[j])): #check the points don't span the entire cell
                    arrConnected[j] = 0 #sets this direction as not connected
                    arrTranslation[j] = self.__ModArray[j]
                    lstPoints.append(arrTotal)
                    lstPoints.append(arrTotal + arrTranslation)
                    lstPoints.append(arrTotal - arrTranslation)
                    arrTotal = np.concatenate(lstPoints)
                    lstPoints = []
            arrTotal = np.unique(arrTotal,axis = 0)
            for k in range(3): #only include a wrapper of length half the cell size extended in each co-ordinate direction
                if arrConnected[k] == 0:
                    arrNearPoints = np.where((arrTotal[:,k] >= -self.__ModArray[k]/2) & (arrTotal[:,k] < 3*self.__ModArray[k]/2))[0]
                    arrTotal = arrTotal[arrNearPoints]  
            arrDistanceMatrix = arrDistanceMatrix = pairwise_distances(arrTotal)    
            clustering = DBSCAN(eps=2, metric = 'precomputed',min_samples = 1).fit(arrDistanceMatrix)
            arrLabels = clustering.labels_
            arrUniqueValues, arrCounts = np.unique(arrLabels, return_counts = True)
            intMaxValue = max(arrCounts)
            arrMaxLabels = arrUniqueValues[arrCounts == intMaxValue]
            lstDistances = []
            for l in arrMaxLabels:
                arrPoints = arrTotal[arrLabels == l]
                lstDistances.append(np.linalg.norm(np.mean(arrPoints,axis=0) - 0.5*self.__ModArray))
            return arrTotal[arrLabels == arrMaxLabels[np.argmin(lstDistances)]]
        else:
            return inGridPoints
    def GetJunctionLinePoints(self, intJunctionLineID = None)->np.array:
        if intJunctionLineID is None:
            return np.matmul(np.matmul(self.MergeMeshPoints(np.argwhere(self.__JunctionLinesArray.astype('int') != 0)) +np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion) 
        elif intJunctionLineID in self.__JunctionLineIDs: 
            return np.matmul(np.matmul(self.MergeMeshPoints(np.argwhere(self.__JunctionLinesArray.astype('int') == intJunctionLineID))+np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion)
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
