import re
import numpy as np
import GeometryFunctions as gf
import GeneralLattice as gl
import LatticeDefinitions as ld
from scipy import spatial, optimize, ndimage, stats 
from skimage.morphology import skeletonize, thin, medial_axis, remove_small_holes, remove_small_objects, skeletonize_3d, binary_dilation
from scipy.cluster.vq import kmeans,vq
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from skimage.filters import gaussian, threshold_otsu
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
                lstColumnTypes = []
                for j in line:
                    if "." in j:
                        lstColumnTypes.append('%s')
                    else:
                        lstColumnTypes.append('%i')
                objTimeStep.SetColumnTypes(lstColumnTypes) 
                objTimeStep.CategoriseAtoms()
                self.__dctTimeSteps[str(timestep)] = objTimeStep            
            Dfile.close()
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
    def SetColumnTypes(self, lstColumnTypes): #records whether columns are integers or floats
        self.__ColumnTypes = lstColumnTypes
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
    def AddColumn(self, arrColumn: np.array, strColumnName: str, strFormat = '%s'):
        self.__AtomData = np.append(self.__AtomData, arrColumn, axis=1)
        self.__ColumnNames.append(strColumnName)
        self.__ColumnTypes += ' ' + strFormat
    def SetColumnToZero(self, strColumnName: str):
        arrColumn = np.zeros(self.GetNumberOfAtoms)
        intColumnIndex = self.GetColumnIndex(strColumnName)
        self.__AtomData[:,intColumnIndex] = arrColumn
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
    def GetColumnIndex(self, strColumnName):
        return self.__ColumnNames.index(strColumnName) 
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
    def PeriodicEquivalents(self, inPositionVector: np.array, blnInsideCell = True)->np.array: #Moved to GeometryFunctions.py for access from other   
        return gf.PeriodicEquivalents(inPositionVector,  self.__CellVectors,self.__BasisConversion, self.__BoundaryTypes, blnInsideCell)
    def MoveToSimulationCell(self, inPositionVector: np.array)->np.array:
        return gf.WrapVectorIntoSimulationCell(self.__CellBasis, self.__BasisConversion, inPositionVector)
    def PeriodicShiftAllCloser(self, inFixedPoint: np.array, inAllPointsToShift: np.array)->np.array:
        return gf.PeriodicShiftAllCloser(inFixedPoint,inAllPointsToShift, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
    def PeriodicShiftCloser(self, inFixedPoint: np.array, inPointToShift: np.array)->np.array:
        return gf.PeriodicShiftCloser(inFixedPoint, inPointToShift, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
    def MakePeriodicDistanceMatrix(self, inVectors1: np.array, inVectors2: np.array)->np.array:
        return gf.MakePeriodicDistanceMatrix(inVectors1, inVectors2, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        return gf.PeriodicMinimumDistance(inVector1, inVector2, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)
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
    def WriteDumpFile(self, strFilename: str):
        strHeader = 'ITEM: TIMESTEP \n'
        strHeader += str(self.GetTimeStep()) + '\n'
        strHeader += 'ITEM: NUMBER OF ATOMS \n'
        strHeader += str(self.GetNumberOfAtoms()) + '\n'
        strHeader += 'ITEM: BOX BOUNDS xy xz yz ' + ' '.join(self.__BoundaryTypes) + '\n'
        for j in range(3):
            strHeader += str(self.__BoundBoxDimensions[j,0]) + ' ' + str(self.__BoundBoxDimensions[j,1]) + ' '  + str(self.__BoundBoxDimensions[j,2]) + '\n'
        strHeader += 'ITEM: ATOMS ' + ' '.join(self.__ColumnNames)
        np.savetxt(strFilename, self.GetAtomData(), fmt= ' '.join(self.__ColumnTypes), header=strHeader, comments='')

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
        self.__DefectiveAtomIDs = []
        self.__NonDefectiveAtomIDs = []
        self.FindPlaneNormalVectors()
        self.__dctLatticeTypes = dict()
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
        if 'StructureType' in self.GetColumnNames():
            self.__DefectiveAtomIDs = []
            self.__NonDefectiveAtomIDs = []
            lstOtherAtoms = list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == 0)[0])
            lstPTMAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == self._LatticeStructure)[0])
            lstNonPTMAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') != self._LatticeStructure)[0])
            self.__PTMAtomIDs = list(self.GetAtomData()[lstPTMAtoms,0].astype('int'))
            self.__NonPTMAtomIDs = list(self.GetAtomData()[lstNonPTMAtoms,0].astype('int'))
            self.__OtherAtomIDs = list(self.GetAtomData()[lstOtherAtoms,0].astype('int'))
            self.FindDefectiveAtoms(fltTolerance)
    def GetLatticeAtomIDs(self):
        return self.__LatticeAtomIDs
    def GetNonLatticeAtomIDs(self):
        return self.__NonLatticeAtomIDs  
    def FindDefectiveAtoms(self, fltTolerance = None):
        if fltTolerance is None:
            #fltStdLatticeValue = np.std(self.GetPTMAtoms()[:,self._intPE])
            #fltTolerance = self.__fltGrainTolerance*fltStdLatticeValue
            arrPTM = stats.gamma.fit(self.GetPTMAtoms()[:,self._intPE])
            fltMedian = 0
            fltMedianPTM = stats.gamma(*arrPTM).median()
            arrNonPTM = stats.gamma.fit(self.GetNonPTMAtoms()[:,self._intPE])
            fltMedianNonPTM = stats.gamma(*arrNonPTM).median()
            fltThreshold = np.mean([fltMedianPTM,fltMedianNonPTM])
        else:
            fltThreshold = fltTolerance
        lstRowDefectiveAtoms = np.where(self.GetPTMAtoms()[:,self._intPE] > fltThreshold)[0]
        lstDefectivePTMIDs = list(self.GetAtomData()[lstRowDefectiveAtoms,0])
        self.__NonLatticeAtomIDs = list(set(lstDefectivePTMIDs) | set(self.__NonPTMAtomIDs))
        setAllLatticeAtomIDs = set(list(self.GetAtomData()[:,0]))
        self.__LatticeAtomIDs = list(setAllLatticeAtomIDs.difference(self.__NonLatticeAtomIDs))
        self.__DefectiveAtomIDs =lstDefectivePTMIDs
        #lstNonDefectiveAtoms = np.where((self.GetColumnByIndex(self._intPE) < fltMeanLatticeValue +fltTolerance) & (self.GetColumnByIndex(self._intPE) > fltMeanLatticeValue - fltTolerance))[0]
        # self.__DefectiveAtomIDs = lstDefectiveAtomIDs
        # setAllLatticeAtomIDs = set(list(self.GetAtomData()[:,0]))
        # self.__DefectiveAtomIDs = setAllLatticeAtomIDs.difference(self.__NonDefectiveAtomIDs)
        # self.__LatticeAtomIDs = list(set(self.__NonDefectiveAtomIDs) & set(self.__PTMAtomIDs))
        # setAllLatticeAtomIDs = set(list(self.GetAtomData()[:,0]))
        # self.__NonLatticeAtomIDs = list(setAllLatticeAtomIDs.difference(self.__LatticeAtomIDs))
    def GetLatticeAtoms(self):
        return self.GetAtomsByID(self.__LatticeAtomIDs)
    def GetNonLatticeAtoms(self):    
        return self.GetAtomsByID(self.__NonLatticeAtomIDs)
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
        self.blnPEAssigned = False
        self.blnVolumeAssigned = False
        self.blnAdjustedMeshPointsAssigned = False
        self.intGrainNumber = -1 #set to a dummy value to check 
        self.intGrainBoundary = -1
        self.__objRealCell = gl.RealCell(ld.GetCellNodes(str(intLatticeType)),fltLatticeParameter*np.ones(3))
    def GetUnassignedGrainAtomIDs(self):
        if 'GrainNumber' not in self.GetColumnNames():
            warnings.warn('Grain labels not set.')
        else: 
            lstUnassignedAtoms = np.where(self.GetColumnByName('GrainNumber') ==-1)[0]
        return list(self.GetAtomData()[lstUnassignedAtoms,0].astype('int'))
    def SetLatticeParameter(self, fltParameter: float):
        self.__LatticeParameter = fltParameter
    def LabelAtomsByGrain(self, fltTolerance = 3.14, fltRadius = None):
        if fltRadius is None:
            fltRadius = 3*self.__LatticeParameter
        objQuantisedCuboidPoints = QuantisedCuboidPoints(self.GetAtomsByID(self.GetNonLatticeAtomIDs())[:,1:4],self.GetUnitBasisConversions(),self.GetCellVectors(),self.__LatticeParameter*np.ones(3),10)
        self.FindDefectiveAtoms(fltTolerance)
        lstGrainAtoms = self.GetLatticeAtomIDs()
        lstNonGrainAtoms = self.GetNonLatticeAtomIDs()
        lstGrainNumbers = objQuantisedCuboidPoints.ReturnGrains(self.GetAtomsByID(lstGrainAtoms)[:,1:4],True)
        self.AppendGrainNumbers(lstGrainNumbers, lstGrainAtoms)
        objQuantisedCuboidPoints.FindJunctionLines()
        self.__JunctionLineIDs = objQuantisedCuboidPoints.GetJunctionLineIDs()
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i] = gl.GeneralJunctionLine(np.round(objQuantisedCuboidPoints.GetJunctionLinePoints(i),1),i)
            self.__JunctionLines[i].SetAdjacentGrains(objQuantisedCuboidPoints.GetAdjacentGrains(i, 'JunctionLine'))
            self.__JunctionLines[i].SetAdjacentGrainBoundaries(objQuantisedCuboidPoints.GetAdjacentGrainBoundaries(i))
            self.__JunctionLines[i].SetPeriodicDirections(objQuantisedCuboidPoints.GetPeriodicExtensions(i,'JunctionLine'))
        self.__GrainBoundaryIDs = objQuantisedCuboidPoints.GetGrainBoundaryIDs()
        for k in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[k] = gl.GeneralGrainBoundary(np.round(objQuantisedCuboidPoints.GetGrainBoundaryPoints(k),1),k)
            self.__GrainBoundaries[k].SetAdjacentGrains(objQuantisedCuboidPoints.GetAdjacentGrains(k, 'GrainBoundary'))
            self.__GrainBoundaries[k].SetAdjacentJunctionLines(objQuantisedCuboidPoints.GetAdjacentJunctionLines(k))
            self.__GrainBoundaries[k].SetPeriodicDirections(objQuantisedCuboidPoints.GetPeriodicExtensions(k,'GrainBoundary'))
            for l in self.MoveToSimulationCell(self.__GrainBoundaries[k].GetMeshPoints()):
                lstSurroundingAtoms = list(self.FindSphericalAtoms(self.GetAtomsByID(lstNonGrainAtoms)[:,0:4],l, fltRadius))
                self.__GrainBoundaries[k].AddAtomIDs(lstSurroundingAtoms)
        #self.RefineGrainLabels()
    def RefineGrainLabels(self): #try to assign any lattice atoms with -1 grain number to a grain.
        lstOfOldIDs = []
        self.MakeGrainTrees() #this will include defects with grain number 0 and 
        lstOfIDs = self.GetUnassignedGrainAtomIDs()
        intCounter = 0
        intGrainNumber = self.GetColumnNames().index('GrainNumber')
        while (lstOfIDs != lstOfOldIDs) and intCounter < 10:
            arrAtoms = self.GetAtomsByID(lstOfIDs)[:,0:4]
            lstGrainLabels = self.__GrainLabels
            if 0 in lstGrainLabels:
                lstGrainLabels.remove(0)
            if -1 in lstGrainLabels:
                lstGrainLabels.remove(-1)
            arrDistances = np.zeros([len(lstOfIDs), len(lstGrainLabels)])
            for intPosition,intLabel in enumerate(lstGrainLabels):
                arrDistances[:,intPosition] = self.__Grains[intLabel].query(arrAtoms[:,1:4],1,distance_upper_bound = self.__objRealCell.GetNearestNeighbourDistance())[0]
            for j in range(len(lstGrainLabels)):
                arrRows = np.where((arrDistances[:,j] == np.min(arrDistances, axis = 1)) & (np.isfinite(arrDistances[:,j])))[0]
                arrGrainNumbers = lstGrainLabels[j]*np.ones(len(arrRows)).astype('int')
                arrNewIDs = arrAtoms[arrRows, 0]
                self.SetColumnByIDs(arrNewIDs, intGrainNumber, arrGrainNumbers)
            self.MakeGrainTrees() #this will include defects with grain number 0 and 
            lstOfOldIDs = lstOfIDs
            lstOfIDs = self.GetUnassignedGrainAtomIDs()
            intCounter += 1
        if len(lstOfIDs) > 0:
            warnings.warn(str(len(lstOfIDs)) + ' grain atom(s) have been assigned a grain number of -1 after ' + str(intCounter) + ' iterations.')
    def GetGrainLabels(self):
        return self.__GrainLabels
    def AppendGrainNumbers(self, lstGrainNumbers: list, lstGrainAtoms = None):
        if 'GrainNumber' not in self.GetColumnNames():
            self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]), 'GrainNumber', '%i')
        arrGrainNumbers = np.array([lstGrainNumbers])
        self.__GrainLabels = list(np.unique(lstGrainNumbers))
        np.reshape(arrGrainNumbers, (len(lstGrainNumbers),1))
        intGrainNumber = self.GetColumnIndex('GrainNumber')
        if lstGrainAtoms is None:
            self.SetColumnByIndex(arrGrainNumbers, intGrainNumber)
        else:
            self.SetColumnByIDs(lstGrainAtoms, intGrainNumber, arrGrainNumbers)
    def AppendGrainBoundaries(self):
        if 'GrainBoundary' not in self.GetColumnNames():
            self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]), 'GrainBoundary', '%i')
        intGrainBoundary = self.GetColumnIndex('GrainBoundary')
        for i in self.__GrainBoundaryIDs:
            lstIDs = self.__GrainBoundaries[i].GetAtomIDs()
            intValue = self.__GrainBoundaries[i].GetID()
            arrValues = intValue*np.ones(len(lstIDs))
            np.reshape(arrValues,(len(lstIDs),1))
            self.SetColumnByIDs(lstIDs, intGrainBoundary, arrValues)
    def AppendJunctionLines(self):
        if 'JunctionLine' not in self.GetColumnNames():
            self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]), 'JunctionLine', '%i')
        intJunctionLine = self.GetColumnIndex('JunctionLine')
        for i in self.__JunctionLineIDs:
            lstIDs = self.__JunctionLines[i].GetAtomIDs()
            intValue = self.__JunctionLines[i].GetID()
            arrValues = intValue*np.ones(len(lstIDs))
            np.reshape(arrValues,(len(lstIDs),1))
            self.SetColumnByIDs(lstIDs, intJunctionLine, arrValues)
    def AssignPE(self):
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i].SetTotalPE(np.sum(self.GetColumnByIDs(self.__JunctionLines[i].GetAtomIDs(),self._intPE)))
        for j in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[j].SetTotalPE(np.sum(self.GetColumnByIDs(self.__GrainBoundaries[j].GetAtomIDs(),self._intPE)))
        self.blnPEAssigned = True
    def AssignVolumes(self):
        for i in self.__JunctionLineIDs:
            self.__JunctionLines[i].SetVolume(np.sum(self.GetColumnByIDs(self.__JunctionLines[i].GetAtomIDs(),self._intVolume)))
        for j in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[j].SetVolume(np.sum(self.GetColumnByIDs(self.__GrainBoundaries[j].GetAtomIDs(),self._intVolume)))
        self.blnVolumeAssigned = True
    def AssignAdjustedMeshPoints(self):
        arrShift =0.5*self.__LatticeParameter*(np.sum(self.GetUnitCellBasis(), axis = 0))
        for i in self.__JunctionLineIDs:
            arrMeshPoints = self.__JunctionLines[i].GetMeshPoints()
            lstOfIDs = list(map(lambda x: self.FindBoxAtoms(self.GetAtomsByID(self.GetNonLatticeAtomIDs())[:,0:4],x-arrShift,self.__LatticeParameter*self.GetUnitCellBasis()[0],self.__LatticeParameter*self.GetUnitCellBasis()[1],self.__LatticeParameter*self.GetUnitCellBasis()[2]),arrMeshPoints))
            lstOfAdjustedMeshPoints = []
            for intCounter, lstJLIDs in enumerate(lstOfIDs):
                if lstJLIDs != []:
                    arrPoints = self.PeriodicShiftAllCloser(arrMeshPoints[intCounter], self.GetAtomsByID(lstJLIDs)[:,1:4])
                    lstOfAdjustedMeshPoints.append(np.round(np.mean(arrPoints, axis= 0),1))
            if len(lstOfAdjustedMeshPoints) > 0:
                arrAdjustedMeshPoints = np.vstack(lstOfAdjustedMeshPoints)
                self.__JunctionLines[i].SetAdjustedMeshPoints(arrAdjustedMeshPoints)
        for j in self.__GrainBoundaryIDs:
            arrMeshPoints = self.__GrainBoundaries[j].GetMeshPoints()
            lstOfIDs = list(map(lambda x: self.FindBoxAtoms(self.GetAtomsByID(self.GetNonLatticeAtomIDs())[:,0:4],x-arrShift,self.__LatticeParameter*self.GetUnitCellBasis()[0],self.__LatticeParameter*self.GetUnitCellBasis()[1],self.__LatticeParameter*self.GetUnitCellBasis()[2]),arrMeshPoints))
            lstOfAdjustedMeshPoints = []
            for intCounter, lstJLIDs in enumerate(lstOfIDs):
                if lstJLIDs != []:
                    arrPoints = self.PeriodicShiftAllCloser(arrMeshPoints[intCounter], self.GetAtomsByID(lstJLIDs)[:,1:4])
                    lstOfAdjustedMeshPoints.append(np.round(np.mean(arrPoints, axis= 0),1))
            if len(lstOfAdjustedMeshPoints) > 0:
                arrAdjustedMeshPoints = np.vstack(lstOfAdjustedMeshPoints)
                self.__GrainBoundaries[j].SetAdjustedMeshPoints(arrAdjustedMeshPoints)
        self.blnAdjustedMeshPointsAssigned = True
    def MakeGrainTrees(self):
        lstGrainLabels = self.__GrainLabels
        if 0 in lstGrainLabels:
            lstGrainLabels.remove(0)
        for k in self.__GrainLabels:
            lstIDs = self.GetGrainAtomIDs(k)
            self.__Grains[k] = spatial.KDTree(self.GetAtomsByID(lstIDs)[:,1:4])
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
    def GetRemainingDefectAtomIDs(self, fltTolerance= None):
        if fltTolerance is not(None):
            self.FindDefectiveAtoms(fltTolerance)
        setRemainingDefectIDs = set(self.GetNonLatticeAtomIDs())
        for j in self.__JunctionLineIDs:
            setRemainingDefectIDs = setRemainingDefectIDs.difference(self.__JunctionLines[j].GetAtomIDs())
        for k in self.__GrainBoundaryIDs:
            setRemainingDefectIDs = setRemainingDefectIDs.difference(self.__GrainBoundaries[k].GetAtomIDs())
        return list(setRemainingDefectIDs)    
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
                elif np.sort(lstGrainDistances)[1] < fltGBLength:
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
    def AddGrainBoundary(self, objGrainBoundary: gl.GeneralGrainBoundary):
        self.__GrainBoundaries[objGrainBoundary.GetID()] = objGrainBoundary
    def GetJunctionLine(self, intJunctionLine):
        return self.__JunctionLines[intJunctionLine]
    def AddJunctionLine(self, objJunctionLine: gl.GeneralJunctionLine):
        self.__JunctionLines[objJunctionLine.GetID()] = objJunctionLine
    def GetJunctionLineIDs(self):
        return self.__JunctionLineIDs
    def SetJunctionLineIDs(self):
        self.__JunctionLineIDs = sorted(list(self.__JunctionLines.keys()))
    def GetGrainBoundaryIDs(self):
        return self.__GrainBoundaryIDs
    def SetGrainBoundaryIDs(self):
        self.__GrainBoundaryIDs = sorted(list(self.__GrainBoundaries.keys()))
    def GetAdjacentGrainBoundaries(self, intGrainBoundary: int):
        lstAdjacentGrainBoundaries = []
        lstJunctionLines = self.__GrainBoundaries[intGrainBoundary].GetAdjacentJunctionLines()
        for k in lstJunctionLines:
            lstAdjacentGrainBoundaries.extend(self.__JunctionLines[k].GetAdjacentGrainBoundaries())
        lstAdjacentGrainBoundaries =  list(np.unique(lstAdjacentGrainBoundaries))
        if intGrainBoundary in lstAdjacentGrainBoundaries:
            lstAdjacentGrainBoundaries.remove(intGrainBoundary)
        return lstAdjacentGrainBoundaries
    def GetInteriorGrainAtomIDs(self,intGrainID):
        lstIDs = self.GetGrainAtomIDs(intGrainID)
        arrIndices = gf.GetBoundaryPoints(self.GetAtomsByID(lstIDs)[:,1:4],self.__objRealCell.GetNumberOfNeighbours(),self.__objRealCell.GetNearestNeighbourDistance())
        lstIDs = np.delete(np.array(lstIDs),arrIndices, axis= 0)
        return list(lstIDs)
    def GetExteriorGrainAtomIDs(self,intGrainID):
        setAllAtomIDs = set(self.GetGrainAtomIDs(intGrainID))
        lstExteriorIDs = self.GetInteriorGrainAtomIDs(intGrainID)
        return list(setAllAtomIDs.difference(lstExteriorIDs))
         
class LAMMPSGlobal(LAMMPSAnalysis3D): #includes file writing and reading to correlate labels over different time steps
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int, fltLatticeParameter: float):
        LAMMPSAnalysis3D.__init__(self, fltTimeStep,intNumberOfAtoms, intNumberOfColumns, lstColumnNames, lstBoundaryType, lstBounds,intLatticeType,fltLatticeParameter)
    def ReadInDefectData(self, strFilename: str):
        objDefect = gl.DefectObject(self.GetTimeStep())
        objDefect.ImportData(strFilename)
        for i in objDefect.GetJunctionLineIDs():
            self.AddJunctionLine(objDefect.GetJunctionLine(i))
        self.SetJunctionLineIDs()
        for j in objDefect.GetGrainBoundaryIDs():
            self.AddGrainBoundary(objDefect.GetGrainBoundary(j))
        self.SetGrainBoundaryIDs()
        return objDefect
    def CorrelateDefectData(self,strPreviousFile: str):
        objCorrelate = LAMMPSCorrelate()
        objCorrelate.SetCellVectors(self.GetCellVectors())
        objCorrelate.SetBasisConversion(self.GetBasisConversions())
        objCorrelate.SetBoundaryTypes(self.GetBoundaryTypes())
        objDefect = gl.DefectObject(self.GetTimeStep())
        for i in self.GetGrainBoundaryIDs():
            objDefect.AddGrainBoundary(self.GetGrainBoundary(i))
        for j in self.GetJunctionLineIDs():
            objDefect.AddJunctionLine(self.GetJunctionLine(j))
        objCorrelate.AddDefectObject(objDefect)
        objPreviousDefect = gl.DefectObject()
        objPreviousDefect.ImportData(strPreviousFile) 
        objCorrelate.AddDefectObject(objPreviousDefect)
        objDefect = objCorrelate.CorrelateDefects(self.GetTimeStep(), objPreviousDefect.GetTimeStep()) #use the timesteps passed here rather than the default previous #timestep
        for i in objDefect.GetJunctionLineIDs():
            self.AddJunctionLine(objDefect.GetJunctionLine(i))
        for j in objDefect.GetGrainBoundaryIDs():
            self.AddGrainBoundary(objDefect.GetGrainBoundary(j))
        self.MakeGrainTrees()
        self.SetJunctionLineIDs()
        self.SetGrainBoundaryIDs()
    def WriteDefectData(self, strFileName: str):
        with open(strFileName, 'w') as fdata:
            fdata.write('Time Step \n')
            fdata.write('{} \n'.format(self.GetTimeStep()))
            for i in self.GetJunctionLineIDs():
                fdata.write('Junction Line \n')
                fdata.write('{} \n'.format(i))
                fdata.write('Mesh Points \n')
                fdata.write('{} \n'.format(self.GetJunctionLine(i).GetMeshPoints().tolist()))
                fdata.write('Adjacent Grains \n')
                fdata.write('{} \n'.format(self.GetJunctionLine(i).GetAdjacentGrains()))
                fdata.write('Adjacent Grain Boundaries \n')
                fdata.write('{} \n'.format(self.GetJunctionLine(i).GetAdjacentGrainBoundaries()))
                fdata.write('Periodic Directions \n')
                fdata.write('{} \n'.format(self.GetJunctionLine(i).GetPeriodicDirections()))
                fdata.write('Atom IDs \n')
                fdata.write('{} \n'.format(self.GetJunctionLine(i).GetAtomIDs()))
                if self.blnVolumeAssigned: #optional data writing should be put here with a boolean flag
                    fdata.write('Volume \n')
                    fdata.write('{} \n'.format(self.GetJunctionLine(i).GetVolume()))
                if self.blnPEAssigned:
                    fdata.write('PE \n')
                    fdata.write('{} \n'.format(self.GetJunctionLine(i).GetTotalPE()))
                if self.blnAdjustedMeshPointsAssigned:
                    fdata.write('Adjusted Mesh Points \n')
                    if len(self.GetJunctionLine(i).GetAdjustedMeshPoints()) > 0:
                        fdata.write('{} \n'.format(self.GetJunctionLine(i).GetAdjustedMeshPoints().tolist()))
                    else:
                        fdata.write('{} \n'.format(self.GetJunctionLine(i).GetAdjustedMeshPoints()))
            for k in self.GetGrainBoundaryIDs():
                fdata.write('Grain Boundary \n')
                fdata.write('{} \n'.format(k))
                fdata.write('Mesh Points \n')
                fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetMeshPoints().tolist()))
                fdata.write('Adjacent Grains \n')
                fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetAdjacentGrains()))
                fdata.write('Adjacent Junction Lines \n')
                fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetAdjacentJunctionLines()))
                fdata.write('Periodic Directions \n')
                fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetPeriodicDirections()))
                fdata.write('Atom IDs \n')
                fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetAtomIDs()))
                if self.blnVolumeAssigned: #optional data writing should be put here with a boolean flag
                    fdata.write('Volume \n')
                    fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetVolume()))
                if self.blnPEAssigned:
                    fdata.write('PE \n')
                    fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetTotalPE()))
                if self.blnAdjustedMeshPointsAssigned:
                    fdata.write('Adjusted Mesh Points \n')
                    if len(self.GetGrainBoundary(k).GetAdjustedMeshPoints()) > 0:
                        fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetAdjustedMeshPoints().tolist()))
                    else:
                        fdata.write('{} \n'.format(self.GetGrainBoundary(k).GetAdjustedMeshPoints()))
            fdata.close()

class LAMMPSCorrelate(object): #add grain boundaries and junction lines over different steps using dfc files
    def __init__(self):
        self.__dctDefects = dict()
        self.__GlobalJunctionLines = dict()
        self.__GlobalGrainBoundaries = dict()
        self.__CellVectors = gf.StandardBasisVectors(3)
        self.__BasisConversion = gf.StandardBasisVectors(3)
        self.__BoundaryTypes = ['pp','pp','pp']
    def AddDefectObject(self, objDefect: gl.DefectObject):
        self.__dctDefects[objDefect.GetTimeStep()] = objDefect
    def GetDefectObject(self, fltTimeStep):
        return self.__dctDefects[fltTimeStep]
    def GetTimeSteps(self):
        return sorted(list(self.__dctDefects.keys()))
    def GetTimeStepPosition(self, fltTimeStep)->int:
        return list(self.__dctDefects.keys()).index(fltTimeStep)
    def ReadInData(self, strFilename: str, blnCorrelateDefects = False):
        objDefect = gl.DefectObject()
        objDefect.ImportData(strFilename)   
        for i in objDefect.GetGrainBoundaryIDs(): #global IDs are set from the first time step
            objDefect.GetGrainBoundary(i).SetID(i)
            objDefect.AddGrainBoundary(objDefect.GetGrainBoundary(i))
        for j in objDefect.GetJunctionLineIDs():
            objDefect.GetJunctionLine(j).SetID(j)
            objDefect.AddJunctionLine(objDefect.GetJunctionLine(j))
        self.__dctDefects[objDefect.GetTimeStep()] = objDefect    
        if len(list(self.__dctDefects.keys())) > 1 and blnCorrelateDefects:
            intLastTimeStep = self.GetTimeSteps()[-1]
            self.CorrelateDefects(objDefect.GetTimeStep(), intLastTimeStep)
    def CorrelateDefects(self,intTimeStep: int,intLastTimeStep: int):
        objDefect = self.__dctDefects[intTimeStep]
        objPreviousDefect = self.__dctDefects[intLastTimeStep]
        lstGB = []
        lstCurrentGBIDs = objDefect.GetGrainBoundaryIDs()
        for j in lstCurrentGBIDs:
            lstGB.append(objDefect.GetGrainBoundary(j).GetMeshPoints())
        lstPreviousGB = []
        lstPreviousGBIDs = objPreviousDefect.GetGrainBoundaryIDs()    
        for k in lstPreviousGBIDs:
            lstPreviousGB.append(objPreviousDefect.GetGrainBoundary(k).GetMeshPoints())
        arrHausdorff = self.MakeHausdorffDistanceMatrix(lstGB, lstPreviousGB) #periodic Hausdroff distance used to match global defects
        if len(lstPreviousGBIDs) != len(lstCurrentGBIDs):
                warnings.warn('Number of grain boundaries changed from ' + str(len(lstPreviousGBIDs)) + ' to ' + str(len(lstCurrentGBIDs)) 
                + ' at time step ' + str(intTimeStep)) 
        while (len(lstCurrentGBIDs) > 0 and len(lstPreviousGBIDs) > 0):
            tupCurrentPrevious = np.unravel_index(arrHausdorff.argmin(), arrHausdorff.shape)
            intCurrent = int(tupCurrentPrevious[0])
            intPrevious = int(tupCurrentPrevious[1])
            objDefect.GetGrainBoundary(lstCurrentGBIDs[intCurrent]).SetID(lstPreviousGBIDs[intPrevious])
            del lstCurrentGBIDs[intCurrent]
            del lstPreviousGBIDs[intPrevious]
            arrHausdorff = np.delete(arrHausdorff, intCurrent, axis = 0)
            arrHausdorff = np.delete(arrHausdorff, intPrevious, axis = 1)
        while len(lstCurrentGBIDs) > 0: #there are more current GBs than previous GBs
            intID = lstCurrentGBIDs.pop(0)
            if len(objDefect.GetGrainBoundaryIDs()) > 0:
                intLastGlobalID = max(objDefect.GetGrainBoundaryIDs())
            else:
                intLastGlobalID = 0
            objDefect.GetGrainBoundary(intID).SetID(intLastGlobalID + 1)
        lstJL = []
        lstCurrentJLIDs = objDefect.GetJunctionLineIDs()
        for l in lstCurrentJLIDs:
            lstJL.append(objDefect.GetJunctionLine(l).GetMeshPoints())
        lstPreviousJL = []
        lstPreviousJLIDs = objPreviousDefect.GetJunctionLineIDs()
        for m in lstPreviousJLIDs:
            lstPreviousJL.append(objPreviousDefect.GetJunctionLine(m).GetMeshPoints())
        arrHausdorff = self.MakeHausdorffDistanceMatrix(lstJL, lstPreviousJL) #periodic Hausdroff distance used to match global defects
        if len(lstPreviousJLIDs) != len(lstCurrentJLIDs):
                warnings.warn('Number of junction lines changed from ' + str(len(lstPreviousJLIDs)) + ' to ' + str(len(lstCurrentJLIDs)) 
                + ' at time step ' + str(intTimeStep)) 
        while (len(lstCurrentJLIDs) > 0 and len(lstPreviousJLIDs) > 0):
            tupCurrentPrevious = np.unravel_index(arrHausdorff.argmin(), arrHausdorff.shape)
            intCurrent = int(tupCurrentPrevious[0])
            intPrevious = int(tupCurrentPrevious[1])
            objDefect.GetJunctionLine(lstCurrentJLIDs[intCurrent]).SetID(lstPreviousJLIDs[intPrevious])
            del lstCurrentJLIDs[intCurrent]
            del lstPreviousJLIDs[intPrevious]
            arrHausdorff = np.delete(arrHausdorff, intCurrent, axis = 0)
            arrHausdorff = np.delete(arrHausdorff, intPrevious, axis = 1)
        while len(lstCurrentJLIDs) > 0: #there are more current GBs than previous GBs
            intID = lstCurrentJLIDs.pop(0)
            if len(objDefect.GetJunctionLineIDs()) > 0:
                intLastGlobalID = max(objDefect.GetJunctionLineIDs())
            else:
                intLastGlobalID = 0
            objDefect.GetJunctionLine(intID).SetID(intLastGlobalID + 1)
        objUpdatedDefect = gl.DefectObject(objDefect.GetTimeStep())
        for n in objDefect.GetGrainBoundaryIDs(): #puts the objects back into new defect object which uses their updated IDs as the dictionary key
            objUpdatedDefect.AddGrainBoundary(objDefect.GetGrainBoundary(n))
        for m in objDefect.GetJunctionLineIDs(): #puts the objects back into new defect object which uses their updated IDs as the dictionary key
            objUpdatedDefect.AddJunctionLine(objDefect.GetJunctionLine(m))    
        self.__dctDefects[intTimeStep] = objUpdatedDefect
        return objUpdatedDefect
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
    def MakeHausdorffDistanceMatrix(self, lstOfCurrentMeshPoints: list,lstOfPreviousMeshPoints: list)-> np.array:
        arrDistanceMatrix = np.zeros([len(lstOfCurrentMeshPoints), len(lstOfPreviousMeshPoints)])
        for intCurrent,arrCurrent in enumerate(lstOfCurrentMeshPoints):
            arrCurrentMean = np.mean(arrCurrent, axis = 0)
            for intPrevious,arrPrevious in enumerate(lstOfPreviousMeshPoints):
                arrPreviousMean = np.mean(arrPrevious, axis = 0)
                arrPeriodicShift = gf.PeriodicEquivalentMovement(arrCurrentMean, arrPreviousMean, self.__CellVectors, self.__BasisConversion, self.__BoundaryTypes)[2]
                arrPrevious = arrPrevious - arrPeriodicShift
                arrDistanceMatrix[intCurrent, intPrevious] = max(spatial.distance.directed_hausdorff(arrCurrent, arrPrevious)[0], spatial.distance.directed_hausdorff(arrPrevious, arrCurrent)[0]) #Hausdorff distance is not symmetric in general and so choose the larger of the two measure.
        return arrDistanceMatrix
        
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
        arrOut = ndimage.filters.gaussian_filter(arrOut, 2, mode = 'wrap')
        fltThreshold = threshold_otsu(arrOut)
        arrOut = (arrOut > fltThreshold)
        arrOut = ndimage.binary_dilation(arrOut, np.ones([3,3,3]))
        arrOut = arrOut.astype('bool').astype('int') # convert to binary
        self.__Iterations = 3
        self.__BinaryArray = arrOut
        self.__InvertBinary = np.invert(self.__BinaryArray.astype('bool')).astype('int')
        self.__Grains, intGrainLabels  = ndimage.measurements.label(self.__InvertBinary, np.array([[[0,0,0],[0,1,0],[0,0,0]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,0],[0,1,0],[0,0,0]]])) #don't allow grains to connect diagonally
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
            self.MakeReturnGrains()
            self.ExpandGrains() #expand all the grains until the grain boundaries are dissolved
    def MakeReturnGrains(self):
        self.__ReturnGrains = np.copy(self.__Grains) #This array is used to evaluate all the lattice points 
        self.__ReturnGrains[self.__ReturnGrains == 0] = -1 #if they are in a defective region assign the value -1
    def GetDefectPositions(self):
        return self.__DefectPositions
    def MergeEquivalentGrains(self):
        for j in self.__EquivalentGrains:
            if len(j) > 0:
                for k in j[1:]:
                    self.__Grains[self.__Grains == k] = j[0]
        lstValues = list(np.unique(self.__Grains))
        lstValues.remove(0)
        for intIndex, intValue in enumerate(lstValues):
            self.__Grains[self.__Grains == intValue] = intIndex+1 
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
    def ReturnGrains(self, inPoints: np.array, blnExpanded = True)->list:
        inPoints = np.matmul(inPoints, self.__Scaling)
        inPoints = np.matmul(inPoints, self.__BasisConversion)
        inPoints = np.round(inPoints, 0).astype('int')
        inPoints = np.mod(inPoints, self.__ModArray)
        if len(np.argwhere(self.__ExpandedGrains == 0)) > 0:
            warnings.warn('Expanded grain method has not removed all grain boundary atoms')
        if blnExpanded:
            return list(self.__ReturnGrains[inPoints[:,0],inPoints[:,1],inPoints[:,2]])
        else:
            arrZeros = np.argwhere(self.__BinaryArray == 0)
            self.__ReturnGrains = np.copy(self.__ExpandedGrains)
            for j in arrZeros:
                self.__ReturnGrains[j[0],j[1],j[2]] = 0
            return list(self.__ReturnGrains[inPoints[:,0],inPoints[:,1],inPoints[:,2]])                     
    def FindGrainBoundaries(self):
        arrGrainBoundaries = np.zeros(self.__ModArray) #temporary array 
        lstGrainBoundaryList = []
        for j in self.__Coordinates:
            j = j.astype('int')
            arrBox  = self.__ExpandedGrains[gf.WrapAroundSlice(np.array([[j[0],j[0]+2],[j[1],j[1]+2],[j[2],j[2]+2]]),self.__ModArray)]
            lstValues = list(np.unique(arrBox))
            if len(lstValues) ==2:
                if lstValues not in lstGrainBoundaryList:
                    lstGrainBoundaryList.append(lstValues)
                arrGrainBoundaries[j[0],j[1],j[2]] = 1 + lstGrainBoundaryList.index(lstValues)
        self.__GrainBoundariesArray = self.FindPeriodicBoundaries(arrGrainBoundaries)
        lstValues = list(np.unique(self.__GrainBoundariesArray)) #renumber the array sequentially for convenience starting at 1
        if 0 in lstValues:
            lstValues.remove(0)
        for intIndex, intValue in enumerate(lstValues):
            self.__GrainBoundariesArray[self.__GrainBoundariesArray == intValue] = intIndex+1 
        lstValues = list(np.unique(self.__GrainBoundariesArray))
        if 0 in lstValues:
            lstValues.remove(0)
        self.__GrainBoundaryIDs = lstValues
    def FindJunctionLines(self):
        self.FindGrainBoundaries()
        arrJunctionLines = np.zeros(self.__ModArray)
        lstJunctionLineList = []
        arrPoints = np.argwhere(self.__GrainBoundariesArray == 0) #junction line points have value 0 in the grain boundaries array
        for j in arrPoints:
            j = j.astype('int')
            arrBox  = self.__GrainBoundariesArray[gf.WrapAroundSlice(np.array([[j[0]-1,j[0]+2],[j[1]-1,j[1]+2],[j[2]-1,j[2]+2]]),self.__ModArray)]
            lstValues = list(np.unique(arrBox)) #select a 3 x 3 x 3 cube with central value of 0
            lstValues.remove(0)
            if len(lstValues) > 2: #count the number of distinct grain boundaries
                if lstValues not in lstJunctionLineList:
                    lstJunctionLineList.append(lstValues)
                arrJunctionLines[j[0],j[1],j[2]] = 1 + lstJunctionLineList.index(lstValues)
        self.__JunctionLinesArray = self.FindPeriodicBoundaries(arrJunctionLines)
        lstValues = list(np.unique(self.__JunctionLinesArray)) #renumber the array sequentially for convenience starting at 1
        if 0 in lstValues:
            lstValues.remove(0)
        for intIndex, intValue in enumerate(lstValues):
            self.__JunctionLinesArray[self.__JunctionLinesArray == intValue] = intIndex+1 
        lstValues = list(np.unique(self.__JunctionLinesArray))
        if 0 in lstValues:
            lstValues.remove(0)
        self.__JunctionLineIDs = lstValues
    def FindPeriodicBoundaries(self,inArray: np.array):
        inArray = inArray.astype('int')
        intMaxNumber = 0
        lstValues = list(np.unique(inArray))
        arrUpdatedValues = np.zeros(np.shape(inArray))
        if 0 in lstValues:
            lstValues.remove(0)
        for j in lstValues:
            arrPoints = np.argwhere(inArray == j)
            arrReturn, intLabels = ndimage.measurements.label(inArray == j, np.ones([3,3,3]))
            arrReturn = self.CheckPeriodicity(arrPoints, arrReturn)
            arrReturn[arrReturn > 0 ] += intMaxNumber
            arrUpdatedValues += arrReturn
            intMaxNumber += intLabels
        return arrUpdatedValues.astype('int')
    def CheckPeriodicity(self, inPoints: np.array, arrCurrent: np.array):
        arrExtended = np.zeros(np.shape(arrCurrent) + 2*np.ones(3).astype('int'))
        arrExtended[1:-1,1:-1,1:-1] = np.copy(arrCurrent)
        arrExtended[0,:,:] = arrExtended[-2,:,:]
        arrExtended[-1,:,:] = arrExtended[1,:,:]
        arrExtended[:,0,:] = arrExtended[:,-2,:]
        arrExtended[:,-1,:] = arrExtended[:,1,:]
        arrExtended[:,:,0] = arrExtended[:,:,-2]
        arrExtended[:,:,-1] = arrExtended[:,:,1]  
        arrTotalPoints = np.argwhere(arrExtended > 0) - np.ones(3)
        arrDistanceMatrix = spatial.distance_matrix(arrTotalPoints,arrTotalPoints)
        arrClosePoints = np.argwhere(arrDistanceMatrix < 2) #allows 3d diagonal connectivity
        arrCurrent = arrCurrent.astype('int')
        for j in arrClosePoints:
            arrRow = tuple(zip(np.mod(arrTotalPoints[j[0]], self.__ModArray).astype('int')))
            arrColumn = tuple(zip(np.mod(arrTotalPoints[j[1]], self.__ModArray).astype('int'))) 
            tupPairs = (arrCurrent[arrRow][0], arrCurrent[arrColumn][0])
            if tupPairs[0] != tupPairs[1]:
                arrCurrent[arrCurrent == max(tupPairs)] = min(tupPairs)
        return arrCurrent
    def BoxIsNotWrapped(self, inPoint: np.array, n: int)->bool: # checks to see if a box array lies inside the simulation cell without wrapping
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
            arrDistanceMatrix  = pairwise_distances(arrTotal)    
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
    def GetGrainCentre(self, intGrainNumber):
        arrPoints = np.argwhere(self.__Grains == intGrainNumber)
        return np.mean(arrPoints, axis =0)