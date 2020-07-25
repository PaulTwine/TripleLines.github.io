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
    def SetColumnValueByIDs(self,lstOfAtomIDs: list, intColumn: int, arrValues: np.array):
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
        self._intStructureType = int(self.GetColumnNames().index('StructureType'))
        self._intPositionX = int(self.GetColumnNames().index('x'))
        self._intPositionY = int(self.GetColumnNames().index('y'))
        self._intPositionZ = int(self.GetColumnNames().index('z'))
        self._intPE = int(self.GetColumnNames().index('c_pe1'))
        self.CellHeight = np.linalg.norm(self.GetCellVectors()[2])
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
        lstLatticeAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == self._LatticeStructure)[0])
        lstUnknownAtoms = list(np.where(np.isin(self.GetColumnByIndex(self._intStructureType).astype('int') ,[0,1],invert=True))[0])
        self.__LatticeAtoms = lstLatticeAtoms
        self.__NonLatticeAtoms = lstOtherAtoms + lstUnknownAtoms
        self.__OtherAtoms = lstOtherAtoms
        self.__UnknownAtoms = lstUnknownAtoms
        self.FindDefectiveAtoms()
        self.FindNonDefectiveAtoms()
    def FindDefectiveAtoms(self, fltTolerance = None):
        if fltTolerance is None:
            fltStdLatticeValue = np.std(self.GetLatticeAtoms()[:,self._intPE])
            fltTolerance = 1.96*fltStdLatticeValue #95% limit assuming Normal distribution
        fltMeanLatticeValue = np.mean(self.GetLatticeAtoms()[:,self._intPE])
        lstDefectiveAtoms = np.where((self.GetColumnByIndex(self._intPE) > fltMeanLatticeValue +fltTolerance) | (self.GetColumnByIndex(self._intPE) < fltMeanLatticeValue - fltTolerance))[0]
        self.__DefectiveAtoms = self.GetAtomData()[lstDefectiveAtoms,0].astype('int')
        return self.GetRows(lstDefectiveAtoms)
    def FindNonDefectiveAtoms(self,fltTolerance = None):
        if fltTolerance is None:
            fltStdLatticeValue = np.std(self.GetLatticeAtoms()[:,self._intPE])
            fltTolerance = 1.96*fltStdLatticeValue #95% limit assuming Normal distribution
        fltMeanLatticeValue = np.mean(self.GetLatticeAtoms()[:,self._intPE])
        lstNonDefectiveAtoms = np.where((self.GetColumnByIndex(self._intPE) <= fltMeanLatticeValue +fltTolerance) | (self.GetColumnByIndex(self._intPE) >= fltMeanLatticeValue  - fltTolerance))[0]
        self.__NonDefectiveAtoms = self.GetAtomData()[lstNonDefectiveAtoms,0].astype('int')
        return self.GetRows(lstNonDefectiveAtoms)
    def GetLatticeAtomIDs(self):
        return self.__LatticeAtoms
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
            for k in range(len(inVector2)):
                arrPeriodicDistance[j,k] = self.PeriodicMinimumDistance(inVector1[j],inVector2[k])
        return arrPeriodicDistance
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        inVector2 = self.PeriodicShiftCloser(inVector1, inVector2)
        return np.linalg.norm(inVector2-inVector1, axis=0)
    def FindNonGrainMediod(self, inPoint: np.array, fltRadius: float, bln2D= True):
        arrReturn = np.ones(3)*self.CellHeight/2
        lstPointsIndices = []
        lstPointsIndices = self.FindCylindricalAtoms(self.GetDefectiveAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            arrPoint = gf.FindGeometricMediod(arrPoints, bln2D)
            arrReturn[0:2] = arrPoint
            return arrReturn  
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
        lstPointsIndices = self.FindCylindricalAtoms(self.GetLatticeAtoms()[:,0:self._intPositionZ+1],inPoint,fltRadius, self.CellHeight, True)
        if len(lstPointsIndices) > 0:
            lstPointsIndices = list(np.unique(lstPointsIndices))
            arrPoints = self.GetAtomsByID(lstPointsIndices)[:,self._intPositionX:self._intPositionZ+1]
            arrPoints = self.PeriodicShiftAllCloser(inPoint, arrPoints)
            return np.mean(arrPoints, axis=0)  
        else:
            return inPoint  
    def SphereLiesInCell(self, arrCentre: np.array, fltRadius: float)->bool:
        arrProjections = np.matmul(arrCentre, self.__PlaneNormalVectors)
        blnInside = False
        if np.all(arrProjections > fltRadius) and np.all(self.__PlaneNormalLimits -arrProjections > fltRadius):
            blnInside = True
        return blnInside        
    def FindSphericalAtoms(self,arrPoints, arrCentre: np.array, fltRadius: float, blnPeriodic =True)->list:
        lstIndices = []
        if blnPeriodic:
            blnPeriodic = self.SphereLiesInCell(arrCentre, fltRadius)
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
        self.__TripleLines = dict()
        self.__Grains = dict()
        self.__LatticeParameter = fltLatticeParameter
        self.__GrainLabels = []
        self.__QCTripleLinePoints = []
        self.__QCGBPoints = []
        self.__TripleLineIDs = []
    def SetLatticeParameter(self, fltParameter: float):
        self.__LatticeParameter = fltParameter
    def LabelAtomsByGrain(self):
        self.__QuantisedCuboidPoints = QuantisedCuboidPoints(self.GetDefectiveAtoms()[:,1:4],self.GetUnitBasisConversions(),self.GetCellVectors(),self.__LatticeParameter*np.ones(3),10)
#        lstGrainAtoms = list(set(self.GetLatticeAtomIDs()) & set(self.GetNonDefectiveAtomIDs()))
        lstGrainAtoms = self.GetNonDefectiveAtomIDs()
        lstGrainNumbers = self.__QuantisedCuboidPoints.ReturnGrains(self.GetAtomsByID(lstGrainAtoms)[:,1:4])
        self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]), 'GrainNumber')
        arrGrainNumbers = np.array([lstGrainNumbers])
        self.__GrainLabels = list(np.unique(lstGrainNumbers))
        np.reshape(arrGrainNumbers, (len(lstGrainNumbers),1))
        self.__intGrainNumber = self.GetNumberOfColumns()-1
        self.SetColumnValueByIDs(lstGrainAtoms, self.__intGrainNumber, arrGrainNumbers)
        self.__QuantisedCuboidPoints.FindTripleLines()
        self.__TripleLineIDs = self.__QuantisedCuboidPoints.GetTripleLineIDs()
        for i in self.__TripleLineIDs:
            self.__TripleLines[i] = gl.GeneralTripleLine(self.__QuantisedCuboidPoints.GetTripleLinePoints(i),i)
        self.__GrainBoundaryIDs = self.__QuantisedCuboidPoints.GetGrainBoundaryIDs()
        for j in self.__GrainBoundaryIDs:
            self.__GrainBoundaries[j] = gl.GeneralGrainBoundary(self.__QuantisedCuboidPoints.GetGrainBoundaryPoints(j),j)
            for k in self.__GrainBoundaries[j].GetMeshPoints():
                self.CheckBoundaries(k, 3*self.__LatticeParameter)
        self.MakeGrainTrees()
    def MakeGrainTrees(self):
        for k in self.__GrainLabels:
            lstIDs = self.GetGrainAtomIDs(k)
            self.__Grains[k] = spatial.KDTree(self.GetAtomsByID(lstIDs)[:,1:4])
    # def CheckBoundaries(self):#check grain points haven't seeped over the gr
    #     for i in self.__GrainLabels:
    #         arrGrainPoints = self.GetAtomsByID(self.GetGrainAtomIDs(i))[:,0:4]
    #         lstRemainingGrainLabels = list(np.copy(self.__GrainLabels))
    #         lstRemainingGrainLabels.remove(i)
    #         clustering = DBSCAN(self.__LatticeParameter/np.sqrt(2), 1).fit(arrGrainPoints[:,1:4])
    #         lstValues = clustering.labels_
    #         arrUniqueValues, arrCounts = np.unique(lstValues, return_counts=True)
    #         arrMax = np.argmax(arrCounts)
    #         arrUniqueValues = np.delete(arrUniqueValues, arrUniqueValues[arrMax]) #assume the largest cluster is correct
    #         for j in arrUniqueValues:
    #             arrPoints = arrGrainPoints[lstValues == j, 0:4]
    #             lstIDs = arrPoints[:,0]
    #             arrPoints = arrPoints[:,1:4]
    #             for k in range(len(lstRemainingGrainLabels)):
    #                 lstDistances = []
    #                 lstDistances.append(min(self.__Grains[lstRemainingGrainLabels[k]].query(arrPoints,1)[0]))
    #             intMin = np.argmin(lstDistances)
    #             if lstDistances[intMin] <= self.__LatticeParameter/np.sqrt(2):
    #                 self.SetColumnValueByIDs(lstIDs, self.__intGrainNumber, lstRemainingGrainLabels[intMin]*np.ones(len(lstIDs)))
    #                 print('points redefined')
    #                 self.MakeGrainTrees()

    def CheckBoundaries(self, arrCentre: np.array, fltRadius):
        lstSurroundingAtomIDs = self.FindSphericalAtoms(self.GetNonDefectiveAtoms()[:,0:4],arrCentre, fltRadius)
        arrPointsAndValues = self.GetAtomsByID(lstSurroundingAtomIDs)[:,[0,1,2,3,self.__intGrainNumber]] #atomic positions, ID and Grain number
        lstGrains =  list(np.unique(self.GetAtomsByID(lstSurroundingAtomIDs)[:,self.__intGrainNumber]).astype('int'))
        if 0 in lstGrains:
            print("Error 0 grain number")
        dctGrainPoints = dict()
        dctGrainIDs = dict()
        for i in lstGrains:
            arrPoints = arrPointsAndValues[arrPointsAndValues[:,4]==i]
            arrPoints = arrPoints[:,1:4]
            dctGrainPoints[i] = self.PeriodicShiftAllCloser(arrCentre,arrPoints)
            dctGrainIDs[i] = np.array(lstSurroundingAtomIDs)[arrPointsAndValues[:,4] ==i]
        lstOfKeys = list(dctGrainPoints.keys())
        while len(lstOfKeys) > 0:
            j = lstOfKeys.pop()
            clustering = DBSCAN(self.__LatticeParameter/np.sqrt(2), 1).fit(dctGrainPoints[j])
            lstValues = clustering.labels_
            arrUniqueValues, arrCounts = np.unique(lstValues, return_counts=True)
            arrUniqueValues = np.delete(arrUniqueValues, arrUniqueValues[np.argmax(arrCounts)]) #assume the largest cluster is correct
            if len(arrUniqueValues) > 1:
                for k in arrUniqueValues:
                    arrClusterPoints = dctGrainPoints[j][lstValues == k]
                    lstClusterIDs = dctGrainIDs[j][lstValues == k]
                    fltMin = fltRadius
                    for l in lstOfKeys:
                        arrDistanceMatrix = spatial.distance_matrix(arrClusterPoints, dctGrainPoints[l])
                        fltCurrentMin = np.min(arrDistanceMatrix)
                        if fltCurrentMin < fltMin:
                            fltMin = fltCurrentMin
                            intClosestGrain = l
                    if fltMin <= 1.01*self.__LatticeParameter/np.sqrt(2):
                        self.SetColumnValueByIDs(lstClusterIDs, self.__intGrainNumber, intClosestGrain*np.ones(len(lstClusterIDs)))

        # for i in self.__GrainBoundaries[1].GetMeshPoints():
        #     lstSurroundingAtomIDs = self.FindSphericalAtoms(self.GetNonDefectiveAtoms()[:,0:4],i, 2*self.__LatticeParameter)
        #     arrPoints = self.GetAtomsByID(lstSurroundingAtomIDs)[:,[0,1,2,3,self.__intGrainNumber]]
        #     lstGrains =  list(np.unique(self.GetAtomsByID(lstSurroundingAtomIDs)[:,self.__intGrainNumber]).astype('int'))
        #     lstGrains.remove(0)
        #     dctGrainPoints = dict()
        #     if len(lstGrains) == 2:
        #             for k in lstOfGrains:
        #                 arrMediod = np.zeros([3, len(lstGrains)])
        #                 dctGrainPoints[k] = arrPoints[arrPoints[:,4] == lstGrain[k]
        #                 arrMediod[k] = gf.FindGeometricMediod(dctGrainPoints[k][:,1:4])
        #                 arrProjections= np.matmul(dctGrainPoints[k][:,1:4]-i, arrMediod-i)
        #                 arrOfIndices = np.where(arrProjections < 0)
                    
                    # if np.size(arrOfIndices) > 0:
                    #     arrIndices0 =np.unique(arrOfIndices[0])
                    #     arrIndices1 = np.unique(arrOfIndices[1])
                    #     intMax = np.argmax([len(arrIndices0),len(arrIndices1)])
                    # if intMax == 0:
                    #     intGrainValue = l
                    #     lstIndicesID = dctGrainPoints[k][arrIndices0,0]
                    # else:
                    #     intGrainValue = intID 
                    #     lstIndicesID = dctGrainPoints[l][arrIndices1,0]
                    #     self.SetColumnValueByIDs(lstIndicesID, self.__intGrainNumber, intGrainValue*np.ones(len(lstIndicesID)))
        # for i in self.__GrainBoundaryIDs:
        #     for j in self.__GrainBoundaries[i].GetMeshPoints():
        #         dctOfGrainPoints = dict()
        #         lstIDs = self.FindSphericalAtoms(self.GetAtomData()[:,0:4],j, self.__LatticeParameter)
        #         arrPoints = self.GetAtomsByID(lstIDs)[:,[0,1,2,3,4,self.__intGrainNumber]]
        #         lstGrains = list(np.unique(arrPoints[:,5].astype('int')))
        #         lstGrains.remove(0)
        #         if len(lstGrains) > 1:
        #             for k in lstGrains:
        #                 dctOfGrainPoints[k] = arrPoints[arrPoints[:,5] == k]
        #             while len(lstGrains) > 0:
        #                 intID = lstGrains.pop(0)
        #                 for l in lstGrains:
        #                     arrDistanceMatrix= spatial.distance_matrix(dctOfGrainPoints[intID][:,1:4],dctOfGrainPoints[l][:,1:4])
        #                     arrOfIndices = np.where(arrDistanceMatrix < self.__LatticeParameter)
        #                     if np.size(arrOfIndices) > 0:
        #                         arrIndices0 =np.unique(arrOfIndices[0])
        #                         arrIndices1 = np.unique(arrOfIndices[1])
        #                         intMax = np.argmax([len(arrIndices0),len(arrIndices1)])
        #                         if intMax == 0:
        #                             intGrainValue = l
        #                             lstIndicesID = dctOfGrainPoints[intID][arrIndices0,0]
        #                         else:
        #                             intGrainValue = intID 
        #                             lstIndicesID = dctOfGrainPoints[l][arrIndices1,0]
        #                         self.SetColumnValueByIDs(lstIndicesID, self.__intGrainNumber, intGrainValue*np.ones(len(lstIndicesID)))
    def FindAllTripleLines(self):
        for i in self.__TripleLineIDs:
            lstTJAtomIDs = []
            lstCloseAtoms = []
            for s in self.__TripleLines[i].GetMeshPoints():
                lstCloseAtoms.extend(self.FindSphericalAtoms(self.GetDefectiveAtoms()[:,0:4],s, np.sqrt(7)*self.__LatticeParameter))
            arrAtoms = self.GetAtomsByID(lstCloseAtoms)
            for j in arrAtoms:
                arrPosition = j[1:4]
                intID = j[0].astype('int')
                lstVectors = []
                for k in self.__GrainLabels:
                    arrPeriodicVariants = self.PeriodicEquivalents(arrPosition)
                    lstDistances,lstIndices = self.__Grains[k].query(arrPeriodicVariants,1)
                    intIndex = lstIndices[np.argmin(lstDistances)]
                    arrGrainPoint = self.__Grains[k].data[intIndex]
                    lstVectors.append(arrGrainPoint)
                arrPoints = np.vstack(lstVectors)
                arrPoints = self.PeriodicShiftAllCloser(arrPosition, arrPoints)
                arrDistanceMatrix = spatial.distance_matrix(arrPoints, arrPoints)
                fltMaxDistance = np.max(arrDistanceMatrix[arrDistanceMatrix > 0])
                fltMinDistance = np.min(arrDistanceMatrix[arrDistanceMatrix > 0])
                if len(arrPoints) ==3:
                    arrCentre = gf.EquidistantPoint(*arrPoints)
                    fltRadius = np.linalg.norm(arrPoints[0]-arrCentre)
                else:
                    arrCentre = np.mean(arrPoints, axis=0)
                    fltRadius = np.max(np.linalg.norm(arrPoints - arrCentre,axis=1))
                if fltMaxDistance < 2*fltMinDistance and fltRadius < 2*self.__LatticeParameter:
                    lstTJAtomIDs.append(intID)
                self.__TripleLines[i].SetAtomIDs(lstTJAtomIDs)    
    def GetGrainBoundaryAtomIDs(self, intGrainBoundary = None):
        if intGrainBoundary is None:
            lstGrainBoundaryIDs = []
            for j in self.__GrainBoundaryIDs:
                lstGrainBoundaryIDs.extend(self.__GrainBoundaries[j].GetAtomIDs())
            return lstGrainBoundaryIDs
        else:
            return self.__GrainBoundaries[intGrainBoundary].GetAtomIDs()
    def GetTripleLineAtomIDs(self, intTripleLine = None):
        if intTripleLine is None:
            lstTripleLineIDs = []
            for j in self.__TripleLineIDs:
                lstTripleLineIDs.extend(self.__TripleLines[j].GetAtomIDs())
            return lstTripleLineIDs
        else:
            return self.__TripleLines[intTripleLine].GetAtomIDs()
    def GetGrainAtomIDs(self, intGrainNumber: int):
        lstGrainAtoms = list(np.where(self.GetColumnByName('GrainNumber').astype('int') == intGrainNumber)[0])
        return self.GetAtomData()[lstGrainAtoms,0].astype('int')
    def GetGrainLabels(self):
        return self.__GrainLabels
    def GetTripleLineMeshPoints(self, intTripleLine = None):
        if intTripleLine is None:
            lstTripleLinePoints = []
            for j in self.__TripleLineIDs:
                lstTripleLinePoints.append(self.__TripleLines[j].GetMeshPoints())
            return np.vstack(lstTripleLinePoints)
        else:
            return self.__TripleLines[intTripleLine].GetMeshPoints()
    def GetGrainBoundaryMeshPoints(self, intGrainBoundary = None):
        if intGrainBoundary is None:
            lstGrainBoundaryIDs = []
            for j in self.__GrainBoundaryIDs:
                lstGrainBoundaryIDs.append(self.__GrainBoundaries[j].GetMeshPoints())
            return np.vstack(lstGrainBoundaryIDs)
        else:
            return self.__GrainBoundaries[intGrainBoundary].GetMeshPoints()
    
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
        self.__TripleLinesArray = np.copy(arrValues)
        self.__GrainBoundariesArray = np.copy(arrValues)
        self.__TripleLines = []
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
        arrOut = arrValues
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
    def ReturnGrains(self, inPoints: np.array)->list:
        inPoints = np.matmul(inPoints, self.__Scaling)
        inPoints = np.matmul(inPoints, self.__BasisConversion)
        inPoints = np.mod(inPoints, self.__ModArray).astype('int')
        #print(np.argwhere(self.__ExpandedGrains == 0)) debugging only
        return list(self.__ExpandedGrains[inPoints[:,0],inPoints[:,1],inPoints[:,2]])                    
    def FindTripleLines(self):
        lstGrainBoundaryList = []
        for j in self.__Coordinates:
            j = j.astype('int')
            arrBox  = self.__ExpandedGrains[gf.WrapAroundSlice(np.array([[j[0],j[0]+2],[j[1],j[1]+2],[j[2],j[2]+2]]),self.__ModArray)]
            lstValues = list(np.unique(arrBox))
            if len(lstValues) > 2:
                self.__TripleLinesArray[j[0],j[1],j[2]] = len(lstValues)
            elif len(lstValues) ==2:
                if lstValues not in lstGrainBoundaryList:
                    lstGrainBoundaryList.append(lstValues)
                self.__GrainBoundariesArray[j[0],j[1],j[2]] = 1 + lstGrainBoundaryList.index(lstValues)
        self.__TripleLinesArray = measure.label(self.__TripleLinesArray).astype('int')
        self.__TripleLineIDs = list(np.unique(self.__TripleLinesArray))
        self.__TripleLineIDs.remove(0)
        self.__GrainBoundariesArray = measure.label(self.__GrainBoundariesArray).astype('int')
        self.__GrainBoundaryIDs = list(np.unique(self.__GrainBoundariesArray))
        self.__GrainBoundaryIDs.remove(0)              
    def GetAdjoiningTripleLines(self, intGrainBoundaryID: int):
            j = intGrainBoundaryID
            arrBox  = self.__TripleLinesArray[gf.WrapAroundSlice(np.array([[j[0]-1,j[0]+2],[j[1]-1,j[1]+2],[j[2]-1,j[2]+2]]),self.__ModArray)]
            return list(np.unique(arrBox))
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
    def GetTripleLinePoints(self, intTripleLineID = None)->np.array:
        if intTripleLineID is None:
            return np.matmul(np.matmul(np.argwhere(self.__TripleLinesArray.astype('int') != 0) +np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion) 
        elif intTripleLineID in self.__TripleLineIDs: 
            return np.matmul(np.matmul(np.argwhere(self.__TripleLinesArray.astype('int') == intTripleLineID)+np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion)
        else:
            warnings.warn(str(intTripleLineID) + ' is an invalid triple line ID')
    def GetTripleLineIDs(self)->list:
        return self.__TripleLineIDs
    def GetGrainBoundaryIDs(self)->list:
        return self.__GrainBoundaryIDs
    def GetGrainPoints(self, intGrainNumber = None):
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



class LAMMPSAnalysis(LAMMPSPostProcess):
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int, fltLatticeParameter: float):
        LAMMPSPostProcess.__init__(self, fltTimeStep,intNumberOfAtoms, intNumberOfColumns, lstColumnNames, lstBoundaryType, lstBounds,intLatticeType)
        self.__GrainBoundaries = dict()
        self.__UniqueGrainBoundaries = dict()
        self.__MergedTripleLines = dict()
        self.__UniqueTripleLines = dict() #periodically equivalent triple lines are merged into a single point
        self.__TripleLines = dict()
        self.__Grains = dict()
        self.__LatticeParameter = fltLatticeParameter
        self.__GrainLabels = []
        self.__QCTripleLinePoints = []
        self.__QCGBPoints = []
        self.__TripleLineIDs = []
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
    #def FindTripleLineEnergy(self, strTripleLineID: str, fltIncrement: float, fltWidth: float,fltMinimumLatticeValue = -3.3600000286, fltTolerance = 0.005, fltRadius = None, blnByVolume = False):
    def FindTripleLineEnergy(self, strTripleLineID: str, fltIncrement: float, fltWidth: float, fltTolerance, fltRadius = None):
        lstL = []
        lstV = []
        lstI = []
        fltEnergy = 0
    #    if blnByVolume: #hard coded for FCC at the minute
    #        fltMinimumLatticeValue = fltMinimumLatticeValue*4/(self.__LatticeParameter**3)
        arrDisplacements = self.DisplacementsToLattice(strTripleLineID, fltWidth, fltIncrement, fltTolerance)
        if len(arrDisplacements) == 3:
            arrCentre = gf.EquidistantPoint(*arrDisplacements)
        else:
            arrCentre = np.mean(arrDisplacements, axis = 0)
            warnings.warn("Error triple line " + str(strTripleLineID) + " has been tested in " + str(len(arrDisplacements)) + " direction(s).")
        self.__UniqueTripleLines[strTripleLineID].SetCentre(arrCentre) 
        self.MakeUniqueGrainBoundaries(self.__UniqueTripleLines[strTripleLineID].GetID())
        arrDisplacements = self.DisplacementsToLattice(strTripleLineID,fltWidth, fltIncrement, fltTolerance)
        arrCentre = gf.EquidistantPoint(*arrDisplacements)
        self.__UniqueTripleLines[strTripleLineID].SetCentre(arrCentre)
        if fltRadius is None:  #can pass a fixed radius as an argument and this is used instead
            fltRadius = np.linalg.norm(arrCentre-arrDisplacements[0])
        self.MakeUniqueGrainBoundaries(self.__UniqueTripleLines[strTripleLineID].GetID())
        self.__UniqueTripleLines[strTripleLineID].SetRadius(fltRadius)
        lstL, lstV, lstI = self.FindThreeGrainStrips(strTripleLineID,fltWidth, fltIncrement, strValue = 'mean')
        intRadius = np.round(fltRadius/fltIncrement,0).astype('int')
        try:
            popt = optimize.curve_fit(self.__Reciprocal, lstL[intRadius:],lstV[intRadius:])[0]
        except RuntimeError:
            warnings.warn("Optimisation error triple line " +  str(strTripleLineID) + " in FindTripleLineEnergy with intRadius = " + str(intRadius))
        self.GetUniqueTripleLines(strTripleLineID).SetFitParameters(popt)
        lstTJIDs = self.FindCylindricalAtoms(self.GetAtomData()[:,0:4],arrCentre,fltRadius,self.CellHeight)
        intTJAtoms = len(lstTJIDs)
        if intTJAtoms >0:
            self.__UniqueTripleLines[strTripleLineID].SetAtomIDs(lstTJIDs)            
            arrTJValues = self.GetAtomsByID(lstTJIDs)[:, self._intPE]
            fltEnergy = np.mean(arrTJValues)
            return fltEnergy, fltRadius, intTJAtoms
        else:
            return 0,0,0
    def DisplacementsToLattice(self, strTripleLineID: str, fltWidth: float, fltIncrement: float, fltTolerance: float)->np.array:
        arrCentre = self.GetUniqueTripleLines(strTripleLineID).GetCentre()
        fltLength = self.FindClosestTripleLineDistance(strTripleLineID)/2
        arrValues = self.FindValuesInCylinder(self.GetLatticeAtoms()[:,0:4],arrCentre,fltLength,self.CellHeight,self._intPE)
        fltMeanLatticeValue = np.mean(arrValues)
        fltVarLatticeValue = np.var(arrValues)
        arrVectors  = self.GBBisectingVectors(strTripleLineID)
        n = len(arrVectors)
        lstDisplacements = []
        for l in range(n):
            v = gf.NormaliseVector(arrVectors[np.mod(l,n)] + arrVectors[np.mod(l+1,n)])
            if np.dot(v,gf.NormaliseVector(arrVectors[np.mod(l+2,n)])) > 0: 
                v = -v
            arrWidth  = fltWidth*np.cross(v,np.array([0,0,1]))
            intEnd = np.floor(fltLength/fltIncrement).astype('int')
            fltDistance = self.InLattice(arrCentre,2*fltWidth*v,arrWidth,fltMeanLatticeValue,fltTolerance,fltIncrement, intEnd)
            lstDisplacements.append(arrCentre + fltDistance*v)
        return np.vstack(lstDisplacements)    
    def InLattice(self, arrStart: np.array, arrDirection: np.array, arrWidth: np.array,fltLatticeValue:float, fltTolerance:float, fltIncrement: float, intEnd: int)->float: #moves from a specified point in a direction
        blnInLattice = False
        intStart = 0
        v = gf.NormaliseVector(arrDirection)
        while not(blnInLattice) and intStart < intEnd:
                    intStart += 1
                    lstI = self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrStart+intStart*fltIncrement*v,arrDirection, 
                                                           arrWidth,np.array([0,0,self.CellHeight]))
                    if len(lstI) > 0:
                        arrPEValues = self.GetAtomsByID(lstI)[:,self._intPE]
                        fltTestValue = np.sum(list(map(lambda x: ((x-fltLatticeValue)/fltLatticeValue)**2, arrPEValues)))
                        fltCriticalValue = fltTolerance**2*len(arrPEValues)
                        if fltTestValue < fltCriticalValue:
                            blnInLattice = True
        return intStart*fltIncrement
    def GBBisectingVectors(self, strTripleLineID: str)->np.array:
        lstOfVectors = []
        for strGB in self.GetUniqueTripleLines(strTripleLineID).GetUniqueAdjacentGrainBoundaries():
            arrVector = self.__UniqueGrainBoundaries[strGB].GetVectorDirection(strTripleLineID, self.__LatticeParameter, bln3D=True) 
            lstOfVectors.append(gf.NormaliseVector(arrVector))
        return np.vstack(lstOfVectors)
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
    def FindTripleLines(self,fltGridLength: float, fltSearchRadius: float):
        fltMidHeight = self.CellHeight/2
        objQPoints = QuantisedRectangularPoints(self.GetDefectiveAtoms()[:,self._intPositionX:self._intPositionY+1],self.GetUnitBasisConversions()[0:2,0:2],20,fltGridLength)
        arrTripleLines = objQPoints.GetTriplePoints()   
        arrTripleLines[:,2] = fltMidHeight*np.ones(len(arrTripleLines))
        for i  in range(len(arrTripleLines)):
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
    def MergePeriodicTripleLines(self, fltRadius = None): #finds equivalent and adjacent triplelines and sets
        lstKeys = self.GetTripleLineIDs()
        counter = 0
        while len(lstKeys) > 0: 
            objTripleLine = self.__TripleLines[lstKeys.pop(0)] #take care not to set any properties on objTripleLine
            lstEquivalentTripleLines =  objTripleLine.GetEquivalentTripleLines()
            for lstID in lstEquivalentTripleLines:
                if lstID in lstKeys:
                    lstKeys.remove(lstID)
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
        lstUniqueKeys = self.GetUniqueTripleLineIDs()
        while len(lstUniqueKeys) > 0:
            objUniqueTripleLine1 = self.__UniqueTripleLines[lstUniqueKeys.pop(0)]
            setAdjacentTripleLines1 = set(objUniqueTripleLine1.GetAdjacentTripleLines())
            for strKey in self.GetUniqueTripleLineIDs():
                if strKey != objUniqueTripleLine1.GetID():
                    objUniqueTripleLine2 = self.__UniqueTripleLines[strKey]
                    lstEquivalentTripleLines2 = objUniqueTripleLine2.GetEquivalentTripleLines()
                    setIntersection = setAdjacentTripleLines1.intersection(lstEquivalentTripleLines2)
                    if len(setIntersection) > 0:
                        objUniqueTripleLine1.SetUniqueAdjacentTripleLines(objUniqueTripleLine2.GetID())
        lstSortedUniqueIDs = self.GetUniqueTripleLineIDs()
        arrUniqueTripleLines =np.zeros([len(lstSortedUniqueIDs),3])
        for strID in lstSortedUniqueIDs:
            if fltRadius is not(None):
                arrCentre = self.MoveTripleLine(self.GetUniqueTripleLines(strID).GetCentre(),fltRadius)
                self.GetUniqueTripleLines(strID).SetCentre(arrCentre)
            arrUniqueTripleLines[int(strID[3:])] = self.GetUniqueTripleLines(strID).GetCentre()
        self.__PeriodicTripleLineDistanceMatrix = self.MakePeriodicDistanceMatrix(arrUniqueTripleLines, arrUniqueTripleLines)
    def MakeGrainBoundaries(self):
        lstTJIDs = self.GetTripleLineIDs()
        counter = 0
        while len(lstTJIDs) > 0:
            strCurrentTJ = lstTJIDs.pop(0)
            lstAdjacentTripleLines = self.GetTripleLines(strCurrentTJ).GetAdjacentTripleLines()
            for j in lstAdjacentTripleLines:
                arrMovedTripleLine = self.PeriodicShiftCloser(self.GetTripleLines(strCurrentTJ).GetCentre(),self.GetTripleLines(j).GetCentre())
                arrLength = arrMovedTripleLine - self.GetTripleLines(strCurrentTJ).GetCentre()
                arrWidth = 25*np.cross(gf.NormaliseVector(arrLength), np.array([0,0,1]))
                arrPoints = self.FindValuesInBox(self.GetDefectiveAtoms()[:,0:4], 
                self.GetTripleLines(strCurrentTJ).GetCentre(),arrLength,arrWidth,self.GetCellVectors()[:,2],[1,2,3])
                arrPoints = self.PeriodicShiftAllCloser(self.GetTripleLines(strCurrentTJ).GetCentre(), arrPoints)
                lstGBID = [strCurrentTJ ,j]
                if int(strCurrentTJ[2:]) > int(j[2:]): #this overwrites the same grainboundary and sorts ID with lowest TJ number first
                    lstGBID.reverse()
                strGBID = str(lstGBID[0]) + ',' + str(lstGBID[1])
                if strGBID not in list(self.__GrainBoundaries.keys()):
                    objGrainBoundary = gl.GrainBoundaryCurve(arrMovedTripleLine,self.GetTripleLines(strCurrentTJ).GetCentre(), lstGBID, arrPoints, 
                    self.CellHeight/2)
                    self.__GrainBoundaries[strGBID] = objGrainBoundary
                    self.__TripleLines[strCurrentTJ].SetAdjacentGrainBoundaries(strGBID)
                    self.__TripleLines[j].SetAdjacentGrainBoundaries(strGBID)
               # else: Debug information only
               #     print('Duplicate GB ' + str(lstGBID))
                counter += 1
        self.MakeUniqueGrainBoundaries()
    def MakeUniqueGrainBoundaries(self, lstTripleLineID = None):
        if lstTripleLineID is None:
            lstUTJIDs = self.GetUniqueTripleLineIDs()
        elif isinstance(lstTripleLineID, str):
            lstUTJIDs = [lstTripleLineID]
        elif isinstance(lstTripleLineID, list):
            lstUTJIDs = lstTripleLineID
        else:
            warnings.warn("Invalid triple line IDs")
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
            arrNextPoint = arrPoint
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
    def GetGrainBoundaryIDS(self):
        return sorted(list(self.__GrainBoundaries.keys()), key=lambda x: int(x.split(',')[0][2:]))
    def GetTripleLineIDs(self):
        return sorted(list(self.__TripleLines.keys()),key = lambda x: int(x[2:]))
    def GetTripleLines(self, strID = None)->gl.TripleLine:
        if strID is None:
            return self.__TripleLines
        else:
            return self.__TripleLines[strID]
    def FindGBStrip(self, intGrainBoundaryNumber: int, fltProportion: float,  fltLength: float,fltWidth: float, fltIncrement:float, strValue = 'sum', blnAccumulative = True):
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
            if blnAccumulative:
                lstIndices.extend(self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrCentre,l*arrGBDirection, 
                                                           fltWidth*arrCrossVector,np.array([0,0,self.CellHeight]))
                                                           )
            else:
                lstIndices = self.FindBoxAtoms(self.GetAtomData()[:,0:4],arrCentre + l*arrGBDirection,fltWidth*arrGBDirection, fltWidth*arrCrossVector,np.array([0,0,self.CellHeight]))                                                                                             
            lstIndices = list(np.unique(lstIndices))
            if strValue == 'mean':
                lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
            elif strValue =='sum':
                lstValues.append(np.sum(self.GetAtomsByID(lstIndices)[:,self._intPE],axis=0))
        return lstLength, lstValues,lstIndices
    def FindClosestTripleLineDistance(self, strTripleLineID: str)->float:
        if strTripleLineID[:1] == 'U':
            intTripleLine = int(strTripleLineID[3:])
        elif strTripleLineID[:1] == 'T':
            intTripleLine = int(strTripleLineID[2:])
        fltDistance = np.sort(self.__PeriodicTripleLineDistanceMatrix[intTripleLine])[1]
        return fltDistance
    def FindStrip(self, arrStart: np.array, arrVector: np.array,fltWidth: float, fltIncrement: float, fltLength: float,  blnAccumulative = True):
        lstLength = []
        lstIndices  = []
        lstI = []
        lstValues = []
        intMax = np.floor(fltLength/(fltIncrement)).astype('int')
        intMin = 1
        v = gf.NormaliseVector(arrVector)
        arrWidth  = fltWidth*np.cross(v,np.array([0,0,1]))
        for j in range(intMin,intMax+1):
            l = fltIncrement*j
            lstLength.append(l)
            
            if blnAccumulative:
                lstI = self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrStart,l*v, 
                                                           arrWidth,np.array([0,0,self.CellHeight]))
            else:
                lstI = self.FindBoxAtoms(self.GetAtomData()[:,0:4],
                                                           arrStart+l*v,fltWidth*v, 
                                                           arrWidth,np.array([0,0,self.CellHeight]))
            if len(lstI) >0:
                lstIndices.extend(lstI)
                lstIndices = list(np.unique(lstIndices))
                if blnAccumulative:
                    lstValues.append(np.mean(self.GetAtomsByID(lstIndices)[:,self._intPE]))
                else:
                    lstValues.append(np.mean(self.GetAtomsByID(lstI)[:,self._intPE],axis=0))
            else:
                lstValues.append(0)
        return lstLength, lstValues, lstIndices   
    def FindThreeGrainStrips(self, strTripleLineID: int,fltWidth: float, fltIncrement: float, strValue = 'mean',fltLength = None):
        lstOfVectors = [] #unit vectors that bisect the grain boundary directions
        lstValues = []
        lstRadii = []
        lstIndices  = []
        lstI = []
        lstVolume = []
        lstN = []
        if fltLength is None:
           # fltClosest = np.sort(self.__TripleLineDistanceMatrix[intTripleLine,list(setTripleLines)])[1]/2
           fltClosest = self.FindClosestTripleLineDistance(strTripleLineID)/2
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
        return lstRadii, lstValues, lstIndices  
    def FindHerringVector(self, strTripleLineID, fltLength = None, blnRemoveTripleLines = False)->np.array:
        if fltLength is None:
            fltLength = 3*self.GetUniqueTripleLines(strTripleLineID).GetRadius()
        arrGBVectors = np.zeros([3,3])
        lstGBIDs = []
        lstTJIDs = self.GetUniqueTripleLines(strTripleLineID).GetAtomIDs()
        fltTJMeanValue = np.sum(self.GetAtomsByID(lstTJIDs)[:,self._intPE])
        objTripleLine = self.GetUniqueTripleLines(strTripleLineID)
        for j, strGB in enumerate(objTripleLine.GetUniqueAdjacentGrainBoundaries()):
            objGB = self.GetUniqueGrainBoundaries(strGB)
            arrGBVectors[j,0:2] = objGB.GetVectorDirection(strTripleLineID, fltLength,False)
            setGBIDs = set(self.FindGBAtoms(strGB, 2*objTripleLine.GetRadius(),fltLength, False)[0])
            if blnRemoveTripleLines:
                lstGBIDs = list(setGBIDs.difference(lstTJIDs))
            else: 
                lstGBIDs = list(setGBIDs)
            fltValue = (np.sum(self.GetAtomsByID(lstGBIDs)[:,self._intPE])-fltTJMeanValue)/self.CellHeight
            arrGBVectors[j,0:2] = fltValue*gf.NormaliseVector(arrGBVectors[j,0:2])
        arrHerring = np.sum(arrGBVectors, axis = 0) 
        return gf.NormaliseVector(arrHerring), np.linalg.norm(arrHerring)
    def FindClosestTripleLineID(self, inPoint: np.array)->str:
        blnFirstTime = True
        for j in self.GetUniqueTripleLineIDs():
            fltNewDistance = np.linalg.norm(self.GetUniqueTripleLines(j).GetCentre()-inPoint)
            if blnFirstTime:
                strID = j
                fltMinDistance = fltNewDistance
                blnFirstTime = False   
            elif fltNewDistance < fltMinDistance:
                fltMinDistance = fltNewDistance
                strID = j
        return strID
 

class QuantisedRectangularPoints(object): #linear transform parallelograms into a rectangular parameter space
    def __init__(self, in2DPoints: np.array, inUnitBasisVectors: np.array, n: int, fltGridSize: float, blnDebug = False):
        self.__GrainValue = 0
        self.__GBValue = 1 #just fixed constants used in the array 
        self.__DislocationValue = 2
        self.__TripleLineValue = 3
        self.__WrapperWidth = n #mininum count specifies how many nonlattice atoms occur in the float grid size before it is counted as a grain boundary grid or triple line
        self.__BasisVectors = inUnitBasisVectors
        self.__InverseMatrix =  np.linalg.inv(inUnitBasisVectors)
        self.__GridSize = fltGridSize
        self.__WrapperVector = np.array([n,n])
        arrPoints =  np.matmul(in2DPoints, self.__BasisVectors)*(1/fltGridSize)
        self.__intMaxHeight = np.round(np.max(arrPoints[:,0])).astype('int')+1
        self.__intMaxWidth = np.round(np.max(arrPoints[:,1])).astype('int')+1
        self.__OriginalGrid =  np.zeros([(self.__intMaxHeight),self.__intMaxWidth])
        self.__ExtendedArrayGrid = np.zeros([np.shape(self.__OriginalGrid)[0]+2*n,np.shape(self.__OriginalGrid)[1]+2*n])
        arrPoints = np.round(arrPoints).astype('int')
        for j in arrPoints:
            self.__OriginalGrid[j[0],j[1]] += 1 #this array represents the simultion cell
        self.FillInGrainBoundaryGaps()
        self.MakeExtendedGrid()
        self.__TriplePoints = []
        self.__Dislocations = []
        self.__GrainBoundaryLabels = []
        self.__GrainBoundaryIDs = []
        self.__blnGrainBoundaries = False #this flag is set once FindGrainBoundaries() is called 
        if not(blnDebug):
            self.FindTriplePoints() #when debugging comment these out and call the methods one at a time in this order
            self.FindGrainBoundaries()
            self.MakeAdjacencyMatrix()
    def FillInGrainBoundaryGaps(self,fltSigma = 0.5):
        self.__fltSigma = fltSigma
        arrHeightValues = np.array(list(range(self.__intMaxHeight)))
        arrWidthValues = np.array(list(range(self.__intMaxWidth)))
        objFunction = RectBivariateSpline(arrHeightValues ,arrWidthValues,self.__OriginalGrid)
        self.__ArrayGrid = objFunction(arrHeightValues,arrWidthValues)
        self.__ArrayGrid = gaussian(self.__ArrayGrid, fltSigma)
        self.__ArrayGrid = (self.__ArrayGrid >= np.mean(self.__ArrayGrid)).astype('int')
        self.__ArrayGrid = remove_small_holes(self.__ArrayGrid.astype('bool'), 4).astype('int')
    def MakeExtendedGrid(self):
        n = self.__WrapperWidth
        self.__ExtendedArrayGrid[n:-n, n:-n] = self.__ArrayGrid
        self.__ExtendedArrayGrid[0:n, n:-n] = self.__ArrayGrid[-n:,:]
        self.__ExtendedArrayGrid[-n:, n:-n] = self.__ArrayGrid[:n,:]
        self.__ExtendedArrayGrid[:,0:n] = self.__ExtendedArrayGrid[:,-2*n:-n]
        self.__ExtendedArrayGrid[:,-n:] = self.__ExtendedArrayGrid[:,n:2*n]
        self.__ExtendedArrayGrid = (self.__ExtendedArrayGrid.astype('bool')).astype('int')
        self.__ExtendedSkeletonGrid = skeletonize(self.__ExtendedArrayGrid).astype('int')
    def GetOriginalGrid(self):
        return self.__OriginalGrid
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
                    if self.CheckEachSide(arrCurrent):
                        self.SetSkeletonValue(x,self.__TripleLineValue)
                elif intSwaps ==6:
                    self.SetSkeletonValue(x,self.__TripleLineValue)
                elif intSwaps < 4 and blnFlagEndPoints and m==3: 
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
    def ClearWrapper(self, blnUsingTripleLines = True): 
        k = self.__WrapperWidth
        self.__ExtendedSkeletonGrid[:k, :] = self.__GrainValue
        self.__ExtendedSkeletonGrid[k:, :] = self.__GrainValue
        self.__ExtendedSkeletonGrid[:, :k] = self.__GrainValue
        self.__ExtendedSkeletonGrid[:, k:] = self.__GrainValue
    def FindTriplePoints(self):
        self.__TriplePoints = self.ClassifyGBPoints(3, True)
        blnBrokenGB = False
        intCounter = 0
        while not(blnBrokenGB) and intCounter < len(self.__Dislocations):
            if self.InGrid(self.__Dislocations[intCounter]):
                blnBrokenGB = True
            else:
                intCounter += 1        
        if blnBrokenGB and self.__fltSigma < 5:
            self.FillInGrainBoundaryGaps(self.__fltSigma+0.25)
            self.MakeExtendedGrid()
            self.FindTriplePoints()
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
        for j in lstStartPoints: #normally 3 points but maybe one or two
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
