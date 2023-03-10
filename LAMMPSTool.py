import re
#from types import NoneType
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
from datetime import datetime
import copy
import warnings
from functools import reduce
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import KMeans
import MiscFunctions as mf
import matplotlib.pyplot as plt
import itertools as it

class LAMMPSLog(object):
    def __init__(self,strFilename):
        self.__FileName = strFilename
        self.__Values = dict()
        self.__ColumnNames = dict()
        self.__Stages = 0
        dctValues = dict()
        dctColumns = dict()
        intStages = 0
        with open(strFilename) as Dfile:
            while True:
                lstBounds = []
                try:
                    line = next(Dfile).strip()
                except StopIteration as EndOfFile:
                    break
                if "Step" == line[:4]:
                    lstColumnNames = line.split(' ')
                    lstRows = []
                    line = next(Dfile).strip()
                    while "Loop" != line[:4]:
                        lstRow = list(map(float,line.strip().split()))
                        lstRows.append(lstRow)
                        line = next(Dfile).strip()
                    arrValues = np.vstack(lstRows)
                    dctValues[intStages] = arrValues
                    dctColumns[intStages] = lstColumnNames
                    intStages += 1    
            self.__Values = dctValues 
            self.__ColumnNames = dctColumns
            self.__Stages = intStages
            Dfile.close()
    def GetNumberOfStages(self):
            return self.__Stages
    def GetValues(self, intStage):
            return self.__Values[intStage]
    def GetColumnNames(self, intStage):
            return self.__ColumnNames[intStage]    

class LAMMPSData(object):
    def __init__(self,strFilename: str, intLatticeType: int, fltLatticeParameter: float, objAnalysis: object):
        self.__dctTimeSteps = dict()
        self.__FileName = strFilename
        lstNumberOfAtoms = []
        lstTimeSteps = []
        lstColumnNames = []
        lstBoundaryType = []
        self.__Dimensions = 3 # assume 3d unless file shows the problem is 2d
        intRow = 0
        with open(strFilename) as Dfile:
            while True:
                lstBounds = []
                try:
                    line = next(Dfile).strip()
                    intRow +=1
                except StopIteration as EndOfFile:
                    break
                if "ITEM: TIMESTEP" != line:
                    raise Exception("Unexpected "+repr(line))
                timestep = int(next(Dfile).strip())
                intRow +=1
                lstTimeSteps.append(timestep)
                line = next(Dfile).strip()
                intRow +=1
                if "ITEM: NUMBER OF ATOMS" != line:
                    raise Exception("Unexpected "+repr(line))
                N = int(next(Dfile).strip())
                intRow +=1
                lstNumberOfAtoms.append(N)
                line = next(Dfile).strip()
                intRow +=1
                if "ITEM: BOX BOUNDS" != line[0:16]:
                    raise Exception("Unexpected "+repr(line))
                lstBoundaryType = line[17:].strip().split()
                lstBounds.append(list(map(float, next(Dfile).strip().split())))
                lstBounds.append(list(map(float, next(Dfile).strip().split())))
                intRow +=2
                if len(lstBoundaryType)%3 == 0:
                    lstBounds.append(list(map(float, next(Dfile).strip().split())))
                    intRow +=1
                else:
                    self.__Dimensions = 2
                line = next(Dfile).strip()
                intRow +=1
                if "ITEM: ATOMS id" != line[0:14]:
                    raise Exception("Unexpected "+repr(line))
                lstColumnNames = line[11:].strip().split()
                intNumberOfColumns = len(lstColumnNames)
                objTimeStep = objAnalysis(timestep, N,intNumberOfColumns,lstColumnNames, lstBoundaryType, lstBounds,intLatticeType, fltLatticeParameter)
                objTimeStep.SetColumnNames(lstColumnNames)
                #arrValues = np.loadtxt(strFilename, skiprows =intRow , max_rows=N)
                #objTimeStep.SetAtomData(arrValues)
                intRow += N
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
                objTimeStep.SetFileName(strFilename)
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
        self.__PeroidicDirections = []
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
        arrColumn = np.zeros(self.GetNumberOfAtoms())
        intColumnIndex = self.GetColumnIndex(strColumnName)
        self.__AtomData[:,intColumnIndex] = arrColumn
    def GetRow(self,intRowNumber: int):
        return self.__AtomData[intRowNumber]
    def GetRows(self, lstOfRows: list):
        return self.__AtomData[lstOfRows,:]
    def GetAtomsByID(self, lstOfAtomIDs: list, intAtomColumn = 0):
        return self.__AtomData[np.isin(self.__AtomData[:,intAtomColumn],lstOfAtomIDs)]
    def SetAtomData(self, inArray:np.array):
        self.__AtomData= inArray
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
        lstPeriodicDirections= []
        for k in self.__BoundaryTypes:
            if k =='pp':
                lstPeriodicDirections.append('p')
            else:
                lstPeriodicDirections.append('n')
        self.__PeroidicDirections = lstPeriodicDirections
    def GetPeriodicDirections(self):
        return self.__PeroidicDirections
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
        arrCellVectors = np.round(arrCellVectors, 10)
        arrNonZero = np.argwhere(arrCellVectors != 0.0)
        if len(arrNonZero) == 3: #if there is no tilt then each coordinate direction is of the form [a 0 0], [0 b 0] and [0 0 c]
            self.__blnCuboid = True
        else: 
            self.__blnCuboid = False
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
    def WrapVectorIntoSimulationBox(self, inVector: np.array)->np.array:
        return gf.WrapVectorIntoSimulationCell(self.__CellBasis, inVector)
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
    def SetFileName(self, strFilename):
        self.__strFilename = strFilename
    def WriteDumpFile(self, strFilename =None):
        if strFilename is None:
            strFilename = self.__strFilename
        strHeader = 'ITEM: TIMESTEP \n'
        strHeader += str(self.GetTimeStep()) + '\n'
        strHeader += 'ITEM: NUMBER OF ATOMS \n'
        strHeader += str(self.GetNumberOfAtoms()) + '\n'
        if self.__blnCuboid:
            strHeader += 'ITEM: BOX BOUNDS ' + ' '.join(self.__BoundaryTypes) + '\n'
            for j in range(3):
                strHeader += str(self.__BoundBoxDimensions[j,0]) + ' ' + str(self.__BoundBoxDimensions[j,1]) + '\n'
        else:
            strHeader += 'ITEM: BOX BOUNDS xy xz yz ' + ' '.join(self.__BoundaryTypes) + '\n'
            for j in range(3):
                strHeader += str(self.__BoundBoxDimensions[j,0]) + ' ' + str(self.__BoundBoxDimensions[j,1]) + ' '  + str(self.__BoundBoxDimensions[j,2]) + '\n'
        strHeader += 'ITEM: ATOMS ' + ' '.join(self.__ColumnNames)
        np.savetxt(strFilename, self.GetAtomData(), fmt= ' '.join(self.__ColumnTypes), header=strHeader, comments='')
    def WriteDataFile(self, strFilename: str, blnIncludeVelocities = False):
        now = datetime.now()
        strDateTime = now.strftime("%d/%m/%Y %H:%M:%S")
        strHeader = '##' + strDateTime +  '\n'
        strHeader += str(self.GetNumberOfAtoms()) + ' atoms \n'
        strHeader += '1' + ' atom types \n'
        lstNames = ['x','y','z']
        if self.__blnCuboid:
            for j in range(3):
                strHeader += str(self.__BoundBoxDimensions[j,0]) + ' ' + str(self.__BoundBoxDimensions[j,1]) + ' '  + str(lstNames[j]) +  'lo ' + str(lstNames[j]) + 'hi \n'
        else:
            #for j in range(3):
            strHeader += str(self.__BoundBoxDimensions[0,0] - min([0,self.__BoundBoxDimensions[0,-1], self.__BoundBoxDimensions[1,1],self.__BoundBoxDimensions[0,-1]+ self.__BoundBoxDimensions[1,-1]])) + ' ' + str(self.__BoundBoxDimensions[0,1] - max([0,self.__BoundBoxDimensions[0,-1], self.__BoundBoxDimensions[1,-1],self.__BoundBoxDimensions[0,-1]+ self.__BoundBoxDimensions[1,-1]])) + ' xlo xhi \n'
            strHeader += str(self.__BoundBoxDimensions[1,0] - min([0,self.__BoundBoxDimensions[2,-1]])) + ' ' + str(self.__BoundBoxDimensions[1,1] - max([0,self.__BoundBoxDimensions[2,-1]])) + ' ylo yhi \n'
            strHeader += str(self.__BoundBoxDimensions[2,0]) +  ' ' + str(self.__BoundBoxDimensions[2,1]) + ' zlo zhi \n' 

                # strHeader += str(self.__BoundBoxDimensions[j,0]) + ' ' + str(self.__BoundBoxDimensions[j,1]) + ' '  + str(lstNames[j]) +  'lo ' + str#(lstNames[j]) + 'hi \n'
            strHeader += str(self.__BoundBoxDimensions[0,-1]) + ' ' + str(self.__BoundBoxDimensions[1,-1]) + ' ' + str(self.__BoundBoxDimensions[2,-1]) + ' xy xz yz \n'
        strHeader += '\nAtoms \n'
        intCols = 5 
        strFormat = "%d " "%d " "%.6f " "%.6f " "%.6f " 
        arrValues = np.ones([self.GetNumberOfAtoms(),intCols]) ##currently hard coded to atom type 1
        arrValues[:,0] = self.GetAtomData()[:,0]
        arrValues[:,2:intCols] = self.GetAtomData()[:,1:intCols-1]
        np.savetxt(strFilename, arrValues, fmt = strFormat, header=strHeader, comments='')
        if blnIncludeVelocities:
            strHeader = '\nVelocities \n '
            arrValues = self.GetAtomData()[:,[0,5,6,7]]
            f = open(strFilename,'a')
            np.savetxt(f,arrValues, fmt = "%d " "%.6f " "%.6f " "%.6f ", header=strHeader,comments='')
            f.close()

class LAMMPSPostProcess(LAMMPSTimeStep):
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int):
        LAMMPSTimeStep.__init__(self,fltTimeStep,intNumberOfAtoms, lstColumnNames, lstBoundaryType, lstBounds)
        self.__Dimensions = self.GetNumberOfDimensions()
        self._LatticeStructure = intLatticeType #lattice structure type as defined by OVITOS
        if 'StructureType' in self.GetColumnNames():
            self._intStructureType = int(self.GetColumnNames().index('StructureType'))
        elif 'c_pt[1]' in self.GetColumnNames():
            self._intStructureType = int(self.GetColumnNames().index('c_pt[1]'))
        else:
            warnings.warn('Error missing atom structure types in dump file.')
        if 'c_v[1]' in self.GetColumnNames():
            self._intVolume = int(self.GetColumnNames().index('c_v[1]'))
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
        if 'StructureType' in self.GetColumnNames() or 'c_pt[1]' in self.GetColumnNames():
            self.__DefectiveAtomIDs = []
            self.__NonDefectiveAtomIDs = []
            lstOtherAtoms = list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == 0)[0])
            lstPTMAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') == self._LatticeStructure)[0])
            lstNonPTMAtoms =  list(np.where(self.GetColumnByIndex(self._intStructureType).astype('int') != self._LatticeStructure)[0])
            self.__PTMAtomIDs = list(self.GetAtomData()[lstPTMAtoms,0].astype('int'))
            self.__NonPTMAtomIDs = list(self.GetAtomData()[lstNonPTMAtoms,0].astype('int'))
            self.__OtherAtomIDs = list(self.GetAtomData()[lstOtherAtoms,0].astype('int'))
            self.FindDefectiveAtoms(fltTolerance)
    def GetGrainPTMAtomIDs(self, fltTolerance = 0.05):
        arrPE = self.GetPTMAtoms()[:,self.GetColumnIndex('c_pe1')]
        mu,st = stats.norm.fit(arrPE)
        tupValues = stats.norm.interval(alpha=1-fltTolerance, loc=mu, scale=st)
        arrRows = np.where((tupValues[0] < arrPE) & (arrPE < tupValues[1]))[0]
        arrV = self.GetPTMAtoms()[:,self.GetColumnIndex('c_')]
        return np.array(self.__PTMAtomIDs)[arrRows]
    def GetLatticeAtomIDs(self):
        return self.__LatticeAtomIDs
    def GetNonLatticeAtomIDs(self):
        return self.__NonLatticeAtomIDs  
    def FindDefectiveAtoms(self, fltTolerance = None):
        if fltTolerance is None:
            if self.GetNumberOfNonPTMAtoms() > 0:
                fltThreshold = np.mean(self.GetNonPTMAtoms()[:,self.GetColumnIndex('c_pe1')])
            else:
                fltThreshold = np.mean(self.GetPTMAtoms()[:,self.GetColumnIndex('c_pe1')])
        else:
            fltThreshold = fltTolerance
        lstRowDefectiveAtoms = np.where(self.GetPTMAtoms()[:,self._intPE] > fltThreshold)[0]
        lstDefectivePTMIDs = list(self.GetAtomData()[lstRowDefectiveAtoms,0])
        self.__NonLatticeAtomIDs = list(set(lstDefectivePTMIDs) | set(self.__NonPTMAtomIDs))
        setAllLatticeAtomIDs = set(list(self.GetAtomData()[:,0]))
        self.__LatticeAtomIDs = list(setAllLatticeAtomIDs.difference(self.__NonLatticeAtomIDs))
        self.__DefectiveAtomIDs =lstDefectivePTMIDs
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
        self.__ExteriorGrains = dict() #include only the interior atoms of each grain
        self.__PeriodicGrains = dict()
        self.__PeriodicGrainBoundaries = dict()
        self.__PeriodicJunctionLines = dict()
        self.__LatticeParameter = fltLatticeParameter
        self.__GrainLabels = []
        self.__GrainBoundaryLabels = []
        self.__JunctionLinesLabels = []
        self.__JunctionLineIDs = []
        self.__GrainBoundaryIDs = []
        self.__DuplicateGBIDs = []
        self.__JunctionMesh = []
        self.__GrainBoundaryMesh = []
        self.__GBSeparation = 0 
        self.blnPEAssigned = False
        self.blnVolumeAssigned = False
        self.blnAdjustedMeshPointsAssigned = False
        self.intGrainNumber = -1 #set to a dummy value to check 
        self.intGrainBoundary = -1
        self.__objRealCell = gl.RealCell(ld.GetCellNodes(str(intLatticeType)),fltLatticeParameter*np.ones(3))
        self.__MaxGBWidth = 0
    def GetRealCell(self):
        return self.__objRealCell
    def GetGrainAtomIDsByEcoOrient(self, strColumnName, intValue):
        arrRow = self.GetColumnByName(strColumnName)
        arrPositions = np.where(arrRow == intValue)
        arrIDs = self.GetAtomData()[arrPositions,0]
        return arrIDs[0]
    def GetUnassignedGrainAtomIDs(self):
        if 'GrainNumber' not in self.GetColumnNames():
            warnings.warn('Grain labels not set.')
        else: 
            lstUnassignedAtoms = np.where(self.GetColumnByName('GrainNumber') ==-1)[0]
        return list(self.GetAtomData()[lstUnassignedAtoms,0].astype('int'))
    def SetLatticeParameter(self, fltParameter: float):
        self.__LatticeParameter = fltParameter
    def FindGrainAtomIDs(self, intN = 1, arrIDs = None):
        if arrIDs is None:
            arrIDs = np.array(self.GetPTMAtomIDs())
        arrGrainAtoms = self.GetAtomsByID(arrIDs)[:,1:4] 
        arrUsedRows = np.array(list(range(len(arrIDs))))
        for i in range(intN):
            objGrainTree = gf.PeriodicWrapperKDTree(arrGrainAtoms,self.GetCellVectors(),gf.FindConstraintsFromBasisVectors(self.GetCellVectors()),2*self.__LatticeParameter,self.GetPeriodicDirections())
            arrDistances1,arrIndices1 =objGrainTree.Pquery(arrGrainAtoms,k = self.__objRealCell.GetNumberOfNeighbours()+1)
            arrIndices1 = mf.FlattenList(arrIndices1)
            arrIndices1= objGrainTree.GetPeriodicIndices(arrIndices1)
            arrTrueDistances1 = arrDistances1[:,1:] 
            arrAllDistances = arrTrueDistances1.ravel()
            fltNearest = np.median(arrAllDistances)
            fltMin = np.min(arrAllDistances)
            arrRows1 = np.where(np.all(arrTrueDistances1 < 2*fltNearest -fltMin,axis =1))[0]
            arrTrueDistances2 = arrTrueDistances1[arrRows1]
            arrUsedRows = arrUsedRows[arrRows1]
            arrGrainAtoms = arrGrainAtoms[arrRows1]
        if len(arrUsedRows) > 0:
            arrIDs = arrIDs[arrUsedRows]
        else:
            arrIDs = []
        self.__GBSeparation =  fltNearest
        return arrIDs        
       
    def MatchPreviousGrainNumbers(self, lstGrainNumbers: list, lstGrainIDs: list):
        arrMatrix = np.zeros([len(self.__GrainLabels)-1, len(lstGrainNumbers)])
        for i in range(len(lstGrainIDs)):
            lstIDs = self.GetGrainAtomIDs(i+1)
            for j in range(len(lstGrainIDs)):
                arrMatrix[i,j] = len(set(lstGrainIDs[j].tolist()).intersection(lstIDs.tolist()))
        lstMatched = []
        for j in arrMatrix:
            lstMatched.append(lstGrainNumbers[np.argmax(j)])
        return lstMatched
    def RelabelGrainNumbers(self,lstCurrentLabels: list,lstNewLabels: list):
        lstAllIDs = []
        for i in lstCurrentLabels:
            lstAllIDs.append(self.GetGrainAtomIDs(i))
        for k in range(len(lstNewLabels)):
            self.SetColumnByIDs(lstAllIDs[k], self.GetColumnByName('GrainNumber'),lstNewLabels[k]*np.ones(len(lstAllIDs[k])))
    def FindPEPerVolume(self, lstIDs=None):
        if lstIDs is None:
            return self.GetColumnByName('c_pe1')/self.GetColumnByName('c_v[1]')
        else:
            intPEColumn = self.GetColumnIndex('c_pe1')
            intVColumn = self.GetColumnIndex('c_v[1]')
            arrValues = self.GetAtomsByID(lstIDs)[:, [intPEColumn, intVColumn]]
            return arrValues[:,0]/arrValues[:,1]
    def ResetGrainNumbers(self):
        if 'GrainNumber' not in self.GetColumnNames():
                self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]),'GrainNumber',strFormat='%i')
        else:
            intCol = self.GetColumnIndex('GrainNumber')
            self.SetColumnByIndex(np.zeros(self.GetNumberOfAtoms()),intCol)
    def PartitionGrains(self, intN: int,intMinGrainSize = 25, fltWrapperWidth = 25):
        arrIDs = self.FindGrainAtomIDs(intN)
        if len(arrIDs) > 0:
            arrPoints = self.GetAtomsByID(arrIDs)[:,1:4]
            clustering = DBSCAN(eps=1.05*self.__GBSeparation).fit(arrPoints)
            arrLabels = clustering.labels_
            arrUniqueLabels,arrCounts = np.unique(arrLabels,return_counts=True)
            if 'GrainNumber' not in self.GetColumnNames():
                self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]),'GrainNumber',strFormat='%i')
            for i in range(len(arrUniqueLabels)):
                j = arrCounts[i]
                k = arrUniqueLabels[i]
                if k != -1 and j >= intMinGrainSize:
                    arrCurrentIDs =  arrIDs[arrLabels==k]
                    intMax = np.max(self.GetColumnByName('GrainNumber'))
                    self.SetColumnByIDs(arrCurrentIDs,self.GetColumnIndex('GrainNumber'),(intMax+1)*np.ones(len(arrCurrentIDs)))
            self.__GrainLabels = self.GetGrainLabels()
            for k in self.__GrainLabels:
                self.__PeriodicGrains[k] = gf.PeriodicWrapperKDTree(self.GetAtomsByID(self.GetGrainAtomIDs(k))[:,1:4],self.GetCellVectors(),gf.FindConstraintsFromBasisVectors(self.GetCellVectors()),fltWrapperWidth,self.GetPeriodicDirections())
        else:
            self.__GrainLabels = []
    def SetPeriodicGrain(self, strName: str, arrIDs: np.array, fltWrapperWidth: float):
        self.__PeriodicGrains[strName] = gf.PeriodicWrapperKDTree(self.GetAtomsByID(arrIDs)[:,1:4],self.GetCellVectors(),gf.FindConstraintsFromBasisVectors(self.GetCellVectors()),fltWrapperWidth,self.GetPeriodicDirections())
        self.__GrainLabels = np.unique(self.__GrainLabels.append(strName)).tolist()  
    def MergePeriodicGrains(self, intCloseAtoms = 5):
        i = 0
        lstKeys = self.GetGrainLabels()
        lstKeys.remove(0)
        arrConstraints = gf.FindConstraintsFromBasisVectors(self.GetCellVectors())
        while i < len(lstKeys):
            blnMerge = False
            j = i+1
            c = 0
            lstAllRows = [] 
            arrOriginalPoints = self.__PeriodicGrains[lstKeys[i]].GetOriginalPoints()
            for c in range(len(arrConstraints)):
                arrVector = arrConstraints[c,:-1]
                fltValue = arrConstraints[c,-1]  
                fltDots = np.matmul(arrVector, np.transpose(arrOriginalPoints))
                arrRows1 = np.where(np.abs(fltDots) < 1.05*self.__GBSeparation)[0]
                if len(arrRows1) > 0: 
                    lstAllRows.extend(arrRows1)
                arrRows2 = np.where(np.abs(fltDots-fltValue) < 1.05*self.__GBSeparation)[0]
                if len(arrRows2) > 0:
                    lstAllRows.extend(arrRows2)
            if len(lstAllRows) > 0 :
                arrRows = np.unique(lstAllRows)
                blnMerge = True
                arrBoundaryPoints = arrOriginalPoints[arrRows]
            while j < len(lstKeys) and blnMerge:
                #arrIndices, arrDistances = self.__PeriodicGrains[lstKeys[i]].Pquery_radius(self.__PeriodicGrains[lstKeys[j]].GetExtendedPoints(),self.__GBSeparation)
                arrIndices, arrDistances = self.__PeriodicGrains[lstKeys[j]].Pquery_radius(arrBoundaryPoints,1.05*self.__GBSeparation)
                arrIndices = mf.FlattenList(arrIndices)
                intLength = 0
                if len(arrIndices) > 0:
                    arrIndices = self.__PeriodicGrains[lstKeys[j]].GetPeriodicIndices(arrIndices)
                    arrIndices = np.unique(arrIndices)
                    intLength = len(arrIndices)
                if intLength > intCloseAtoms:
                #if np.max(np.unique(lstLengths) > 4):
                    self.AppendGrainNumbers(lstKeys[i]*np.ones(len(self.GetGrainAtomIDs(lstKeys[j]))),self.GetGrainAtomIDs(lstKeys[j]))
                    arrNewPoints = np.append(self.GetAtomsByID(self.GetGrainAtomIDs(lstKeys[i]))[:,1:4],self.GetAtomsByID(self.GetGrainAtomIDs(lstKeys[j]))[:,1:4], axis=0)
                    self.__PeriodicGrains[lstKeys[i]] = gf.PeriodicWrapperKDTree(arrNewPoints,self.GetCellVectors(),gf.FindConstraintsFromBasisVectors(self.GetCellVectors()),self.__PeriodicGrains[lstKeys[i]].GetWrapperLength(),self.GetPeriodicDirections())
                    #self.__PeriodicGrains.pop(lstKeys[j])
                    del self.__PeriodicGrains[lstKeys[j]] 
                    lstKeys = self.GetGrainLabels()
                    if 0 in lstKeys:
                        lstKeys.remove(0)
                else:    
                    j +=1
            i +=1
        lstGrainLabels = self.GetGrainLabels()
        if 0 in lstGrainLabels:
            lstGrainLabels.remove(0)
        n = 1
        for l in lstGrainLabels:
            if l != n:
                self.AppendGrainNumbers(n*np.ones(len(self.GetGrainAtomIDs(l))),self.GetGrainAtomIDs(l))
                self.__PeriodicGrains[n] = gf.PeriodicWrapperKDTree(self.__PeriodicGrains[l].GetOriginalPoints(),self.GetCellVectors(),gf.FindConstraintsFromBasisVectors(self.GetCellVectors()),self.__PeriodicGrains[l].GetWrapperLength(),self.GetPeriodicDirections())
                self.__PeriodicGrains.pop(l)
            n +=1 
        self.__GrainLabels = self.GetGrainLabels()
    def GetAllGrainAtomIDs(self):
        lstAllGrainIDs = []
        for j in self.__GrainLabels:
            if j > 0:
                lstAllGrainIDs.extend(self.GetGrainAtomIDs(j).tolist())
        return lstAllGrainIDs
    def GetNonGrainAtomIDs(self):
        lstAllGrainIDs = []
        for j in self.__GrainLabels:
            if j > 0:
                lstAllGrainIDs.extend(self.GetGrainAtomIDs(j).tolist())
        return list(set(range(1,self.GetNumberOfAtoms()+1)).difference(lstAllGrainIDs))
    def EstimateLocalGrainBoundaryWidth(self):
        lstGrains = self.GetGrainLabels()
        if 0 in lstGrains:
            lstGrains.remove(0)
        intL = len(lstGrains)
        lstMaxDistances = []
        if len(lstGrains) <= 1:
            fltMax = 0
        else:
            fltMax = 0
            for i in range(1,intL+1):
                lstAllDistances = []
                arrPoints = self.__PeriodicGrains[i].GetExtendedPoints()
                for j in range(1, intL+1):
                    if i != j:
                        arrDistances1, arrIndices1 = self.__PeriodicGrains[j].Pquery(arrPoints, k=1)
                        arrIndices1 = mf.FlattenList(arrIndices1)
                        arrClosePoints = self.__PeriodicGrains[j].GetExtendedPoints()[arrIndices1]
                        arrDistances2, arrIndices2 = self.__PeriodicGrains[i].Pquery(arrClosePoints, k=1)
                        lstAllDistances.append(mf.FlattenList(arrDistances2))
                if len(lstAllDistances) > 0:
                    arrAllDistances = np.transpose(np.vstack(lstAllDistances))
                    arrSorted = np.sort(arrAllDistances, axis=1)
                    lstMaxDistances.append(np.max(arrSorted[:,0]))
            fltMax = np.max(lstMaxDistances)
        self.__MaxGBWidth = fltMax
        return fltMax
    def GetLabels(self, strColumn):
        lstReturn = []
        if strColumn in self.GetColumnNames():
            lstReturn = list(np.unique(self.GetColumnByName(strColumn), axis=0).astype('int'))
        return lstReturn
    def GetTripleLineIDs(self, intTripleLine, strColumnName = 'TripleLine'):
        intCol = self.GetColumnIndex(strColumnName)
        arrRows = np.where(self.GetAtomData()[:,intCol] == intTripleLine)[0]
        return self.GetAtomData()[arrRows,0]
    def GetGrainBoundaryIDs(self, intGB, strColumnName = 'GrainBoundary'):
        intCol = self.GetColumnIndex(strColumnName)
        arrRows = np.where(self.GetAtomData()[:,intCol] == intGB)[0]
        return self.GetAtomData()[arrRows][:,0]
    def FindGrainBoundaries(self, fltGBWidth, fltSearchWidth = None):
        if fltSearchWidth is None:
            fltSearchWidth = self.__MaxGBWidth
        lstGrains = self.GetGrainLabels()
        lstGrains.remove(0)
        lstTwos = list(it.combinations(lstGrains, 2))
        lstIDs = []
        self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]),'GrainBoundary', strFormat = '%i')
        intGBCol = self.GetColumnIndex('GrainBoundary')
        lstMeshPoints = []
        lstUsedTwos = []
        lstAllIDs = []
        for k in lstTwos:
            lstTemp,mpts = self.FindMeshAtomIDs(k,fltGBWidth)
            if len(mpts) > 0:
                mpts = self.WrapVectorIntoSimulationBox(mpts)
            if len(lstTemp)> 0:
                lstIDs.append(lstTemp)
                lstMeshPoints.append(mpts)
                lstUsedTwos.append(k)
                lstAllIDs.extend(lstTemp)
        intN = len(lstUsedTwos)
        lstDuplicateIDs = []
        for d in range(intN):
            for e in range(d+1, intN):
                if len(set(lstUsedTwos[d]).intersection(lstUsedTwos[e])) >= 1:
                    lstDuplicateIDs.extend(list(set(lstIDs[d]).intersection(lstIDs[e])))
        arrDuplicates = np.unique(lstDuplicateIDs).astype('int')
        self.__DuplicateGBIDs = arrDuplicates
        lstAllGBMesh = []
        for l in range(intN): 
            lstGBIDs = []  #lstIDs[l]
            lstExtraIDs = []
            arrMesh = self.FindDefectiveMesh(lstUsedTwos[l][0],lstUsedTwos[l][1],fltSearchWidth)
            if len(arrMesh) > 0  and len(lstMeshPoints[l]) > 0:
                arrMesh = np.append(arrMesh, lstMeshPoints[l], axis=0)
            elif len(lstMeshPoints[l]) > 0:
                arrMesh = lstMeshPoints[l]
            if len(arrMesh) > 0:
                arrMesh = self.WrapVectorIntoSimulationBox(arrMesh)
                arrMesh = np.unique(arrMesh, axis=0)
                lstAllGBMesh.append(arrMesh)
                arrIDs = self.GetGrainAtomIDs(0)
                arrIndices, arrDistances = self.__PeriodicGrains[0].Pquery_radius(arrMesh, fltGBWidth)
                arrIndices = self.__PeriodicGrains[0].GetPeriodicIndices(arrIndices)
                arrIndices = np.unique(mf.FlattenList(arrIndices))
                lstExtraIDs.extend(arrIDs[arrIndices])
                for m in lstUsedTwos[l]:
                    arrIDs = self.GetGrainAtomIDs(m)
                    arrIndices, arrDistances = self.__PeriodicGrains[m].Pquery_radius(arrMesh, fltGBWidth)
                    arrIndices = np.unique(mf.FlattenList(arrIndices))
                    arrIndices = self.__PeriodicGrains[m].GetPeriodicIndices(arrIndices)
                    arrIndices = np.unique(arrIndices)
                    if len(arrIndices) > 0 :
                        lstExtraIDs.extend(arrIDs[arrIndices])
                lstExtraIDs  = list(np.unique(lstExtraIDs))
                # if len(lstExtraIDs) > 0:
                #     arrIndices = self.__PeriodicGrains[0].GetPeriodicIndices(lstExtraIDs)
                #     lstExtraIDs = arrIDs[arrIndices]
            if len(lstExtraIDs) > 0:
                lstGBIDs.extend(lstExtraIDs)
                lstGBIDs = np.unique(lstGBIDs).tolist()
                self.SetColumnByIDs(lstGBIDs,intGBCol, (l+1)*np.ones(len(lstGBIDs)))
                lstGBIDs = self.GetGBAtomIDs(l+1)
                self.__PeriodicGrainBoundaries[l+1] = gf.PeriodicWrapperKDTree(self.GetAtomsByID(lstGBIDs)[:,1:4],self.GetCellVectors(), gf.FindConstraintsFromBasisVectors(self.GetCellVectors()),fltGBWidth,self.GetPeriodicDirections())
        self.SetColumnByIDs(arrDuplicates, intGBCol, -1*np.ones(len(arrDuplicates))) #remove any duplicate GBs which may leave a small gap around the triple oine  
        return lstAllGBMesh #returns the mesh points for plotting              
    def FindJunctionLines(self, fltRadius, intOrder, fltSearchRadius = None):
        if fltSearchRadius is None:
            fltSearchRadius = self.__MaxGBWidth
        self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]),'TripleLine', strFormat = '%i')
        intTJCol = self.GetColumnIndex('TripleLine')
        t=1
        lstAllTJs= []
        lstMergedPoints = self.FindJunctionMesh(fltRadius,intOrder)
        self.__JunctionMesh = lstMergedPoints
        for m in lstMergedPoints:
           # blnTJ = False
            lstAllIDs = []
            lstGrainBoundaries = self.GetGrainBoundaryLabels()
            lstGrainBoundaries.remove(0)
            lstTemp = []
            intTJ = 0
            for l in lstGrainBoundaries:
                arrIDs1 = self.GetGBAtomIDs(l)
                objTree = gf.PeriodicWrapperKDTree(self.GetAtomsByID(arrIDs1)[:,1:4],self.GetCellVectors(),gf.FindConstraintsFromBasisVectors(self.GetCellVectors()),fltSearchRadius, self.GetPeriodicDirections())
                arrIndices1, arrDistances = objTree.Pquery_radius(m, fltRadius)
                arrIndices1 = np.unique(mf.FlattenList(arrIndices1))
                if len(arrIndices1)> 0:
                    arrIndices1 = objTree.GetPeriodicIndices(arrIndices1)
                    arrIndices1 = np.unique(arrIndices1)
                    lstTemp.extend(arrIDs1[arrIndices1])
                    intTJ += 1
                    # if l == -1: #this is an overlapped region of grain boundaries
                    #     blnTJ =  True
            if  intTJ == intOrder +1: #check this is a triple line (3 real grain boundaries and -1 is the intersection region)
                lstAllIDs.extend(lstTemp)
                arrIDs2 = self.GetGrainAtomIDs(0)
                arrIndices2, arrDistances2 = self.__PeriodicGrains[0].Pquery_radius(m, fltRadius)
                arrIndices2 = np.unique(mf.FlattenList(arrIndices2))
                arrIndices2 = self.__PeriodicGrains[0].GetPeriodicIndices(arrIndices2)
                arrIndices2 = np.unique(arrIndices2)
                lstAllIDs.extend(arrIDs2[arrIndices2])   
            # if objDuplicate is not None:
            #     arrIndices, arrDistances = objDuplicate.Pquery_radius(m,fltWidth)
            #     arrIndices = np.unique(mf.FlattenList(arrIndices))
            #     if len(arrIndices) > 0:
            #         arrIndices = objDuplicate.GetPeriodicIndices(arrIndices)
            #         lstAllIDs.extend(self.__DuplicateGBIDs[arrIndices])
            arrIndices3 = np.unique(lstAllIDs)
            if len(arrIndices3) > 0:
                #lstTID = (arrIDs[np.unique(arrIndices)]).tolist()
                self.SetColumnByIDs(arrIndices3,intTJCol,t*np.ones(len(arrIndices3)))
                lstAllTJs.extend(arrIndices3.tolist())
                t +=1
        lstAllTJs = np.unique(lstAllTJs).tolist()
        intGBCol = self.GetColumnIndex('GrainBoundary')
        self.SetColumnByIDs(lstAllTJs,intGBCol,0*np.ones(len(lstAllTJs)))
    def FindJunctionMesh(self,fltWidth: float, intOrder: int):
        arrReturn = [] 
        lstMeshPoints = []
        arrOverlapIDs = self.GetGrainBoundaryIDs(-1)
        arrOverlapPoints = self.GetAtomsByID(arrOverlapIDs)[:,1:4]
        lstTJPoints = []
        lstSplitPoints = []
        clustering = DBSCAN(2*self.__LatticeParameter,min_samples=5).fit(arrOverlapPoints)
        arrLabels = clustering.labels_
        for a in np.unique(arrLabels):
            if a != -1:
                arrRows = np.where(arrLabels == a)[0]
                lstSplitPoints.append(arrOverlapPoints[arrRows])
        lstMatches = gf.GroupClustersPeriodically(lstSplitPoints, self.GetCellVectors(),2*self.__LatticeParameter)
        lstMergedPoints = []
        for l in lstMatches:
            if len(l) == 1:
                lstMergedPoints.append(lstSplitPoints[l[0]])
            else:
                lstTemp = []    
                for n in l:
                    lstTemp.extend(lstSplitPoints[n])
                lstMergedPoints.append(np.vstack(lstTemp))
        lstMeshTJPoints = []
        lstGrainLabels = self.GetGrainLabels()
        lstGrainLabels.remove(0)
        for k in lstMergedPoints:
            arrAllPoints = np.zeros([len(k), len(k[0]), intOrder])
            intPos = 0
            for j in range(len(lstGrainLabels)):
                arrDistances, arrIndices = self.__PeriodicGrains[lstGrainLabels[j]].Pquery(k,1)
                arrDistances = mf.FlattenList(arrDistances)
                arrIndices = mf.FlattenList(arrIndices)
                if np.all(arrDistances <= self.__MaxGBWidth):
                    arrAllPoints[:,:,intPos] = self.__PeriodicGrains[lstGrainLabels[j]].GetExtendedPoints()[arrIndices]
                    intPos += 1
            arrMeanPoints = np.mean(arrAllPoints, axis=2)
            arrRows = np.where(np.linalg.norm(k-arrMeanPoints,axis=1) < self.__MaxGBWidth/2)[0]
            if len(arrRows) > 0:
                arrReturn = self.WrapVectorIntoSimulationBox(arrMeanPoints[arrRows])
                lstMeshTJPoints.append(arrReturn) 
        return lstMeshTJPoints
        
    def GetPeriodicGrainBoundary(self, intKey):
        return self.__PeriodicGrainBoundaries[intKey]
    def FindMeshAtomIDs(self, lstGrains, fltWidth):
        arrReturn = []
        intNumberOfGrains = len(lstGrains)
        if set(lstGrains).issubset(set(self.__GrainLabels)):
           # arrPoints = self.GetAtomsByID(self.GetNonGrainAtomIDs())[:,0:4]
            lstIDs = []
            lstIDs.extend(self.GetGrainAtomIDs(0))
            #lstGBLabels =  self.GetGrainBoundaryLabels()
            # if -1 in lstGBLabels:
            #     lstIDs.extend(self.GetGBAtomIDs(-1))
            # #arrPoints = self.GetAtomsByID(self.GetGrainAtomIDs(0))
            arrPoints = self.GetAtomsByID(lstIDs)
            arrIDs = arrPoints[:,0]
            arrPoints = arrPoints[:,1:4]
            lstAllDistances = []
            lstAllPoints = []
            arrAllPoints = np.zeros([len(arrPoints),3,intNumberOfGrains])
            i = 0
            for l in lstGrains:
                arrDistances1, arrIndices1 = self.__PeriodicGrains[l].Pquery(arrPoints, k=1)
                arrGrainPoints = self.__PeriodicGrains[l].GetExtendedPoints()
                arrAllPoints[:,:,i] = arrGrainPoints[mf.FlattenList(arrIndices1)]
                #lstAllPoints.append(arrGrainPoints[mf.FlattenList(arrIndices1)])
                lstAllDistances.append(np.array([x[0] for x in arrDistances1]))
                i += 1
            arrMean = np.mean(arrAllPoints,axis=2)
            arrMeanDistances = np.linalg.norm(arrPoints - arrMean, axis=1)
            #arrAllPoints = np.transpose(np.vstack(lstAllPoints))
            arrAllDistances = np.transpose(np.vstack(lstAllDistances))
            #arrSum = np.sum(arrAllDistances[:,:intNumberOfGrains], axis=1)
            arrRows1 = np.where(np.all(arrAllDistances <= self.__MaxGBWidth,axis=1))[0]
            arrRows2 = np.where(arrMeanDistances <= fltWidth)[0]
           # arrRows = np.where(np.all(arrSum <= fltWidth*intNumberOfGrains))[0]
            arrReturn = []
            arrMeanPoints = []
            if len(arrRows1) > 0 and len(arrRows2) >0:
                arrRows = np.array(list(set(arrRows1.tolist()).intersection(arrRows2.tolist())))
                if len(arrRows) > 0:
                    arrReturn = arrIDs[arrRows]
                    arrMeanPoints = arrMean[arrRows]
        return arrReturn, arrMeanPoints            
    def FindDefectiveMesh(self,inGrain1, inGrain2, fltWidth = None):
        if fltWidth is None:
            fltWidth = self.__MaxGBWidth
        lstPermutations =  list(it.permutations([inGrain1,inGrain2],2))
        lstAllPoints = []
        for i in lstPermutations:
            intGrain1 = i[0]
            intGrain2 = i[1]
            arrDistances1, arrIndices1 = self.__PeriodicGrains[intGrain1].Pquery(self.__PeriodicGrains[intGrain2].GetExtendedPoints(),k=1) 
            arrDistances1 = np.array([x[0] for x in arrDistances1])
            arrRows1 = np.where(arrDistances1 < fltWidth)[0]
            if len(arrRows1) > 0:
                arrIndices1 = np.array([x[0] for x in arrIndices1[arrRows1]])
                arrPoints1 = self.__PeriodicGrains[intGrain1].GetExtendedPoints()[arrIndices1]
                arrDistances2, arrIndices2 = self.__PeriodicGrains[intGrain2].Pquery(arrPoints1,k=1)
                arrIndices2 = np.array([x[0] for x in arrIndices2])
                arrPoints2 = self.__PeriodicGrains[intGrain2].GetExtendedPoints()[arrIndices2]
                lstAllPoints.append(np.unique((arrPoints1+arrPoints2)/2,axis=0))
        if len(lstAllPoints)> 0:
            return np.concatenate(lstAllPoints,axis=0)
        else:
            return [] 
    def SpreadToHigherEnergy(self, inlstIDs: list):
        lstReturnIDs = list(set(inlstIDs))
        lstAllIDs = list(set(inlstIDs))
        while len(lstReturnIDs) > 0:
            lstNewIDs = []
            arrCentres = self.GetAtomsByID(lstReturnIDs)[:,0:4]
            for x in arrCentres:
                lstNewIDs.extend(self.HigherPEPoints(x))
            lstReturnIDs = list(set(lstNewIDs).difference(lstAllIDs))
            lstAllIDs.extend(lstReturnIDs)
        return list(set(lstAllIDs).difference(inlstIDs))
    def HigherPEPoints(self,arrPoint):
        lstReturn = []
        intID = arrPoint[0]
        lstDistances,lstIndices = self.__NearestNeighbour.kneighbors(np.array([arrPoint[1:4]]))
        lstIDs = self.GetRows(lstIndices[0])[:,0].astype('int')
        # if intID in lstIDs:
        #     intIndex = lstIDs.index(intID)
        #     del lstIDs[intIndex]
        #     del lstDistances[intIndex]
        arrPE = self.GetColumnByIDs(lstIDs,self.GetColumnIndex('c_pe1'))
        s =self.__LatticeParameter
        arrDistances = np.array(list(map(lambda x: 1/(x/s+1), lstDistances[0])))
        fltCurrentPE = np.mean(arrPE*arrDistances)
        arrCentres = self.GetAtomsByID(lstIDs)[:,0:4]
        for i in arrCentres:
            lstNewDistances,lstNewIndices = self.__NearestNeighbour.kneighbors(np.array([i[1:4]]))
            lstNewIDs = self.GetRows(lstNewIndices[0])[:,0].astype('int')
            # if i[0] in lstNewIDs:
            #     intIndex = lstNewIDs.index(intID)
            #     del lstNewIDs[intIndex]
            #     del lstNewDistances[intIndex]
            arrPE = self.GetColumnByIDs(lstNewIDs,self.GetColumnIndex('c_pe1'))
            arrDistances = np.array(list(map(lambda x: 1/(x/s+1), lstNewDistances[0])))
            fltNewPE = np.mean(arrPE*arrDistances)
            if fltNewPE >= fltCurrentPE:
                  lstReturn.append(int(i[0]))
        return lstReturn
    def GetLabels(self,strName):
        lstReturn = []
        if strName in self.GetColumnNames():
            lstReturn = list(np.unique(self.GetColumnByName(strName), axis=0).astype('int'))
        return lstReturn
    def GetGrainLabels(self):
        if 'GrainNumber' in self.GetColumnNames():
            self.__GrainLabels = list(np.unique(self.GetColumnByName('GrainNumber'), axis=0).astype('int'))  
        else:
            self.__GrainLabels = []
        return self.__GrainLabels
    def GetGrainBoundaryLabels(self):
        self.__GrainBoundaryLabels = list(np.unique(self.GetColumnByName('GrainBoundary'), axis=0).astype('int'))
        return self.__GrainBoundaryLabels  
    def GetTripleLineLabels(self):
        self.__GrainBoundaryLabels = list(np.unique(self.GetColumnByName('GrainBoundary'), axis=0).astype('int'))
        return self.__GrainBoundaryLabels  
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
    
    
    def GetGrainAtomIDs(self, intGrainNumber):
        lstGrainAtoms = list(np.where(self.GetColumnByName('GrainNumber').astype('int') == intGrainNumber)[0])
        return self.GetAtomData()[lstGrainAtoms,0].astype('int')
    def GetGBAtomIDs(self, intGBNumber):
        lstGBAtoms = list(np.where(self.GetColumnByName('GrainBoundary').astype('int') == intGBNumber)[0])
        return self.GetAtomData()[lstGBAtoms,0].astype('int')
    def GetAtomIDsByOrientation(self,inQuaternion: np.array, intLatticeType: int,fltTolerance = 0.001):
        lstRows = []
        intFirst = self.GetColumnIndex('c_pt[1]')
        intSecond = self.GetColumnIndex('c_pt[7]')
        arrQuaternions = self.GetAtomData()[:,intFirst:intSecond+1]
        #arrRows = np.matmul(arrQuaternions[:,1:5], inQuaternion)
        #lstRows.append(np.where((np.abs(arrRows) > 1- fltTolerance) & (arrQuaternions[:,0].astype('int') == intLatticeType))[0])
        for j in gf.CubicQuaternions():
            y = gf.QuaternionProduct(inQuaternion,j)
            #y = gf.QuaternionProduct(j,inQuaternion)
            arrRows = np.matmul(arrQuaternions[:,1:5], y)
            lstRows.append(np.where((np.abs(arrRows) > 1-fltTolerance) & (arrQuaternions[:,0].astype('int') == intLatticeType))[0])
        arrRows2 = np.unique(np.concatenate(lstRows))
        rtnValue = []
        if len(arrRows2) > 0:
            arrIDs = self.GetColumnByIndex(0)[arrRows2].astype('int')
            rtnValue = arrIDs
            # clustering = DBSCAN(eps=1.05*self.__objRealCell.GetNearestNeighbourDistance()).fit(self.GetAtomsByID(arrIDs)[:,1:4])
            # arrLabels = clustering.labels_
            # arrUniqueLabels,arrCounts = np.unique(arrLabels,return_counts=True)
            # arrMax = np.argmax(arrCounts)
            # if arrUniqueLabels[arrMax] != -1:
            #     rtnValue =  arrIDs[arrLabels==arrUniqueLabels[arrMax]]
            #     self.AddColumn(np.zeros([self.GetNumberOfAtoms(),1]),'GrainNumber',strFormat='%i')
            #     self.SetColumnByIDs(rtnValue,self.GetColumnIndex('GrainNumber'),np.ones(len(rtnValue)))
        return rtnValue 
    def GetAtomIDsByOrderParameter(self, intOrderIndex: list):
        strColumnName = 'f_' + str(intOrderIndex) + '[2]'
        arrRows1 = np.where(self.GetColumnByName(strColumnName) == 1)[0]
        arrRows2 = np.where(self.GetColumnByName(strColumnName) == -1)[0]
        arrIDs1 =  self.GetColumnByIndex(0)[arrRows1].astype('int')
        arrIDs2 =  self.GetColumnByIndex(0)[arrRows2].astype('int')
        return arrIDs1, arrIDs2
        


         
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
        self.__BasisVectors = np.zeros([3,3])
        for i in range(len(arrCuboidCellVectors)): #calculate a scaling factor that splits into exact integer multiples
            intValue = np.round(np.linalg.norm(arrCuboidCellVectors[i])/arrGridDimensions[i]).astype('int')
            arrModArray[i] = intValue
            arrGridDimensions[i] = np.linalg.norm(arrCuboidCellVectors[i])/intValue
            self.__BasisVectors[i,i] = intValue
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
    def GetExtraJunctionLinePoints(self, intJunctionLine: int): #includes periodic extension for junction line length calculation
        arrPoints = np.argwhere(self.__JunctionLinesArray == intJunctionLine)      
        lstPeriodicDirections = self.GetPeriodicExtensions(intJunctionLine, 'JunctionLine')
        lstExtraPoints = []
        for k in lstPeriodicDirections:
            arrRows = np.where(arrPoints[:,k] == 0)[0]
            arrVector = np.zeros(3)
            arrVector[k] = self.__ModArray[k]
            if len(arrRows) > 0:
                lstExtraPoints.append(arrPoints[arrRows]+arrVector)
        arrPoints = np.concatenate(lstExtraPoints, axis=0)
        return np.matmul(np.matmul(self.MergeMeshPoints(arrPoints) +np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion) 
    def GetExtraGrainBoundaryPoints(self, intGrainBoundary: int): #includes periodic extension for the grain boundary surface area calculation
        arrPoints = np.argwhere(self.__GrainBoundariesArray == intGrainBoundary)      
        lstPeriodicDirections = self.GetPeriodicExtensions(intGrainBoundary, 'GrainBoundary')
        lstExtraPoints = []
        for k in lstPeriodicDirections:
            arrRows = np.where(arrPoints[:,k] == 0)[0]
            arrVector = np.zeros(3)
            arrVector[k] = self.__ModArray[k]
            if len(arrRows) > 0:
                lstExtraPoints.append(arrPoints[arrRows]+arrVector)
        arrPoints = np.concatenate(lstExtraPoints, axis=0)
        return np.matmul(np.matmul(self.MergeMeshPoints(arrPoints) +np.ones(3)*0.5, self.__InverseScaling), self.__InverseBasisConversion) 
    def GetSurfaceMesh(self, intGrainBoundary: int):
        arrMeshPoints = np.argwhere(self.__GrainBoundariesArray == intGrainBoundary)
        arrPoints = self.MergeMeshPoints(arrMeshPoints)
        lstTJIDs = self.GetAdjacentJunctionLines(intGrainBoundary)
        lstReturnPoints = []
        lstBoundaryType = ['pp','pp','pp']
        for k in self.GetCompleteDirections(arrMeshPoints):
            lstBoundaryType[k] = 'f'
        objNearest = KDTree(arrPoints)
        if len(lstTJIDs) > 1:
            arrStartPoints = self.MergeMeshPoints(np.argwhere(skeletonize_3d(self.__JunctionLinesArray == lstTJIDs[0])))
            arrDistances, arrIndices = objNearest.query(arrStartPoints, k=1, return_distance=True)
            arrEndPoints = self.MergeMeshPoints(np.argwhere(skeletonize_3d(self.__JunctionLinesArray == lstTJIDs[0])))
            if np.all(arrDistances > np.round(np.sqrt(3),3)):
           # if len(arrIndices) == 0: #this junction line is no longer adjacent to the merged mesh points
                arrEndPoints = arrStartPoints
                arrStartPoints = self.MergeMeshPoints(np.argwhere(skeletonize_3d(self.__JunctionLinesArray == lstTJIDs[1])))
                arrIndices = np.unique(np.concatenate(objNearest.query(arrStartPoints, k=1,return_distance=False),axis=0))
            #    intEnd = 0
        blnNonZero = True
       # lstReturnPoints.append(arrStartPoints)
        objEndPoints = KDTree(arrEndPoints)
        while blnNonZero and len(arrPoints)> 0:
            objNearest= KDTree(arrPoints)
            arrDistances, arrIndices = objNearest.query(arrStartPoints, k=1)    
            arrRows = np.where(arrDistances <= np.round(np.sqrt(3),3))[0]
            if len(arrIndices) > 0 and len(arrRows) > 0:
                arrIndices = np.unique(np.concatenate(arrIndices[arrRows],axis=0))
                arrStartPoints = arrPoints[arrIndices]   
                lstReturnPoints.append(arrStartPoints)
                arrPoints = np.delete(arrPoints, arrIndices.astype('int'),axis=0)
            else:
                blnNonZero = False
        # if len(lstTJIDs) == 2:
        #     arrEndPoints =  np.argwhere(skeletonize_3d(self.__JunctionLinesArray == lstTJIDs[intEnd]))
        #     arrEndPoints = gf.PeriodicShiftAllCloser(np.mean(lstReturnPoints[-1],axis=0),arrEndPoints, self.__BasisVectors, np.linalg.inv(self.__BasisVectors),lstBoundaryType)
        #     lstReturnPoints.append(arrEndPoints)
        return lstReturnPoints
    def GetCompleteDirections(self, arrPoints: np.array):
        lstPeriodicDirections = []
        for j in range(3):
            if list(np.unique(arrPoints[:,j])) == list(range(self.__ModArray[j])):
                lstPeriodicDirections.append(j)
        return lstPeriodicDirections
    def GetGrainBoundaryArray(self):
        return self.__GrainBoundariesArray