import re
import numpy as np
import GeometryFunctions as gf
from scipy import spatial, optimize
#from sklearn.cluster import AffinityPropagation
from skimage.morphology import skeletonize, thin, medial_axis, label, remove_small_holes

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
                objTimeStep = LAMMPSPostProcess(timestep, N,intNumberOfColumns,lstColumnNames, lstBoundaryType, lstBounds,intLatticeType)
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


class LAMMPSPostProcess(LAMMPSTimeStep):
    def __init__(self, fltTimeStep: float,intNumberOfAtoms: int, intNumberOfColumns: int, lstColumnNames: list, lstBoundaryType: list, lstBounds: list,intLatticeType: int):
        LAMMPSTimeStep.__init__(self,fltTimeStep,intNumberOfAtoms, lstColumnNames, lstBoundaryType, lstBounds)
        self.__Dimensions = self.GetNumberOfDimensions()
        self.__LatticeStructure = intLatticeType #lattice structure type as defined by OVITOS
        self.__intStructureType = int(self.GetColumnNames().index('StructureType'))
        self.__intPositionX = int(self.GetColumnNames().index('x'))
        self.__intPositionY = int(self.GetColumnNames().index('y'))
        self.__intPositionZ = int(self.GetColumnNames().index('z'))
        self.__intPE = int(self.GetColumnNames().index('c_pe1'))
        self.__GrainBoundaries = []
    def CategoriseAtoms(self):    
        lstOtherAtoms = list(np.where(self.GetColumnByIndex(self.__intStructureType).astype('int') == 0)[0])
        lstLatticeAtoms =  list(np.where(self.GetColumnByIndex(self.__intStructureType).astype('int') == self.__LatticeStructure)[0])
        lstUnknownAtoms = list(np.where(np.isin(self.GetColumnByIndex(self.__intStructureType).astype('int') ,[0,1],invert=True))[0])
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
        return arrPoints[:,self.__intPositionX], arrPoints[:,self.__intPositionY], arrPoints[:,self.__intPositionZ]
    def __GetCoordinate(self, intIndex: int):
        arrPoint = self.GetRow(intIndex)
        return arrPoint[self.__intPositionX:self.__intPositionZ+1]
    def __GetCoordinates(self, strList: list):
        arrPoints = self.GetRows(strList)
        return arrPoints[:,self.__intPositionX:self.__intPositionZ+1]
    def MakePeriodicDistanceMatrix(self, inVector1: np.array, inVector2: np.array)->np.array:
        arrPeriodicDistance = np.zeros([len(inVector1), len(inVector2)])
        for j in range(len(inVector1)):
            for k in range(j,len(inVector2)):
                arrPeriodicDistance[j,k] = self.PeriodicMinimumDistance(inVector1[j],inVector2[k])
        return arrPeriodicDistance + arrPeriodicDistance.T - np.diag(arrPeriodicDistance.diagonal())
    def GetNumberOfTripleLines(self)->int:
        return len(self.__TripleLines)
    def GetTripleLines(self)->np.array:
        return self.__TripleLines
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        arrVectorPeriodic = self.PeriodicEquivalents(np.abs(inVector1-inVector2))
        return np.min(np.linalg.norm(arrVectorPeriodic, axis=1))
    def FindNonGrainMean(self, inPoint: np.array, fltRadius: float): 
        lstPointsIndices = []
        lstPoints =[]
        arrPeriodicPositions = self.PeriodicEquivalents(inPoint)
        arrPeriodicTranslations = arrPeriodicPositions - inPoint
        for intIndex,arrPoint in enumerate(arrPeriodicPositions): 
                lstPointsIndices= self.__NonLatticeTree.query_ball_point(arrPoint, fltRadius)
                if len(lstPointsIndices) > 0:
                    lstPoints.extend(np.subtract(self.__NonLatticeTree.data[lstPointsIndices],arrPeriodicTranslations[intIndex]))
        if len(lstPoints) ==0:
            return inPoint
        else:
            return np.mean(np.array(lstPoints), axis=0)
    def FindTriplePoints(self,fltGridLength: float, blnFindGrainBoundaries = False):
        lstGrainBoundaries = []
        fltMidHeight = self.GetCellVectors()[2,2]/2
        objQPoints = QuantisedRectangularPoints(self.GetOtherAtoms()[:,self.__intPositionX:self.__intPositionY+1],self.GetUnitBasisConversions()[0:2,0:2],5,fltGridLength)
        arrTripleLines = objQPoints.FindTriplePoints()
        self.__TripleLineDistanceMatrix = spatial.distance_matrix(arrTripleLines[:,0:2], arrTripleLines[:,0:2])
        arrTripleLines[:,2] = fltMidHeight*np.ones(len(arrTripleLines))
        for j  in range(len(arrTripleLines)):
            arrTripleLines[j] = self.FindNonGrainMean(arrTripleLines[j], fltGridLength/np.sqrt(2))
        self.__TripleLines = arrTripleLines 
        if blnFindGrainBoundaries:
            lstGrainBoundaries = objQPoints.GetGrainBoundaries()
           # arrGrainBoundaries = np.ones([len(arr2DGrainBoundaries),3])*self.GetCellVectors()[2]/2
           # arrGrainBoundaries[:,0:2] = arr2DGrainBoundaries
            for j in range(len(lstGrainBoundaries)):
                for k,Points in enumerate(lstGrainBoundaries[j]):
                    arrPoint = np.array([Points[0], Points[1],fltMidHeight])
                    lstGrainBoundaries[j][k] = self.FindNonGrainMean(arrPoint, fltGridLength/np.sqrt(2))
            self.__GrainBoundaries = lstGrainBoundaries
        return arrTripleLines
    def GetGrainBoundaries(self, intValue = None):
        if intValue is None:
            return self.__GrainBoundaries
        else:
            return self.__GrainBoundaries[intValue]
    def FindCylindricalAtoms(self, arrCentre: np.array, fltRadius: float)->list:
        lstPoints = []
        arrCentres = self.PeriodicEquivalents(arrCentre)
        for j in arrCentres:
            lstPoints.extend(np.where(np.linalg.norm(self.GetAtomData()[:,1:3]-j[0:2],axis=1) 
                         <fltRadius)[0])
        return list(np.unique(lstPoints))
    def FindBoxAtoms(self, arrCentre: np.array, arrLength: np.array, arrWidth: np.array)->list:
        lstPoints = []
        fltLength = np.linalg.norm(arrLength, axis=0)
        fltWidth = np.linalg.norm(arrWidth, axis=0)
        arrCentre3d = np.array([arrCentre[0],arrCentre[1],0])
        arrCentres = self.PeriodicEquivalents(arrCentre3d)
        for j in arrCentres:
            arrCurrentPoints = self.GetAtomData()[:,1:3]-j[0:2]
            lstPoints.extend(np.where((np.abs(np.dot(arrCurrentPoints, arrLength/fltLength )) < fltLength/2) 
             & (np.abs(np.dot(arrCurrentPoints, arrWidth/fltWidth)) < fltWidth/2))[0])
        return list(np.unique(lstPoints))
    def FindValuesInCylinder(self, arrCentre: np.array, fltRadius: float, intColumn: int):
        lstIndices = self.FindCylindricalAtoms(arrCentre, fltRadius)
        return self.GetRows(lstIndices)[:,intColumn]
    def MergePeriodicTripleLines(self, fltDistanceTolerance: float):
        lstMergedIndices = []
        setIndices = set(range(self.GetNumberOfTripleLines()))
        lstCurrentIndices = []
        arrPeriodicDistanceMatrix = self.MakePeriodicDistanceMatrix(self.__TripleLines,self.__TripleLines)
        while len(setIndices) > 0:
            lstCurrentIndices = list(*np.where(arrPeriodicDistanceMatrix[setIndices.pop()] < fltDistanceTolerance))
            lstMergedIndices.append(lstCurrentIndices)
            setIndices = setIndices.difference(lstCurrentIndices)
        return lstMergedIndices
    def EstimateTripleLineEnergy(self, fltPEDatum: float, fltGridLength, blnFloatPerVolume = True, blnAsymptoticLinear = True):
        arrEnergy = np.zeros([len(self.__TripleLines),3])
        self.FindTriplePoints(fltGridLength)
        #fltClosest = np.min(self.__TripleLineDistanceMatrix[self.__TripleLineDistanceMatrix !=0])
        fltHeight = np.dot(self.GetCellVectors()[2], np.array([0,0,1]))
        for j in range(len(self.__TripleLines)):
            lstRadius = []
            lstPEValues = []
            lstExcessEnergy = []
            lstPEFit = []
            lstGBFit = []
            fltClosest = np.sort(self.__TripleLineDistanceMatrix[j])[1] #finds the closet triple line
            for k in range(0, np.floor(fltClosest/2).astype('int')): #only search halfway between the points
                lstRadius.append(k)
                lstPEValues = self.FindValuesInCylinder(self.__TripleLines[j],k,self.__intPE)
                lstExcessEnergy.append(np.sum(lstPEValues)-fltPEDatum*len(lstPEValues))
            if blnAsymptoticLinear:
                poptA,popvA = optimize.curve_fit(gf.AsymptoticLinear,lstRadius, lstExcessEnergy)    
            else:    
                popt,popv = optimize.curve_fit(gf.PowerRule,lstRadius, lstExcessEnergy)
                popt2,popv2 = optimize.curve_fit(gf.LinearRule, lstRadius[-5:], lstExcessEnergy[-5:])
            for r in lstRadius:
                if blnAsymptoticLinear:
                    lstPEFit.append(gf.AsymptoticLinear(r, poptA[0],poptA[1]))
                    lstGBFit.append(poptA[0]*r)
                else:
                    lstPEFit.append(gf.PowerRule(r, popt[0],popt[1]))
                    lstGBFit.append(gf.LinearRule(r, popt2[0], popt2[1])-popt2[1]) #fitted to the last few points
            arrEnergy[j,0] = lstPEFit[-1]
            arrEnergy[j,1] = lstGBFit[-1]
            arrEnergy[j,2] = lstPEFit[-1]- lstGBFit[-1]
            if blnFloatPerVolume:
                fltVolume = fltClosest**2*fltHeight*np.pi/4
                arrEnergy[j,:] = (1/fltVolume)*arrEnergy[j,:]
        return arrEnergy


class QuantisedRectangularPoints(object): #linear transform parallelograms into a rectangular parameter space
    def __init__(self, in2DPoints: np.array, inUnitBasisVectors: np.array, n: int, fltGridSize: float):
        self.__WrapperWidth = n
        self.__BasisVectors = inUnitBasisVectors
        self.__InverseMatrix =  np.linalg.inv(inUnitBasisVectors)
        self.__GridSize = fltGridSize
        arrPoints =  np.matmul(in2DPoints, self.__BasisVectors)*(1/fltGridSize)
        #arrPoints[:,0] = np.linalg.norm(inBasisVectors[0]/fltGridSize, axis=0)*arrPoints[:,0]
        #arrPoints[:,1] = np.linalg.norm(inBasisVectors[1]/fltGridSize, axis=0)*arrPoints[:,1]
        intMaxHeight = np.ceil(np.max(arrPoints[:,0])).astype('int')
        intMaxWidth = np.ceil(np.max(arrPoints[:,1])).astype('int')
        self.__ArrayGrid =  np.zeros([(intMaxHeight+1),intMaxWidth+1])
        arrPoints = np.round(arrPoints).astype('int')
        for j in arrPoints:
            self.__ArrayGrid[j[0],j[1]] = 1 #this array represents the simultion cell
        self.__ExtendedArrayGrid = np.zeros([np.shape(self.__ArrayGrid)[0]+2*n,np.shape(self.__ArrayGrid)[1]+2*n])
        self.__ExtendedArrayGrid[n:-n, n:-n] = self.__ArrayGrid
        self.__ExtendedArrayGrid[1:n+1, n:-n] = self.__ArrayGrid[-n:,:]
        self.__ExtendedArrayGrid[-n-1:-1, n:-n] = self.__ArrayGrid[:n,:]
        self.__ExtendedArrayGrid[:,1:n+1] = self.__ExtendedArrayGrid[:,-2*n:-n]
        self.__ExtendedArrayGrid[:,-n-1:-1] = self.__ExtendedArrayGrid[:,n:2*n]
        self.__ExtendedSkeletonGrid = skeletonize(self.__ExtendedArrayGrid).astype('int')
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
    def GetExtendedArrayGrid(self):
        return self.__ExtendedArrayGrid
    def GetExtendedSkeletonPoints(self):
        return self.__ExtendedSkeletonGrid   
    def ClassifyGBPoints(self,m:int,blnFlagEndPoints = False)-> np.array:
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
        #        if intSwaps < 4 and blnFlagEndPoints:
        #            self.SetSkeletonValue(x,self.__DislocationValue)
        #self.__Dislocations = np.argwhere(self.__ExtendedSkeletonGrid == self.__DislocationValue)
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
    def FindTriplePoints(self)->np.array:
        self.__ResetSkeletonGrid()
        self.__TriplePoints = self.ClassifyGBPoints(3, False)
        return self.__ConvertToCoordinates(self.__TriplePoints)
    def FindGrainBoundaries(self)->np.array:
        self.ClassifyGBPoints(5,False)
        k = self.__WrapperWidth 
        arrSkeleton = np.copy(self.__ExtendedSkeletonGrid)
        arrSkeleton[:k,:] = 0
        arrSkeleton[-k:,:] = 0
        arrSkeleton[:,:k] = 0
        arrSkeleton[:,-k:] = 0
        arrSkeleton = (arrSkeleton == self.__GBValue).astype('int')
        self.__GrainBoundaryLabels = label(arrSkeleton)
        self.__NumberOfGrainBoundaries = len(np.unique(self.__GrainBoundaryLabels)) -1
        self.__blnGrainBoundaries = True
    def GetGrainBoundaryLabels(self):
        if  not self.__blnGrainBoundaries:
            self.FindGrainBoundaries()
        return self.__GrainBoundaryLabels
    def GetNumberOfGrainBoundaries(self):
        if not self.__blnGrainBoundaries:
            self.FindGrainBoundaries()
        return self.__NumberOfGrainBoundaries
    def GetGrainBoundaries(self):
        lstGrainBoundaries = []
        if not self.__blnGrainBoundaries:
            self.FindGrainBoundaries()
        for j in range(1,self.__NumberOfGrainBoundaries+1): #label 0 is the background
            arrPoints = np.argwhere(self.__GrainBoundaryLabels == j) #find the positions for each label
            lstGrainBoundaries.append(self.__ConvertToCoordinates(arrPoints))
        return lstGrainBoundaries

# class Quantised2DPoints(object):
#     def __init__(self, in2DPoints: np.array, infltSideLength: float, inBoundaryVectors: np.array, intWrapperWidth: int):
#         self.__GridLength = infltSideLength
#         self.__DataPoints = in2DPoints
#         self.__MaxX = np.max(np.abs(in2DPoints[:,0]))
#         self.__MaxY = np.max(np.abs(in2DPoints[:,1]))
#         self.__WrapperWidth = intWrapperWidth
#         self.__GrainValue = 0
#         self.__GBValue = 1 #just fixed constants used in the array 
#         self.__DislocationValue = 2
#         self.__TripleLineValue = 3
#         self.QuantisedVectors(infltSideLength,inBoundaryVectors)
#         self.Skeletonised = False
#         self.MakeGrid()
#         self.ExtendGrid()
#         self.__TriplePoints = []
#     def MakeGrid(self): 
#         arrReturn = np.zeros([np.round(self.__MaxX/self.__GridLength+1).astype('int'),np.round(self.__MaxY/self.__GridLength+1).astype('int')])
#         for j in self.__DataPoints:
#             arrReturn[(np.round(j[0]/self.__GridLength)).astype('int'),(np.round(j[1]/self.__GridLength)).astype('int')] +=1
#         self.__GridArray = arrReturn.astype('bool').astype('int')
#         self.__Height, self.__Width = np.shape(arrReturn)
#     def ExtendGrid(self): #2n excess rows and columns around the GB array
#         n = self.__WrapperWidth
#         self.__ExtendedGrid = np.zeros([np.round(self.__MaxX/self.__GridLength+2*n+1).astype('int'),
#                                         np.round(self.__MaxY/self.__GridLength +2*n+1).astype('int')])
#         self.__ExtendedGrid[n: -n, n: -n] = self.__GridArray.astype('int')                                
#     def CopyGBToWrapper(self,arrPoints: np.array, arrShift: np.array, intValue = None):
#         for j in arrPoints:
#                 intCurrent =  int(self.__ExtendedGrid[j[0],j[1]])
#                 if intCurrent == self.__GBValue or intCurrent == self.__DislocationValue:
#                     if intValue is None:
#                        # self.__ExtendedGrid[j[0]+arrShift[0],j[1]+arrShift[1]] = intCurrent
#                         if intCurrent !=0:
#                             self.SetSkeletonValue(j+arrShift,intCurrent)
#                     else:
#                         self.__ExtendedGrid[j[0]+arrShift[0],j[1]+arrShift[1]] = intValue   
#     def CopyOppositeEdges(self, arrPoints, arrShift):
#         for j in arrPoints:
#             intCurrent =  int(self.__SkeletonGrid[j[0],j[1]])
#             if intCurrent != self.__GrainValue:
#             # self.__ExtendedGrid[j[0]+arrShift[0],j[1]+arrShift[1]] = intCurrent
#                 self.SetSkeletonValue(j+arrShift,intCurrent)
#     def QuantisedVectors(self,infltSize: float, inBoundaryVectors: np.array):
#         self.__QVectors = np.array([gf.QuantisedVector(inBoundaryVectors[0]/infltSize),gf.QuantisedVector(inBoundaryVectors[1]/infltSize)]) 
#     def GetQVectors(self)->np.array:
#         return self.__QVectors
#     def GetExtendedGrid(self)->np.array:
#         return self.__ExtendedGrid
#     def GetGrid(self)->np.array:
#         return self.__GridArray
#     def CopyPointsToWrapper(self, intValue =1):
#         self.__vctDown = self.__QVectors[0]#assumes vertically down
#         self.__vctAcross = self.__QVectors[1]#assumes diagonal going right and possibly down or  up
#         self.__vctExtDown = gf.ExtendQuantisedVector(self.__QVectors[0][-1],2*self.__WrapperWidth)
#         vctAcrossTranslation = np.array([0,self.__WrapperWidth])
#         self.__vctDownEdge = self.__vctExtDown + vctAcrossTranslation
#         self.__vctExtAcross = gf.ExtendQuantisedVector(self.__QVectors[1][-1],2*self.__WrapperWidth)
#         vctDownTranslation = np.array([self.__WrapperWidth - self.__vctAcross[self.__WrapperWidth][0],0])
#         self.__vctAcrossEdge = self.__vctExtAcross + vctDownTranslation
#         for k in range(self.__WrapperWidth):
#             self.CopyGBToWrapper(self.__vctAcrossEdge+self.__vctDown[k], self.__vctDown[-1], intValue)
#             self.CopyGBToWrapper(self.__vctAcrossEdge+self.__vctDown[-1]-self.__vctDown[k], -self.__vctDown[-1],intValue)
#             self.CopyGBToWrapper(self.__vctDownEdge+self.__vctAcross[k], self.__vctAcross[-1],intValue)
#             self.CopyGBToWrapper(self.__vctDownEdge+self.__vctAcross[-1] -self.__vctAcross[k], -self.__vctAcross[-1],intValue)
#     def ClearWrapperValues(self):
#         lstPoints = []
#         self.__SkeletonGrid[:,:self.__WrapperWidth] = self.__GrainValue #removes left wrapper column
#         self.__SkeletonGrid[:,-self.__WrapperWidth:] = self.__GrainValue #remove right wrapper column
#         for k in range(0,self.__WrapperWidth-1):
#             # lstPoints.extend(self.__vctDownEdge-self.__vctAcross[k])
#             # lstPoints.extend(self.__vctDownEdge+self.__vctAcross[k] + self.__vctAcross[-1])
#             # lstPoints.extend(self.__vctAcrossEdge-self.__vctDown[k])
#             # lstPoints.extend(self.__vctAcrossEdge+self.__vctDown[k] + self.__vctDown[-1])
#             lstPoints.extend(self.__vctAcross+np.array([k, self.__WrapperWidth]))
#             lstPoints.extend(self.__vctAcross+np.array([k+1+self.__Height, self.__WrapperWidth]))
#             #lstPoints.extend(self.__vctExtDown+np.array([0 ,self.__Width + self.__WrapperWidth + k]))
#             for j in lstPoints:
#                 self.SetSkeletonValue(j,self.__GrainValue)
#        # self.CopyOppositeEdges(self.__vctAcrossEdge,self.__vctDown[-1])
#     def __ConvertToCoordinates(self, inArrayPosition: np.array): #takes array position and return real 2D coordinates
#         return (inArrayPosition-np.array([self.__WrapperWidth, self.__WrapperWidth]))*self.__GridLength 
#     def GetTriplePoints(self):
#         if len(self.__TriplePoints) > 0:
#             return self.__ConvertToCoordinates(self.__TriplePoints)
#     def GetDefects(self):
#         if len(self.__Dislocations) > 0:
#             return self.__ConvertToCoordinates(self.__Dislocations)
#     def GetSkeletonPoints(self):
#         if not self.Skeletonised:
#             self.SkeletonisePoints()
#         return self.__SkeletonGrid
#     def SetSkeletonValue(self, arrPosition:np.array, intValue):
#         intDown, intAcross = np.shape(self.__SkeletonGrid)
#         if arrPosition[0] < intDown and arrPosition[1] < intAcross:
#             self.__SkeletonGrid[arrPosition[0],arrPosition[1]] = intValue
#     def SkeletonisePoints(self):
#         if not self.Skeletonised:
#             arrOutSk = self.GetExtendedGrid().astype('int')
#             arrOutSk = skeletonize(arrOutSk)
#             arrOutSk = remove_small_holes(arrOutSk, 4)
#             arrOutSk = (thin(arrOutSk)).astype('int')
#             self.__SkeletonGrid = arrOutSk.astype('int')
#             self.Skeletonised = True
#     def ClassifyGBPoints(self,m:int,blnFlagEndPoints = False)-> np.array:
#         self.SkeletonisePoints()
#         arrTotal =np.zeros(4*m)
#         intLow = int((m-1)/2)
#         intHigh = int((m+1)/2)
#         arrArgList = np.argwhere(self.__SkeletonGrid==self.__GBValue)
#         arrCurrent = np.zeros([m,m])
#         for x in arrArgList: #loop through the array positions which have GB atoms
#             arrCurrent = self.GetSkeletonPoints()[x[0]-intLow:x[0]+intHigh,x[1]-intLow:x[1]+intHigh] #sweep out a m x m square of array positions 
#             intSwaps = 0
#             if np.shape(arrCurrent) == (m,m): #centre j. This check avoids boundary points
#                 intValue = arrCurrent[0,0]
#                 arrTotal[:m ] = arrCurrent[0,:]
#                 arrTotal[m:2*m] =  arrCurrent[:,-1]
#                 arrTotal[2*m:3*m] = arrCurrent[-1,::-1]
#                 arrTotal[3*m:4*m] = arrCurrent[-1::-1,0]
#                 for k in arrTotal:
#                     if (k!= intValue): #the move has changed from grain (int 0) to grain boundary (int 1) or vice versa
#                         intSwaps += 1
#                         intValue = k
#                 if intSwaps == 6:
#                     if not (arrCurrent[0].all() == self.__GBValue or arrCurrent[-1].all() == self.__GBValue or arrCurrent[:,0].all() == self.__GBValue or  arrCurrent[:,-1].all() ==self.__GBValue):
#                         self.SetSkeletonValue(x,self.__TripleLineValue)
#                 if intSwaps < 4:
#                     if blnFlagEndPoints:
#                         self.SetSkeletonValue(x,self.__DislocationValue)
#         self.__TriplePoints = np.argwhere(self.__SkeletonGrid == self.__TripleLineValue)
#         self.__Dislocations = np.argwhere(self.__SkeletonGrid == self.__DislocationValue)
#     def FindGrainBoundaries(self)->list:
#         lstGBPoints = []
#         self.SkeletonisePoints()
#         self.ClassifyGBPoints(5, blnFlagEndPoints=False)
#         self.ClearWrapperValues()
#         arrLabels = label(self.__SkeletonGrid == self.__GBValue)
#         self.__GrainBoundaries = arrLabels
#         self.__NumberOfGrainBoundaries = len(np.unique(arrLabels))
#         for j in range(1,self.__NumberOfGrainBoundaries+1):
#             lstGBPoints.append(np.argwhere(arrLabels == j))
#         return lstGBPoints
#     def GetGrainBoundaries(self):
#         return self.__GrainBoundaries
