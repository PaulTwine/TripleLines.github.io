# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:35:27 2019

@author: twine
"""
import GeometryFunctions as gf
import numpy as np
import itertools as it
import LatticeDefinitions as ld
import re
           
class PureCell(object):
    def __init__(self,inCellNodes: np.array): 
        self._CellNodes = inCellNodes
        self._NumberOfCellNodes = len(inCellNodes)
        self._Dimensions = len(inCellNodes[0])
    def CellCentre(self)->np.array:
        return 0.5*np.ones(self.Dimensions)
    def CellNode(self, inIndex: int)->np.array:
        return self._CellNodes[inIndex]
    @property
    def CellNodes(self)->np.array:
        return self._CellNodes
    @CellNodes.setter
    def CellNodes(self, inArray: np.array):
        self._CellNodes = inArray
    @property
    def NumberOfCellNodes(self):
        return self._NumberOfCellNodes
    @property
    def Dimensions(self):
        return self._Dimensions
    def DirectionalMotif(self, intBasisVector: int)->np.array:
        inBasisVector = gf.StandardBasisVectors(self.Dimensions)[intBasisVector]
        lstDeletedIndices = []
        for i,arrNode in enumerate(self._CellNodes):
            if np.dot(inBasisVector, np.subtract(arrNode, self.CellCentre())) < 0:
                lstDeletedIndices.append(i)
        arrMotif = np.delete(self._CellNodes, lstDeletedIndices, axis=0)       
        return arrMotif
    def SnapToNode(self, inNodePoint: np.array)->np.array:
        if (np.max(inNodePoint) < 1.0 and np.min(inNodePoint) >= 0.0):
            arrDistances = np.zeros(self.NumberOfCellNodes)
            for j,arrNode in enumerate(self._CellNodes):
                arrDistances[j] = gf.RealDistance(inNodePoint, arrNode)
            intMinIndex = np.argmin(arrDistances)
        else:
            raise Exception('Cell co-ordinates must range from 0 to 1 inclusive')
        return self._CellNodes[intMinIndex] 
class PureLattice(PureCell):
    def __init__(self, inCellPositions: np.array, inCellNodes: np.array):
        PureCell.__init__(self,inCellNodes)
        self._CellPositions = inCellPositions
        self._NumberOfCells = len(inCellPositions)
        self.__MakeLatticePoints()
    @property
    def GetLatticePoints(self)-> np.array:
        return self._LatticePoints
    @property
    def CellPositions(self)->np.array:
        return self._CellPositions
    def __MakeLatticePoints(self):
        arrLatticePoints = np.empty([self.NumberOfCellNodes*self._NumberOfCells, self.Dimensions])
        for i, position in enumerate(self._CellPositions):
            for j, cell in enumerate(self._CellNodes):
                arrLatticePoints[j+i*self._NumberOfCellNodes] = np.add(position,cell)
        self._LatticePoints = np.unique(arrLatticePoints, axis=0)
    def LinearConstrainCellPositions(self,inConstraint: np.array):
        arrPoints = gf.CheckLinearConstraint(self._CellPositions, inConstraint)
        self._CellPositions = arrPoints
        self.__MakeLatticePoints()
    def LinearConstrainLatticePoints(self,inConstraint: np.array):
        arrPoints = gf.CheckLinearConstraint(self._LatticePoints, inConstraint)
        self._LatticePoints = arrPoints
    def AllAdjacentLatticePoints(self, inLatticePoint: np.array)->np.array:
        return np.add(inLatticePoint, self._CellNodes)
    def FindNearestCellNode(self, inLatticeCoordinate: np.array)->np.array:
        arrDistances = np.zeros(self.NumberOfCellNodes)
        arrLatticeCoordinate = list(map(np.round,inLatticeCoordinate))
        arrLatticePoints = self.AllAdjacentLatticePoints(arrLatticeCoordinate)
        for j,arrNode in enumerate(arrLatticePoints):
            arrDistances[j] = gf.RealDistance(arrNode, inLatticeCoordinate)
        intMinIndex = np.argmin(arrDistances)
        return self.CellNodes[intMinIndex]
class RealLattice(PureLattice):
    def __init__(self,inCellPositions: np.array, inCellNodes: np.array, inBasisVectors: np.array):
        PureLattice.__init__(self,inCellPositions,inCellNodes)
        self._BasisVectors = inBasisVectors
        self._Origin = np.zeros(len(inBasisVectors[0]))
        self._LatticeParameters =np.ones(len(inBasisVectors))
        self.MakeRealPoints()
        self.RealToLatticeMatrix()
    def GetRealCoordinate(self, inLatticeCoordinate: np.array)->np.array:
        arrRealCoordinate = self._Origin
        for j in range(len(inLatticeCoordinate)):
            arrRealCoordinate = np.add(arrRealCoordinate, np.multiply(inLatticeCoordinate[j]*self._LatticeParameters[j],self._BasisVectors[j]))
        return arrRealCoordinate         
    def RealToLatticeMatrix(self):
        arrMatrix = np.zeros([self._Dimensions, self._Dimensions])
        for count,j in enumerate(self._BasisVectors):
            arrMatrix[count] = j*self._LatticeParameters[count]
        self._RealToLatticeMatrix =np.linalg.inv(arrMatrix) 
    def GetLatticeCoordinate(self, inRealCoordinate:np.array)-> np.array:
        #return self.GetCellCoordinate(inRealCoordinate)
        self.RealToLatticeMatrix()
        arrRealCoordinate = np.add(inRealCoordinate, np.multiply(-1,self._Origin))
        arrLatticeCoordinate = np.matmul(arrRealCoordinate,self._RealToLatticeMatrix)
        arrNodeCoordinate = self.FindNearestCellNode(arrLatticeCoordinate)
        arrLatticeCoordinate = np.array(list(map(np.round,arrLatticeCoordinate)))
        return np.add(arrLatticeCoordinate,arrNodeCoordinate)
    def GetCellCoordinate(self, inRealCoordinate:np.array)-> np.array: 
        self.RealToLatticeMatrix()
        arrRealCoordinate = np.add(inRealCoordinate, np.multiply(-1,self._Origin))
        arrLatticeCoordinate = np.matmul(arrRealCoordinate,self._RealToLatticeMatrix)
        return arrLatticeCoordinate
        #return np.array(list(map(np.rint,arrLatticeCoordinate)))  
    @property
    def NumberOfBasisVectors(self)->int:
        return len(self._BasisVectors)
    @property
    def BasisVectors(self)->np.array:
        return self._BasisVectors
    def GetBasisVector(self, inVectorNumber: int)->np.array:
        return self._BasisVectors[inVectorNumber]
    def SetBasisVector(self, inVector: np.array, inVectorNumber: int):
        self._BasisVectors[inVectorNumber] = inVector
    @property
    def LatticeParameters(self)->np.array:
        return self._LatticeParameters
    @LatticeParameters.setter
    def LatticeParameters(self, inParameters: np.array):
        self._LatticeParameters = inParameters 
        self.MakeRealPoints()       
    @property
    def GetRealPoints(self)->list:
        return self._RealPoints
    def MakeRealPoints(self):
        self._RealPoints = np.array(list(map(self.GetRealCoordinate, self._LatticePoints)))
    def RotateAxes(self,inAngle: float, vctAxis: np.array):
        vctAxis = gf.NormaliseVector(vctAxis)
        vctBasis = np.zeros([self.NumberOfBasisVectors,self.Dimensions])
        for j in range(self.NumberOfBasisVectors):
            vctBasis[j]= gf.RotateVector(self.GetBasisVector(j),vctAxis,inAngle) 
        self._BasisVectors = vctBasis
        self.MakeRealPoints()
    def GetOrigin(self)->np.array:
        return self._Origin
    def SetOrigin(self, inOrigin: np.array):
        self._Origin = inOrigin
    def LinearConstrainRealPoints(self, inConstraint: np.array):
        arrPoints = gf.CheckLinearConstraint(self._RealPoints, inConstraint)
        self._RealPoints = arrPoints
        self._LatticePoints  = list(map(self.GetLatticeCoordinate, arrPoints))   
    def GetBoundingBox(self)->np.array:
        arrBoundingBox = np.zeros([self._Dimensions,2])
        for j in range(self._Dimensions):
            arrBoundingBox[j] = np.array([min(self._RealPoints[:,j]), max(self._RealPoints[:,j])])
        return arrBoundingBox
    def NearestCellPoint(self, inRealPoint: np.array)->np.array:    
        arrLatticeCoordinate = self.GetLatticeCoordinate(inRealPoint)
        arrLatticeCoordinate = list(map(np.floor, arrLatticeCoordinate))
        arrRealCellPoint = self.GetRealCoordinate(arrLatticeCoordinate)
        return list(np.array(arrRealCellPoint,dtype= float))
    def SnapToLattice(self, inRealPoint: np.array)->np.array:    
        arrLatticeCoordinate = self.GetLatticeCoordinate(inRealPoint)
        arrLatticeCoordinate = list(map(np.floor, arrLatticeCoordinate))
        arrRealLatticePoint = self.GetRealCoordinate(arrLatticeCoordinate)
        return list(np.array(arrRealLatticePoint,dtype= float))
class RealGrain(RealLattice):
    def __init__(self, inCellPositions: np.array, inCellNodes: np.array, inBasisVectors: np.array):
        RealLattice.__init__(self,inCellPositions, inCellNodes, inBasisVectors)
        self._AtomType = 1
        self.RealCellNodes()
    def RealTranslation(self, inVector: np.array):
        self.SetOrigin(inVector+self.GetOrigin())
        self.MakeRealPoints()
    def TranslateGrain(self, inVector: np.array):
        arrRealCoordinate = self.SnapToLattice(inVector)
        self.SetOrigin(arrRealCoordinate+self.GetOrigin())
        self.MakeRealPoints()
    def MatLabPlot(self):
        return zip(*self._RealPoints)
    @property 
    def GetNumberOfAtoms(self)->int:
        return len(self._RealPoints)
    @property 
    def GetAtomType(self)->int:
        return self._AtomType
    def SetAtomType(self, intAtomType: int):
        self._AtomType = intAtomType
    def RealCentre(self)->np.array:
        arrCentre = np.zeros(self.Dimensions)
        for j in self._RealPoints:
            arrCentre = np.add(arrCentre, j)
        return arrCentre/(self.GetNumberOfAtoms)
    def RotateAboutCentre(self,inAngle: float, inAxis: np.array):
        self.RotateGrain(inAngle, self.RealCentre(), inAxis)
    def RotateGrain(self, inAngle: float, inPoint: np.array, inAxis: np.array):
        self.RotateAxes(inAngle,inAxis)
        arrPointToOrigin = np.add(self._Origin, np.multiply(-1, inPoint))
        arrRotatedPoint = gf.RotateVector(arrPointToOrigin,inAxis,inAngle)
        arrOriginTranslation = np.add(np.multiply(-1,arrPointToOrigin),arrRotatedPoint) 
        self.RealTranslation(arrOriginTranslation)
    def RealCellNodes(self)->np.array:
        arrRealCellNodes = np.zeros([self.NumberOfCellNodes,self.Dimensions])
        for count,j in enumerate(self._CellNodes):
            arrRealCellNodes[count] = self.GetRealCoordinate(j)
        self._RealCellNodes = arrRealCellNodes
        return self._RealCellNodes
    def NearestNeighbourDistance(self)->float:
        self.RealCellNodes()# check that the lattice parameters haven't changed
        fltShortestDistance = min(self._LatticeParameters)
        for count, j in enumerate(self._RealCellNodes):
            for i in range(count+1, self.NumberOfCellNodes):
                fltCurrentRealDistance = gf.RealDistance(j, self._RealCellNodes[i])
                if (fltCurrentRealDistance < fltShortestDistance):
                    fltShortestDistance = fltCurrentRealDistance
        self._NearestNeighboutDistance = fltShortestDistance
        return fltShortestDistance
    def RemovePlaneOfAtoms(self, inPlane: np.array):
        arrPointsOnPlane = gf.CheckLinearEquality(self._RealPoints, inPlane)
        self._RealPoints = arrPointsOnPlane
        return arrPointsOnPlane    

    
#Generates a cuboidgrain lattice. The basis vectors can be changed to make this parallelpiped    
class CuboidGrain(RealGrain):
    def __init__(self, inDimensions: np.array, inCellNodes: np.array):
       # for j in range(len(inDimensions)):
       #     inDimensions[j] = inDimensions[j]-1
        RealGrain.__init__(self, gf.CreateCuboidLatticePoints(inDimensions), inCellNodes, gf.StandardBasisVectors(len(inDimensions))) 
        

# Pass a list of boundary vectors the last vector should be the vertical vector 
class ExtrudedPolygon(RealGrain):
    def __init__(self, inBoundaryVectors: np.array, inCellNodes: np.array, arrBasisVectors = None):
        self._Dimensions = len(inBoundaryVectors[0])
        if arrBasisVectors is None:
            arrBasisVectors = gf.StandardBasisVectors(self.Dimensions)
        self._BoundaryVectors = inBoundaryVectors
        arrPoints = gf.CreateCuboidPoints(gf.FindBoundingBox(inBoundaryVectors))
        RealGrain.__init__(self,arrPoints, inCellNodes, arrBasisVectors)
        arrFinalVector = inBoundaryVectors[len(inBoundaryVectors)-1]
        arrPointOnPlane =np.zeros([self._Dimensions])
        for j in range(len(self._BoundaryVectors)-1):
            arrPointOnPlane = np.add(arrPointOnPlane,self._BoundaryVectors[j])
            arrConstraint = gf.FindPlane(self._BoundaryVectors[j],arrFinalVector,arrPointOnPlane)
            self.LinearConstrainRealPoints(arrConstraint)
class OrientedExtrudedPolygon(ExtrudedPolygon):
    def __init__(self,inBoundaryVectors: np.array, inCellNodes: np.array,inAngle: float, inAxis: np.array, arrBasisVectors = None):
        self.__OrientationAngle = inAngle
        self.__RotationAxis = inAxis
        inBoundaryVectors = gf.RotateVectors(inAngle, inAxis, inBoundaryVectors)
        ExtrudedPolygon.__init__(self, inBoundaryVectors, inCellNodes, arrBasisVectors)
        self.RotateGrain(-inAngle,self.GetOrigin(), inAxis)
    def GetQuaternionOrientation(self)->np.array:
        return gf.GetQuaternion(self.__RotationAxis,self.__OrientationAngle)


class OrientedExtrudedHexagon(OrientedExtrudedPolygon):
    def __init__(self, intCellsLong: int, intCellsHigh: int,inCellNodes: np.array, inAngle: float, inAxis: np.array, arrBasisVectors=None):
        z = np.array([0,0,intCellsHigh])
        T0 = intCellsLong*np.array([-1/2, -1*np.sqrt(3)/2,0]) #down and left
        T1 = intCellsLong*np.array([1,0,0])
        T2 = intCellsLong*np.array([-1/2, 1*np.sqrt(3)/2, 0]) #up and left
        OrientedExtrudedPolygon.__init__(self,np.array([T1, -T0,T2,-T1, T0,-T2, z]),inCellNodes, inAngle, z)

class SimulationCell(object):
    def __init__(self, inBoxVectors: np.array):
        self.Dimensions = len(inBoxVectors[0])
        self.BoundaryTypes = ['p']*self.Dimensions #assume periodic boundary conditions as a default
        self.SetOrigin(np.zeros(self.Dimensions))
        self.GrainList = [] #list of RealGrain objects which form the simulation cell
        self.SetParallelpipedVectors(inBoxVectors)
        self.NoDuplicates = False
    def AddGrain(self,inGrain: RealGrain):
        self.GrainList.append(inGrain)
    def GetGrain(self, intGrainIndex: int):
        return self.GrainList[intGrainIndex]
    def GetNumberOfGrains(self)->int:
        return len(self.GrainList)
    def GetTotalNumberOfAtoms(self):
        if self.NoDuplicates:
            intNumberOfAtoms = len(self.__UniqueRealPoints)
        else: 
            intNumberOfAtoms = 0
            for j in self.GrainList:
                intNumberOfAtoms += j.GetNumberOfAtoms
        return intNumberOfAtoms
    def SetBoundaryTypes(self,inPosition: int, inString: str): #boundaries can be periodic 'p' or fixed 'f'
        if (inString == 'p' or inString == 'f'): 
            self.BoundaryTypes[inPosition] = inString    
    def GetNumberOfAtomTypes(self):
        lstAtomTypes = []
        for j in self.GrainList:
            intCurrentAtomType = j.GetAtomType
            if intCurrentAtomType not in lstAtomTypes: 
                lstAtomTypes.append(intCurrentAtomType)
        return len(lstAtomTypes)
    def GetMinimumSimulationBox(self)->np.array:
        arrSize = self.GrainList[0].GetBoundingBox()
        for CurrentGrain in self.GrainList:
            arrCurrentBox = CurrentGrain.GetBoundingBox()
            for coordinate, arrDimensions in enumerate(arrCurrentBox):
                if arrDimensions[0] < arrSize[coordinate][0]:
                    arrSize[coordinate][0] = arrDimensions[0]
                if arrDimensions[1] > arrSize[coordinate][1]:
                    arrSize[coordinate][1] = arrDimensions[1]
        return arrSize
    def UseMinimumSimulationBox(self):
        arrSize = self.GetMinimumSimulationBox()
        self._Size = arrSize
    def WriteLAMMPSDataFile(self,inFileName: str):        
        with open(inFileName, 'w') as fdata:
            fdata.write('#Python Generated Data File\n')
            fdata.write('{} natoms\n'.format(self.GetTotalNumberOfAtoms()))
            fdata.write('{} atom types\n'.format(self.GetNumberOfAtomTypes()))
            fdata.write('{} {} xlo xhi\n'.format(self.__xlo,self.__xhi))
            fdata.write('{} {} ylo yhi\n'.format(self.__ylo,self.__yhi))
            if self.Dimensions == 3:
                fdata.write('{} {} zlo zhi\n'.format(self.__zlo,self.__zhi))
            if self.Dimensions ==2:
                fdata.write('{}  xy \n'.format(self.__xy))
            elif self.Dimensions ==3:
                fdata.write('{}  {} {} xy xz yz \n'.format(self.__xy,self.__xz,self.__yz))
            fdata.write('\n')
            fdata.write('Atoms\n\n')
            if self.NoDuplicates:
                for j in range(len(self.__UniqueRealPoints)):
                    fdata.write('{} {} {} {} {}\n'.format(j+1,self.__AtomTypes[j], *self.__UniqueRealPoints[j]))
            else:
                count = 1
                for j in self.GrainList:
                    for position in j._RealPoints:
                        fdata.write('{} {} {} {} {}\n'.format(count,j.GetAtomType, *position))
                        count = count + 1
    def SetOrigin(self,inOrigin: np.array):
        self.__Origin = inOrigin
        self.__xlo = inOrigin[0]
        self.__ylo = inOrigin[1]
        self.__zlo = inOrigin[2]
    def GetOrigin(self):
        return self.__Origin 
    def SetParallelpipedVectors(self,inArray: np.array): 
        self.__BoxVectors = inArray
        self.__xhi = inArray[0][0] - self.__Origin[0]
        self.__xy = inArray[1][0] 
        self.__yhi = inArray[1][1] - self.__Origin[1]
        if self.Dimensions == 3:
            self.__xz = inArray[2][0] 
            self.__yz = inArray[2][1] 
            self.__zhi = inArray[2][2] - self.__Origin[2]
    def RemoveDuplicateAtoms(self)->np.array:
        lstUniqueRowindices = []
        arrAllAtoms = np.zeros([self.GetTotalNumberOfAtoms(),self.Dimensions])
        arrAllAtomTypes = np.ones([self.GetTotalNumberOfAtoms()],dtype=np.int8)
        i = 0
        for objGrain in self.GrainList:
            for fltPoint in objGrain.GetRealPoints:
                arrAllAtomTypes[i] = objGrain.GetAtomType
                arrAllAtoms[i] = fltPoint
                i = i + 1
        self.__UniqueRealPoints,lstUniqueRowindices = np.unique(arrAllAtoms,axis=0,return_index=True)
        self.__AtomTypes = arrAllAtomTypes[lstUniqueRowindices]  
        self.NoDuplicates = True
    def ApplySimulationCellConstraint(self):
        lstPlanes = []
        if self.Dimensions == 3:
            lstPlanes.append(gf.FindPlane(self.__BoxVectors[0], self.__BoxVectors[2], self.__Origin))
            lstPlanes.append(gf.FindPlane(-self.__BoxVectors[0], self.__BoxVectors[2], self.__BoxVectors[1]))
            lstPlanes.append(gf.FindPlane(-self.__BoxVectors[1], self.__BoxVectors[2], self.__Origin))
            lstPlanes.append(gf.FindPlane(self.__BoxVectors[1], self.__BoxVectors[2], self.__BoxVectors[0]))
            for j in lstPlanes:
                for k in self.GrainList:
                    k.LinearConstrainRealPoints(j)

objPureCell = PureLattice(gf.CreateCuboidPoints(np.array([[0,5],[0,4],[0,3]])),ld.FCCCell)
print(objPureCell.DirectionalMotif(0))