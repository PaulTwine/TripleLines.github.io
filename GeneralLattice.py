from cmath import pi
#from statistics import Normal
import numpy as np
import GeometryFunctions as gf
import LatticeDefinitions as ld
import scipy as sc
from sklearn.neighbors import KDTree
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify
from sympy.abc import x,y,z
from datetime import datetime
import copy as cp
import warnings
#import lammp
from decimal import Decimal
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
from scipy import linalg


class PureCell(object):
    def __init__(self,inCellNodes: np.array): 
        self.__CellNodes = inCellNodes
        self.__NumberOfCellNodes = len(inCellNodes)
        self.__Dimensions = len(inCellNodes[0])
        self.__MinimalNodeMotif = np.unique(np.mod(self.__CellNodes, np.ones([self.__NumberOfCellNodes, self.__Dimensions])),axis=0)
        arrDistanceMatrix = sc.spatial.distance_matrix(inCellNodes,inCellNodes)
        self.__NearestNode = np.min(arrDistanceMatrix[arrDistanceMatrix > 0])
    def GetNearestNodeDistance(self):
        return self.__NearestNode
    def UnitVector(self, intNumber: int)->np.array:
        arrVector = np.zeros(self.Dimensions())
        if intNumber >= 0:
            arrVector[intNumber] = 1
        else:
            arrVector[-intNumber] = -1
        return np.array(arrVector)
    def GetNumberOfNodesPerCell(self):
        return len(self.__MinimalNodeMotif)
    def GetCellCentre(self)->np.array:
        return 0.5*np.ones(self.Dimensions())
    def GetCellNode(self, inIndex: int)->np.array:
        return self.__CellNodes[inIndex]
    def GetCellNodes(self)->np.array:
        return self.__CellNodes
    def GetNumberOfCellNodes(self):
        return self.__NumberOfCellNodes
    def Dimensions(self)->int:
        return self.__Dimensions
    def GetMinimalNodeMotif(self): #assumes 
        return self.__MinimalNodeMotif
    def ApplyLatticeShift(self, inVector:np.array):
        arrNodes = np.mod(self.__MinimalNodeMotif + inVector,np.ones(3))
        self.__MinimalNodeMotif = arrNodes
    def GetCellDirectionalMotif(self, intBasisVector: int, intSign = 1)->np.array:
        inBasisVector = self.UnitVector(intBasisVector)
        if intSign ==-1:
                inBasisVector = -1*inBasisVector
        lstDeletedIndices = []
        for i,arrNode in enumerate(self.GetCellNodes()):
            if np.dot(inBasisVector, np.subtract(arrNode, self.GetCellCentre())) < 0:
                lstDeletedIndices.append(i)
        arrMotif = np.delete(self.GetCellNodes(), lstDeletedIndices, axis=0)       
        return arrMotif
    def SnapToCellNode(self, inNodePoint: np.array)->np.array:
        if (np.max(inNodePoint) < 1.0 and np.min(inNodePoint) >= 0.0):
            arrDistances = np.zeros(self.GetNumberOfCellNodes())
            for j,arrNode in enumerate(self.__CellNodes):
                arrDistances[j] = gf.RealDistance(inNodePoint, arrNode)
            intMinIndex = np.argmin(arrDistances)
        else:
            raise Exception('Cell co-ordinates must range from 0 to 1 inclusive')
        return self.__CellNodes[intMinIndex] 

class RealCell(PureCell):
    def __init__(self,inCellNodes: np.array, inLatticeParameters: np.array, inCellVectors=None):
        PureCell.__init__(self, inCellNodes)
        self.__LatticeParameters = inLatticeParameters
        if inCellVectors is None:
            self.__CellVectors = gf.StandardBasisVectors(self.Dimensions())
        else:
            self.__CellVectors = inCellVectors
        self.__RealCellVectors = np.zeros([self.Dimensions(),self.Dimensions()])
        for j in range(self.Dimensions()):
            self.__RealCellVectors[j] = inLatticeParameters[j]*self.__CellVectors[j]
        self.__RealNodes = np.ones([self.GetNumberOfCellNodes(),self.Dimensions()])
        for k in range(self.GetNumberOfCellNodes()):
            self.__RealNodes[k] = np.matmul(self.__RealCellVectors,self.GetCellNodes()[k])
        arrDistanceMatrix = sc.spatial.distance_matrix(self.__RealNodes,self.__RealNodes)
        self.__NearestNeighbourDistance = np.min(arrDistanceMatrix[arrDistanceMatrix > 0])
        arrPoints = np.matmul(self.GetMinimalNodeMotif(), np.transpose(self.__RealCellVectors))
        lstPoints = []
        for j in range(self.Dimensions()):
            lstPoints.append(arrPoints)
            lstPoints.append(arrPoints + self.__RealCellVectors[j])
            lstPoints.append(arrPoints  - self.__RealCellVectors[j])
            arrPoints = np.concatenate(lstPoints, axis = 0)
            lstPoints = []
        self.__NumberOfNeighbours = len(np.where(np.linalg.norm(arrPoints,axis=1) <= self.__NearestNeighbourDistance)[0]) -1 #include all the points close to the origin but don't the actual point
        lstPoints = []
    def GetRealCellCentre(self):
        return np.matmul(self.__RealCellVectors, 0.5*np.ones(self.Dimensions()))
    def GetUnitCellVectors(self)->np.array:
        return self.__CellVectors
    def GetRealCellVectors(self)->np.array:
        return self.__RealCellVectors
    def GetLatticeParameters(self):
        return self.__LatticeParameters
    def GetNumberOfNeighbours(self):
        return self.__NumberOfNeighbours
    def GetNearestNeighbourDistance(self):
        return self.__NearestNeighbourDistance
    def GetCellVolume(self):
        return np.abs(np.dot(self.__RealCellVectors[0], np.cross(self.__RealCellVectors[1],self.__RealCellVectors[2])))    
    def GetQuaternion(self):
        return gf.GetQuaternionFromBasisMatrix(self.__CellVectors)


class PureLattice(PureCell):
    def __init__(self,inCellNodes):
        PureCell.__init__(self, inCellNodes)
        self.__LatticePoints = []
        self.__CellPoints = []
        self.__intConstraintRound = 10
        self.__inLatticeConstraints =[]
    def SetLatticePoints(self, inLatticePoints):
        self.__LatticePoints = inLatticePoints
    def GetLatticePoints(self):
        return self.__LatticePoints
    def MakeLatticePoints(self, inCellPoints):
        arrLatticePoints = np.zeros([self.GetNumberOfNodesPerCell()*len(inCellPoints), self.Dimensions()])
        intCounter = 0
        arrNodes = self.GetMinimalNodeMotif()
        for position in inCellPoints:
            for cell in arrNodes:
                arrLatticePoints[intCounter] = np.add(position,cell)
                intCounter +=1
        return np.unique(arrLatticePoints, axis = 0)
    def CheckLatticeConstraints(self,inPoints: np.array, fltTolerance=1e-5)-> np.array: #returns indices to delete   
        lstIndices = []
        for j in self.__LatticeConstraints:
            arrPositions = np.subtract(np.matmul(inPoints, np.transpose(j[:-1])), j[-1])
            arrClosed = np.where(np.round(arrPositions,self.__intConstraintRound) > fltTolerance)[0]
            lstIndices.append(arrClosed)
        return np.unique(np.concatenate(lstIndices))
    #FindBoxConstraint only works for linear constraints. 
    #Searches for all the vertices where three constraints #simultaneously apply and then finds the points furthest from the origin.        
    def FindBoxConstraints(self,inConstraints: np.array, fltTolerance = 1e-5)->np.array:
        intLength = len(inConstraints)
        intCombinations = int(np.math.factorial(intLength)/(np.math.factorial(3)*np.math.factorial(intLength-3)))
        arrMatrix = np.zeros([3,4])
        arrPoints = np.zeros([intCombinations, 3])
        arrRanges = np.zeros([3,2])
        counter = 0
        for i in range(intLength):
            for j in range(i+1,intLength):
                for k in range(j+1,intLength):
                    arrMatrix[0] = inConstraints[i]
                    arrMatrix[1] = inConstraints[j]
                    arrMatrix[2] = inConstraints[k]
                    if abs(np.linalg.det(arrMatrix[:,:-1])) > fltTolerance:
                        arrPoints[counter] = np.matmul(np.linalg.inv(arrMatrix[:,:-1]),arrMatrix[:,-1])
                        counter += 1
                    else:
                        arrPoints = np.delete(arrPoints, counter, axis=0)
        for j in range(len(arrRanges)):
            arrRanges[j,0] = np.min(arrPoints[:,j])
            arrRanges[j,1] = np.max(arrPoints[:,j])
        return(arrRanges)
    def SetLatticeConstraints(self, inLatticeConstraints):
        self.__LatticeConstraints = inLatticeConstraints
    def GetLatticeConstraints(self):
        return self.__LatticeConstraints
    def GetLatticeBoundaryPoints(self):
        arrBoundaryPoints = gf.GetBoundaryPoints(self.__LatticePoints, self.GetNumberOfNeighbours(),1.05*self.GetNearestNodeDistance())
        return self.__LatticePoints[arrBoundaryPoints]
    def GetNumberOfLatticePoints(self):
        return len(self.__LatticePoints)
    def DeleteLatticePoints(self, lstDeletedIndices):
        self.__LatticePoints  = np.delete(self.__LatticePoints, lstDeletedIndices, axis=0)
    def GetConstraintTypes(self):
        return self.__ConstraintTypes
    def FindLatticeDuplicates(self, inArrayConstraints: np.array):
        lstIndices = []
        arrPoints = np.copy(self.__LatticePoints)
        for j in inArrayConstraints:
            arrPositions = np.subtract(np.matmul(self.__LatticePoints, np.transpose(j[:-1])), j[-1])
            arrEndPoints = np.where(abs(np.round(arrPositions,self.__intConstraintRound)) < 1e-5)[0]
            arrPoints[arrEndPoints] =  arrPoints[arrEndPoints] - j[0:-1]*j[-1]
        arrUniqueRows = np.unique(np.round(arrPoints,1), axis=0, return_index=True)[1]
        lstIndices = list(set(range(self.GetNumberOfLatticePoints())).difference(arrUniqueRows.tolist()))
        return lstIndices

class GeneralLattice(PureLattice,RealCell):
    def __init__(self,inBasisVectors:np.array,inCellNodes: np.array,inLatticeParameters:np.array,inOrigin: np.array,inCellBasis = None):
        PureLattice.__init__(self,inCellNodes)
        RealCell.__init__(self, inCellNodes,inLatticeParameters, inCellBasis)
        self.__UnitBasisVectors  = inBasisVectors # Cartesian basis vectors for the lattice
        self.__RealPoints = []
        self.__Origin = inOrigin
        self.__LatticeParameters = inLatticeParameters
        self.__RealBasisVectors = np.zeros([self.Dimensions(),self.Dimensions()])
        self.__LinearConstraints = []
        self.__ConstrainType = []
        for j in range(self.Dimensions()):
            self.__RealBasisVectors[j] = inLatticeParameters[j]*inBasisVectors[j]
         #default RemovedBoundaryPoints = []
        self.__blnFoundBoundaryPoints = False
        self.__Periodicity = ['p','p','p']
    def TranslateGrain(self, inVector):
        self.__RealPoints = self.__RealPoints+inVector
        self.__Origin = self.__Origin + inVector
    def SetPeriodicity(self, inList):
        self.__Periodicity = inList
    def GetPeriodicity(self, intIndex = None):
        if intIndex is None:
            return self.__Periodicity
        else:
            return self.__Periodicity[intIndex]
    def GetConstraintRounding(self):
        return self.__intConstraintRound
    def SetConstraintRounding(self, intValue: int):
        self.__intConstraintRound = intValue #take care to call MakeRealPoints() for this to take effect
    def GetUnitBasisVectors(self)->np.array:
        return self.__UnitBasisVectors
    def GetRealBasisVectors(self)->np.array:
        return self.__RealBasisVectors
    def FindRealPointIndices(self, inRealPoints: np.array)->list:
        arrCheck = (self.__RealPoints[:, None] == inRealPoints).all(-1).any(-1)
        lstIndices = list(arrCheck == True)
        return lstIndices
    def GetNumberOfPoints(self):
        return len(self.__RealPoints)
    def RemovePoints(self, inRealPoints: np.array):
        lstIndices = self.FindRealPointIndices(inRealPoints)
        self.DeletePoints(lstIndices)
    def DeletePoints(self,lstDeletedIndices: list):
        self.DeleteLatticePoints(lstDeletedIndices)
        self.__RealPoints = np.delete(self.__RealPoints, lstDeletedIndices, axis=0) 
    def GetRealPoints(self)->np.array: #if points on the boundary have been removed don't include them unless blnRemoved is set to false
        return self.__RealPoints
    def MakeRealPoints(self, inClosedConstraints: np.array):
        #assumes constraints are closed (e.g. includes boundary points) To change this call
        #SetOpenConstraints(arrPositions) and an array of which constraints are open
        self.__LinearConstraints = inClosedConstraints
        self.GenerateLatticeConstraints(inClosedConstraints)
        arrBounds = self.FindBoxConstraints(self.GetLatticeConstraints())
        arrBounds[:,0] = np.floor(arrBounds[:,0]) -np.ones(self.Dimensions())
        arrBounds[:,1] = np.ceil(arrBounds[:,1]) +np.ones(self.Dimensions()) #add one extra lattice points in each 
        #abstract direction as using the minimal node motif
        arrCellPoints = np.array(gf.CreateCuboidPoints(arrBounds))
        arrLatticePoints = self.MakeLatticePoints(arrCellPoints)
        arrLatticePoints = np.delete(arrLatticePoints, self.CheckLatticeConstraints(arrLatticePoints), axis = 0)
        self.GenerateRealPoints(arrLatticePoints)
    def GenerateRealPoints(self, inLatticePoints = None):
        self.SetLatticePoints(inLatticePoints)
        arrRealPoints = np.round(np.matmul(inLatticePoints, self.GetRealCellVectors()),10)
        self.__RealPoints = np.round(np.matmul(arrRealPoints, self.GetUnitBasisVectors()),10)
        self.__RealPoints = np.add(self.__Origin, self.__RealPoints)
    def CheckRealLinearConstraints(self,inPoints: np.array)-> np.array: #returns indices to delete for real coordinates  
        lstIndices = []
        for j in self.__LinearConstraints:
            arrPositions = np.subtract(np.matmul(inPoints, np.transpose(j[:-1])),j[-1])
            arrClosed = np.where(np.round(arrPositions,self.__intConstraintRound) > 0)[0]
            lstIndices.append(arrClosed)
        return np.unique(np.concatenate(lstIndices))       
    def GetNumberOfPoints(self)->int:
        return len(self.__RealPoints)
    def GetOrigin(self)->np.array:
        return self.__Origin
    def SetOrigin(self, inArray: np.array):
        self.__Origin = inArray
    def MatLabPlot(self):
        return tuple(zip(*self.GetRealPoints()))
    def LinearConstrainRealPoints(self, inConstraint: np.array):
        lstDeletedIndices = gf.CheckLinearConstraint(self.__RealPoints, inConstraint)
        self.DeletePoints(lstDeletedIndices)
    def RemovePlaneOfAtoms(self, inPlane: np.array):
        lstDeletedIndices = gf.CheckLinearEquality(self.__RealPoints, inPlane, 0.01)
        self.DeletePoints(lstDeletedIndices)
    def ApplyGeneralConstraint(self,strFunction, strVariables='[x,y,z]',fltTolerance = 1e-5, strDomain = ''): #default scalar value is less than or equal to 0 if "inside" the region
        lstVariables = parse_expr(strVariables)
        fltFunction = lambdify(lstVariables,parse_expr(strFunction))
        arrFunction = lambda X : fltFunction(X[0],X[1],X[2])
        arrLess = np.array(list(map(arrFunction, self.__RealPoints)))
        if len(strDomain) >0 :
            fltDomain = lambdify(lstVariables,parse_expr(strDomain))
            arrDomain = lambda X : fltDomain(X[0],X[1],X[2])
            arrDomain = np.array(list(map(arrDomain, self.__RealPoints)))
            lstDeletedIndices = np.where((arrLess > fltTolerance) & (arrDomain >= 0))[0]
        else:
            lstDeletedIndices = np.where(arrLess > fltTolerance)[0]
        self.DeletePoints(lstDeletedIndices)
    def GetQuaternionOrientation(self)->np.array:
        return gf.FCCQuaternionEquivalence(gf.GetQuaternionFromBasisMatrix(np.transpose(self.GetUnitBasisVectors())))     
    def GetLinearConstraints(self):
        return self.__LinearConstraints
    def FindBoundaryPoints(self, inPeriodicVectors = None):
        self.__BoundaryPointIndices = gf.GetBoundaryPoints(self.GetRealPoints(), self.GetNumberOfNeighbours(), 1.05*self.GetNearestNeighbourDistance(),inPeriodicVectors)
        lstInteriorPoints = list(set(range(self.GetNumberOfPoints())).difference(list(self.__BoundaryPointIndices)))
        self.__InteriorPointIndices = lstInteriorPoints
    def GetInteriorPoints(self, inPeriodicVectors = None):
        self.FindBoundaryPoints(inPeriodicVectors)
        return self.__RealPoints[self.__InteriorPointIndices]
    def GetNumberOfInteriorPoints(self, inPeriodicVectors=None):
        self.FindBoundaryPoints(inPeriodicVectors)
        return len(self.__InteriorPointIndices)
    def GetBoundaryPoints(self, inPeriodicVectors = None):
        self.FindBoundaryPoints(inPeriodicVectors)
        return self.__RealPoints[self.__BoundaryPointIndices]
    def GetBoundaryIndices(self, inPeriodicVectors = None):
        self.FindBoundaryPoints(inPeriodicVectors)    
        return self.__BoundaryPointIndices
    def FoundBoundaries(self):
        return self.__blnFoundBoundaryPoints
    def SetOpenConstraints(self, inRealConstraints: np.array, fltTolerance=1e-3):
        lstIndices = []
        lstConstraints = []
        intCounter = 0
        for i in inRealConstraints:
            if self.GetPeriodicity(intCounter) == 'p':
                j = self.ConvertRealToLatticeConstraint(i)
                lstConstraints.append(j)
                arrPositions = np.subtract(np.matmul(self.GetLatticePoints(), np.transpose(j[:-1])), j[-1])
                arrClosed = np.where(np.abs(arrPositions) < fltTolerance)[0]
                lstIndices.append(arrClosed)
            intCounter +=1
        if len(lstIndices) > 0:
            self.__OpenConstraints = np.vstack(lstConstraints)
            return list(np.unique(np.concatenate(lstIndices)))
        else: 
            self.__OpenConstraints = []
            return []
    def FindPeriodicDuplicates(self, inCellVectors: np.array):
        return gf.FindDuplicates(self.GetRealPoints(),inCellVectors, 1e-5, self.GetPeriodicity())
    def GenerateLatticeConstraints(self, inConstraints: np.array):
        rtnArray = np.zeros([len(inConstraints),len(inConstraints[0])])
        for i in range(len(inConstraints)):
            rtnArray[i] = self.ConvertRealToLatticeConstraint(inConstraints[i])
        self.SetLatticeConstraints(rtnArray)
    def ConvertRealToLatticeConstraint(self, inConstraint)->np.array: #assumes the real origin coincides with lattice origin
        rtnArray = np.zeros(len(inConstraint))
        tmpArray = np.zeros([3])
        arrVector = inConstraint[:-1]
        fltLength = np.linalg.norm(arrVector)
        arrVector = arrVector/fltLength
        arrConstraint = inConstraint[3]*arrVector/fltLength**2
        arrVector = np.matmul(arrVector, np.linalg.inv(self.GetRealBasisVectors()))
        arrConstraint = np.matmul(arrConstraint, np.linalg.inv(self.GetRealBasisVectors()))
        for k in range(3):
            rtnArray[k] = np.dot(arrVector,self.GetUnitCellVectors()[k]) # generally a non-Carteisan Basis
            tmpArray[k] = np.dot(arrConstraint, self.GetUnitCellVectors()[k])
        fltLength = np.linalg.norm(rtnArray[:3])
        rtnArray[:3] = rtnArray[:3]/fltLength 
        rtnArray[3] = np.round(gf.InnerProduct(rtnArray[:3], tmpArray,np.linalg.inv(self.GetUnitCellVectors())),10)
        return rtnArray
       
        
class GeneralGrain(GeneralLattice):
    def __init__(self,inBasisVectors:np.array,inCellNodes: np.array,inLatticeParameters:np.array,inOrigin: np.array,inCellBasis = None):
        self.__AtomType = 1
        self.__VacancyIndices = []
        self.__InteriorAtomIndices = []
        GeneralLattice.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
    def MakeVacancySpace(self, inPoint: np.array, intNumber: int):
        self.__SpatialPoints = KDTree(self.GetAtomPositions())
        arrPoints = self.__SpatialPoints.query(np.array(inPoint),intNumber)[1]
        lstIndices = self.FindRealPointIndices(arrPoints)
        self.AddVacancies(lstIndices)
    def MakeNVacancies(self, intNumber, fltSeparation):
        self.__SpatialPoints = KDTree(self.GetAtomPositions())
        lstPoints = []
        lstPositions = []
        self.FindBoundaryPoints()
        intCounter = 0
        intIndex = 0
        while len(lstPoints) < intNumber:
            arrPoint = self.GetAtomPositions()[intIndex]
            if len(lstPoints) > 0:
                arrDistanceMatrix= sc.spatial.distance_matrix(np.array([arrPoint]),np.vstack(lstPoints))
                lstPositions = np.where(arrDistanceMatrix <= fltSeparation)[0]
                if len(lstPositions) == 0:
                    lstPoints.append(arrPoint)
                    self.AddVacancies([intIndex])   
            else:
                lstPoints.append(arrPoint)
                self.AddVacancies([intIndex])
            intCounter +=1
    def SetOpenBoundaryPoints(self,inVectors: np.array):
        if len(inVectors) > 0:
            lstDeletedIndices = self.SetOpenConstraints(inVectors)
            self.AddVacancies(lstDeletedIndices)
    def AddVacancies(self, inList): #pass the row indices of the real points
        self.__VacancyIndices.extend(inList)
        self.__VacancyIndices =list(np.unique(self.__VacancyIndices)) 
    def GetNumberOfVacancies(self):
        return len(self.__VacancyIndices)
    def GetVacancies(self):
        return self.GetRealPoints()[self.__VacancyIndices]
    def GetAtomPositions(self):
        setAll = set(range(self.GetNumberOfPoints()))
        lstRows = list(setAll.difference(self.__VacancyIndices))
        return self.GetRealPoints()[lstRows]
    def GetInteriorAtomPositions(self,inPeriodicVectors=None):
        setAll = set(range(self.GetNumberOfPoints()))
        setAll = setAll.difference(self.GetBoundaryAtomIndices(inPeriodicVectors))
        lstRows = list(setAll.difference(self.__VacancyIndices))
        self.__InteriorAtomIndices = lstRows
        return self.GetRealPoints()[lstRows]
    def GetNumberOfInteriorAtoms(self,inPeriodicVectors=None):
        if len(self.__InteriorAtomIndices) ==0:
            self.GetInteriorAtomPositions(inPeriodicVectors)
        return len(self.__InteriorAtomIndices)
    def GetNumberOfAtoms(self):
        return self.GetNumberOfPoints()-self.GetNumberOfVacancies()
    def GetAtomType(self)->int:
        return self.__AtomType
    def SetAtomType(self, inInt):
        self.__AtomType = inInt
    def GetBoundaryAtoms(self, inPeriodicVectors = None):
        lstRows = self.GetBoundaryAtomIndices(inPeriodicVectors)
        return self.GetRealPoints()[lstRows]
    def GetBoundaryAtomIndices(self, inPeriodicVectors = None):
        self.FindBoundaryPoints(inPeriodicVectors)
        setBoundaryPoints = set(self.GetBoundaryIndices(inPeriodicVectors))
        lstRows = list(setBoundaryPoints.difference(self.__VacancyIndices))
        return lstRows
    def GetNumberOfBoundaryAtoms(self, inPeriodicVectors=None):
        return len(self.GetBoundaryAtoms(inPeriodicVectors))


class IrrregularExtrudedGrain(GeneralGrain):
    def __init__(self, arrEdgeBaseVectors: np.array, fltHeight: float, inBasisVectors: np.array, inCellNodes: np.array ,inLatticeParameters: np.array, inOrigin: np.array, inCellBasis= None):
        if np.round(np.linalg.norm(np.sum(arrEdgeBaseVectors,axis=0)),10) == 0:
            intConstraints = len(arrEdgeBaseVectors)
            arrConstraints = np.zeros([intConstraints+2,4])
            arrVectorSum = np.zeros(3)
            for j in range(intConstraints):
                arrEdge = arrEdgeBaseVectors[j]
                arrOut = gf.NormaliseVector(np.cross(arrEdge,np.array([0,0,1])))
                arrConstraints[j,:3] = arrOut
                fltDistance = np.dot(arrOut,arrVectorSum)
                arrConstraints[j,3] = fltDistance
                arrVectorSum += arrEdge
            arrConstraints[-2,:3] = np.array([0,0,1])
            arrConstraints[-2,3] = fltHeight
            arrConstraints[-1,:3] = -np.array([0,0,1])
            arrConstraints[-1,3] = 0
            GeneralGrain.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
            self.MakeRealPoints(arrConstraints)
        else:
            raise('Base perimeter is not closed')




class ExtrudedRectangle(GeneralGrain):
    def __init__(self, fltLength: float, fltWidth: float, fltHeight: float, inBasisVectors: np.array, inCellNodes: np.array ,inLatticeParameters: np.array, inOrigin: np.array, inCellBasis= None):
        arrConstraints = np.zeros([6,4])
        arrConstraints[0] = np.array([1,0,0,fltLength])
        arrConstraints[1] = np.array([-1,0,0,0])
        arrConstraints[2] = np.array([0,1,0,fltWidth])
        arrConstraints[3] = np.array([0,-1,0,0])
        arrConstraints[4] = np.array([0,0,1,fltHeight])
        arrConstraints[5] = np.array([0,0,-1,0])
        GeneralGrain.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
        self.MakeRealPoints(arrConstraints)

class ExtrudedParallelogram(GeneralGrain):
    def __init__(self, arrLength: np.array, arrWidth: float, fltHeight: float, inBasisVectors: np.array, inCellNodes: np.array ,inLatticeParameters: np.array, inOrigin: np.array, inCellBasis= None):
        arrZ = np.array([0,0,1])
        arrConstraints = np.zeros([6,4])
        arrNormal = gf.NormaliseVector(np.cross(arrLength, arrZ))
        arrConstraints[0,:3] = arrNormal
        arrConstraints[0,3] = 0
        arrNormal = -gf.NormaliseVector(np.cross(arrLength, arrZ))
        arrConstraints[1,:3] = arrNormal
        arrConstraints[1,3] = np.dot(arrWidth,arrNormal)
        arrNormal = -gf.NormaliseVector(np.cross(arrWidth, arrZ))
        arrConstraints[2,:3] = arrNormal
        arrConstraints[2,3] = 0
        arrNormal = gf.NormaliseVector(np.cross(arrWidth, arrZ))
        arrConstraints[3,:3] = arrNormal
        arrConstraints[3,3] = np.dot(arrLength,arrNormal)
        arrConstraints[4,:3] = -arrZ
        arrConstraints[4,3] = 0
        arrConstraints[5,:3] = arrZ
        arrConstraints[5,3] = fltHeight
        GeneralGrain.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
        self.MakeRealPoints(arrConstraints)


class ExtrudedRegularPolygon(GeneralGrain):
    def __init__(self, fltSideLength: float, fltHeight: float, intNumberOfSides: int, inBasisVectors: np.array, inCellNodes: np.array, inLatticeParameters: np.array, inOrigin: np.array, inCellBasis=None):
        intDimensions = len(inBasisVectors[0])
        arrConstraints = np.zeros([intNumberOfSides + 2,intDimensions+1])
        arrNormalVector = np.zeros(intDimensions)
        arrSideVector = np.zeros(intDimensions)
        arrVerticalVector = np.zeros(intDimensions)
        arrNormalVector[1] = -1
        arrSideVector[0] = fltSideLength
        arrVerticalVector[-1] = 1
        fltAngle = 2*np.pi/intNumberOfSides
        arrNextPoint = np.zeros([intDimensions])
        for j in range(intNumberOfSides):
            arrNextPoint += arrSideVector
            for k in range(len(arrConstraints[0])-1):
                arrConstraints[j,k] = arrNormalVector[k]
            arrConstraints[j, -1] = np.dot(arrNormalVector, arrNextPoint)
            arrNormalVector = gf.RotateVector(arrNormalVector,arrVerticalVector, fltAngle)
            arrSideVector = gf.RotateVector(arrSideVector,arrVerticalVector, fltAngle)
        arrConstraints[-2,-2] = -1
        arrConstraints[-2,-1] = 0
        arrConstraints[-1,-2] = 1
        arrConstraints[-1,-1] = fltHeight
        GeneralGrain.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
        self.MakeRealPoints(arrConstraints)

class ExtrudedCylinder(GeneralGrain):
    def __init__(self, fltRadius: float, fltHeight: float, inBasisVectors: np.array, inCellNodes: np.array, inLatticeParameters: np.array, inOrigin: np.array, inCellBasis=None):
        arrConstraints = np.zeros([6,4])
        arrConstraints[0] = np.array([1,0,0,fltRadius])
        arrConstraints[1] = np.array([0,1,0,fltRadius])
        arrConstraints[2] = np.array([-1,0,0,fltRadius])
        arrConstraints[3] = np.array([0,-1,-0,fltRadius])
        arrConstraints[4] = np.array([0,0,1,fltHeight])
        arrConstraints[5] = np.array([0,0,-1,0])
        GeneralGrain.__init__(self,inBasisVectors,inCellNodes,inLatticeParameters,inOrigin,inCellBasis)
        self.MakeRealPoints(arrConstraints)
        strCylinder = gf.ParseConic([0,0],[fltRadius,fltRadius],[2,2])
        self.ApplyGeneralConstraint(strCylinder)
        
class ParallelopiedGrain(GeneralGrain):
    def __init__(self, arrEdgeVectors: np.array, inBasisVectors: np.array, inCellNodes: np.array, inLatticeParameters: np.array, inOrigin: np.array, inCellBasis=None):
        arrConstraints = gf.FindConstraintsFromBasisVectors(arrEdgeVectors)
        arrConstraints2 = -arrConstraints
        arrConstraints2[:,-1] = np.zeros(3)
        arrConstraints = np.append(arrConstraints, arrConstraints2, axis=0)
        GeneralGrain.__init__(self,inBasisVectors, inCellNodes,inLatticeParameters,inOrigin,inCellBasis)
        self.MakeRealPoints(arrConstraints)

class IrregularSlantedGrain(GeneralGrain):
    def __init__(self, arrBaseEdgeVectors: np.array, inAxis: np.array, inBasisVectors: np.array, inCellNodes: np.array, inLatticeParameters: np.array, inOrigin: np.array, inCellBasis=None):
        if np.round(np.linalg.norm(np.sum(arrBaseEdgeVectors,axis=0)),10) == 0:
            intConstraints = len(arrBaseEdgeVectors)
            arrConstraints = np.zeros([intConstraints+2,4])
            arrVectorSum = np.zeros(3)
            for j in range(intConstraints):
                arrEdge = arrBaseEdgeVectors[j]
                arrOut = gf.NormaliseVector(np.cross(arrEdge,inAxis))
                arrConstraints[j,:3] = arrOut
                fltDistance = np.dot(arrOut,arrVectorSum)
                arrConstraints[j,3] = fltDistance
                arrVectorSum += arrEdge
            arrConstraints[-2,:3] = gf.NormaliseVector(inAxis)
            arrConstraints[-2,3] = np.linalg.norm(inAxis)
            arrConstraints[-1,:3] = -gf.NormaliseVector(inAxis)
            arrConstraints[-1,3] = 0
            GeneralGrain.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
            self.MakeRealPoints(arrConstraints)
        else:
            raise('Base perimeter is not closed')
               
class BaseSuperCell(object):
    def __init__(self,inBasisVectors: np.array, lstBoundaryTypes: list):
        self.__BasisVectors = inBasisVectors
        self.__Dimensions = np.shape(inBasisVectors)[0]
        self.__BoundaryTypes = lstBoundaryTypes
        self.__InverseMatrix = np.linalg.inv(inBasisVectors)
    def GetBasisVectors(self):
        return self.__BasisVectors
    def GetDimensions(self):
        return self.__Dimensions
    def GetBoundaryTypes(self):
        return self.__BoundaryTypes
    def GetInverseMatrix(self): ##multiplied by a real point gives a SuperCell coordinate
        return self.__InverseMatrix
    def WrapVectorIntoSimulationBox(self, inVector: np.array)->np.array:
        return gf.WrapVectorIntoSimulationCell(self.__BasisVectors, self.__InverseMatrix, inVector)


class SimulationCell(object):
    def __init__(self, inBoxVectors: np.array):
        self.Dimensions = len(inBoxVectors[0])
        self.__BoundaryTypes = ['p']*self.Dimensions #assume periodic boundary conditions as a default
        self.SetOrigin(np.zeros(self.Dimensions))
        self.dctGrains = dict() #dictionary of RealGrain objects which form the simulation cell
        self.SetParallelpipedVectors(inBoxVectors)
        self.blnPointsAreWrapped = False
        self.__FileHeader = ''
        self.GrainList = []
        self.__AllAtomPositions = []
        self.__AllAtomTypes = []
        self.__GrainAtomPositions = []
        self.__GrainAtomTypes = []
        self.__GrainAtomPositions = []
        self.__NonGrainAtomPositions = []
        self.__NonGrainAtomTypes = []
    def SetAllAtomPositions(self):
        self.SetGrainAtoms()
        if len(self.__NonGrainAtomPositions) > 0:
            arrPoints = np.append(self.__GrainAtomPositions , self.__NonGrainAtomPositions,axis=0)
        else:
            arrPoints = self.__GrainAtomPositions
        arrDelete = gf.FindDuplicates(arrPoints,self.__BasisVectors,1e-5,self.__BoundaryTypes)
        self.__AllAtomPositions = arrPoints
        if len(self.__NonGrainAtomTypes) > 0:
            self.__AllAtomTypes = np.append(self.__GrainAtomTypes , self.__NonGrainAtomTypes,axis=0)
        else:
            self.__AllAtomTypes = self.__GrainAtomTypes
        if len(arrDelete) > 0:
            self.__AllAtomPositions = np.delete(self.__AllAtomPositions, arrDelete, axis=0)
            self.__AllAtomTypes = np.delete(self.__AllAtomTypes, arrDelete, axis=0)
    def AddGrain(self,inGrain, strName = None):
        if strName is None:
            strName = str(len(self.dctGrains.keys())+1)
        self.dctGrains[strName] = inGrain
        self.GrainList = list(self.dctGrains.keys())
    def RemoveGrain(self, strName: str):
        self.dctGrains.popitem(strName)
    def RemoveAllGrains(self):
        self.GrainList = []
        self.dctGrains = dict()
    def GetGrain(self, strName: str):
        return self.dctGrains[strName]
    def GetNumberOfGrains(self)->int:
        return len(self.GrainList)
    def GetUpdatedAtomNumbers(self):
        intNumberOfAtoms = 0
        if len(self.__GrainAtomPositions) ==0:
            for j in self.GrainList:
                intNumberOfAtoms += self.GetGrain(j).GetNumberOfInteriorAtoms(self.__BasisVectors)
        else:
            intNumberOfAtoms = len(self.__GrainAtomPositions) 
        if len(self.__NonGrainAtomPositions) ==0:
            for k in self.GrainList:
                lstGBPoints = []
                lstGBPoints.append(self.GetGrain(k).GetBoundaryAtoms(self.__BasisVectors))
            arrGBPoints = np.vstack(lstGBPoints)
            intNumberOfAtoms += len(np.unique(np.round(arrGBPoints,5),axis=0))
        else:
            intNumberOfAtoms += len(self.__NonGrainAtomPositions)
        return intNumberOfAtoms
    def GetTotalNumberOfAtoms(self):
        if len(self.__AllAtomPositions) ==0:
            self.SetAllAtomPositions()
        return len(self.__AllAtomPositions)
    def GetRealBasisVectors(self):
        return self.__BasisVectors
    def SetBoundaryTypes(self,inList: list): #boundaries can be periodic 'p' or fixed 'f'
        self.BoundaryTypes = inList  
    def GetAllAtomTypes(self):
        lstAtomTypes = []
        for j in self.GrainList:
            intCurrentAtomType = self.GetGrain(j).GetAtomType()
            if intCurrentAtomType not in lstAtomTypes: 
                lstAtomTypes.append(intCurrentAtomType)
        return lstAtomTypes  
    def GetNumberOfAtomTypes(self):
        return len(self.GetAllAtomTypes())
    def SetFileHeader(self, inString: str):
        self.__FileHeader = inString
    def WrapVectorIntoSimulationBox(self, inVector: np.array)->np.array:
        return gf.WrapVectorIntoSimulationCell(self.__BasisVectors, inVector)
    def WriteLAMMPSDataFile(self,inFileName: str):        
        now = datetime.now()
        strDateTime = now.strftime("%d/%m/%Y %H:%M:%S")
        self.SetAllAtomPositions()
        with open(inFileName, 'w') as fdata:
            fdata.write('## ' + strDateTime + ' ' + self.__FileHeader + '\n')
            fdata.write('{} atoms\n'.format(self.GetTotalNumberOfAtoms()))
            fdata.write('{} atom types\n'.format(self.GetNumberOfAtomTypes()))
            fdata.write('{} {} xlo xhi\n'.format(self.__xlo,self.__xhi))
            fdata.write('{} {} ylo yhi\n'.format(self.__ylo,self.__yhi))
            fdata.write('{} {} zlo zhi\n'.format(self.__zlo,self.__zhi))
            if not self.__blnCuboid:   
                fdata.write('{}  {} {} xy xz yz \n'.format(self.__xy,self.__xz,self.__yz))
            fdata.write('\n')
            fdata.write('Atoms\n\n')
            for i in range(len(self.__AllAtomPositions)):
                fdata.write('{} {} {} {} {}\n'.format(i+1,self.__AllAtomTypes[i].astype('int'), *self.__AllAtomPositions[i]))          
              
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
            self.__BasisVectors = np.array([[self.__xhi,0,0],[self.__xy, self.__yhi,0],[self.__xz, self.__yz,self.__zhi]])
            lstBasisVectors = []
            for j in self.__BasisVectors:
                lstBasisVectors.append(gf.NormaliseVector(j))
            self.__UnitBasisVectors = np.array(lstBasisVectors)
            self.__InverseUnitBasis = np.linalg.inv(self.__UnitBasisVectors)
            self.__InverseBasis = np.linalg.inv(self.__BasisVectors)
            arrNonZero = np.argwhere(self.__BasisVectors !=0)
            if len(arrNonZero) ==3:
                self.__blnCuboid = True
            else:
                self.__blnCuboid = False
    def GetCentre(self):
        return 0.5*np.sum(self.__BasisVectors,axis=0)
    def RemoveAtomsOutsideSimulationCell(self):
        arrAllAtoms = np.zeros([self.GetUpdatedAtomNumbers(),self.Dimensions])
        arrAllAtomTypes = np.ones([self.GetUpdatedAtomNumbers()],dtype=np.int8)
        i = 0
        for j in self.GrainList:
            arrPoints = self.GetGrain(j).GetAtomPositions()
            for fltPoint in arrPoints:
                arrAllAtomTypes[i] = self.GetGrain(j).GetAtomType()
                arrAllAtoms[i] = fltPoint
                i = i + 1
        lstRows = gf.RemoveVectorsOutsideSimulationCell(self.__BasisVectors,arrAllAtoms)
        self.__GrainAtomPositions = arrAllAtoms[lstRows]
        self.__GrainAtomTypes = arrAllAtomTypes[lstRows]
    def UpdateAtomsPositions(self):
        lstGrainAtoms = []
        lstGrainAtomTypes = []
        lstGBPoints = []
        lstGBAtomTypes = []
        if len(self.__GrainAtomPositions) == 0: #only update this first time
            for j in self.GrainList:
                lstGrainAtoms.append(self.WrapVectorIntoSimulationBox(self.GetGrain(j).GetInteriorAtomPositions(self.__BasisVectors)))
                lstGrainAtomTypes.append(np.ones(self.GetGrain(j).GetNumberOfInteriorAtoms(self.__BasisVectors))*self.GetGrain(j).GetAtomType())    
            self.__GrainAtomPositions = np.vstack(lstGrainAtoms)
            self.__GrainAtomTypes = np.concatenate(lstGrainAtomTypes,axis=0).astype('int')
        if len(self.__NonGrainAtomPositions) ==0: #only update this first time
            for k in self.GrainList:
                lstGBPoints.append(self.WrapVectorIntoSimulationBox(self.GetGrain(k).GetBoundaryAtoms(self.__BasisVectors)))
                lstGBAtomTypes.append(np.ones(self.GetGrain(k).GetNumberOfBoundaryAtoms(self.__BasisVectors))*self.GetGrain(k).GetAtomType())
            self.__NonGrainAtomPositions = np.vstack(lstGBPoints)
            self.__NonGrainAtomTypes = np.concatenate(lstGBAtomTypes,axis=0).astype('int')
        self.__AllAtomPositions = np.append(self.__NonGrainAtomPositions, self.__GrainAtomPositions,axis=0)
    def WrapAllAtomsIntoSimulationCell(self, intRound=5)->np.array:
        lstUniqueRowIndices = []
        self.UpdateAtomsPositions()
        arrRounded = np.round(self.WrapVectorIntoSimulationBox(self.__AllAtomPositions), intRound)
        self.__AllAtomPositions = self.RemoveRealDuplicates(arrRounded)
        lstUniqueRowIndices = np.unique(self.__AllAtomPositions,axis=0, return_index = True)[1]
        if len(lstUniqueRowIndices) < len(self.__GrainAtomPositions):
            warnings.warn(str(self.GetUpdatedAtomNumbers() - len(lstUniqueRowIndices)) + ' duplicate atoms detected within simulation cell and have been removed.')
            self.__GrainAtomPositions = self.__GrainAtomPositions[lstUniqueRowIndices]
            self.__GrainAtomTypes = self.__GrainAtomTypes[lstUniqueRowIndices]
        self.blnPointsAreWrapped = True
    def GetDuplicatePoints(self):
        arrRows = np.unique(self.__AllAtomPositions, axis=0, return_inverse=True)[1]
        arrRows = np.array(list(set(range(len(self.__AllAtomPositions))).difference(arrRows.tolist())))
        if len(arrRows) > 0:
            return self.__GrainAllAtomPositions[arrRows]
        else:
            return []
    def SetGrainAtoms(self):
        lstGrainAtoms = []
        lstGrainAtomTypes = []
        for i in self.GrainList:
                lstGrainAtoms.append(self.RemoveRealDuplicates(self.GetGrain(i).GetInteriorAtomPositions(self.__BasisVectors)))
                lstGrainAtomTypes.append(np.ones(len(lstGrainAtoms[-1]))*self.GetGrain(i).GetAtomType())
        if len(lstGrainAtoms) > 0:
            self.__GrainAtomPositions = np.vstack(lstGrainAtoms)
            self.__GrainAtomTypes = np.concatenate(lstGrainAtomTypes,axis=0).astype('int')
    def GetNonGrainAtoms(self, lstAtomTypes: list):
        lstGBAtoms = []
        for k in self.GrainList:
            if self.GetGrain(k).GetAtomType() in lstAtomTypes:
                lstGBAtoms.append(self.GetGrain(k).GetBoundaryAtoms(self.__BasisVectors))
        return self.RemoveRealDuplicates(np.vstack(lstGBAtoms))
    def RemoveGrainPeriodicDuplicates(self):
        for i in self.GrainList:
            arrIndices = self.GetGrain(i).FindPeriodicDuplicates(self.__BasisVectors)
            if len(arrIndices) > 0:
                self.GetGrain(i).AddVacancies(arrIndices.tolist())
    def GetCoincidentLatticePoints(self, lstPairOfGrainIDs: list, fltDistance = 1e-5):
        lstGBAtoms = self.GetGrain(lstPairOfGrainIDs[0]).GetBoundaryAtoms(self.__BasisVectors)
        arrGBAtoms = np.vstack(lstGBAtoms)
        objGBTree = gf.PeriodicWrapperKDTree(arrGBAtoms,self.__BasisVectors,self.GetRealConstraints(),fltDistance/2)
        arrExtendedGBAtoms = objGBTree.GetExtendedPoints()
        arrIndices,arrDistances = objGBTree.Pquery_radius(self.GetGrain(lstPairOfGrainIDs[1]).GetBoundaryAtoms(self.__BasisVectors),fltDistance) #by default points are returned in distance order
        arrUniqueIndices = np.unique(np.hstack(objGBTree.GetPeriodicIndices(arrIndices)))
        return arrExtendedGBAtoms[arrUniqueIndices]
    def MergeTooCloseAtoms(self,fltDistance:float, intAtomType: int, intLimit = 50):
        if fltDistance == 0:
            fltDistance = 1e-5
        lstGBAtoms = []
        lstMergedAtoms = []
        blnStop = False
        arrGBAtoms = self.GetNonGrainAtoms([intAtomType])
        i = 0
        while not(blnStop) and i < intLimit:
            lstMergedAtoms = []
            objGBTree = gf.PeriodicWrapperKDTree(arrGBAtoms,self.__BasisVectors,self.GetRealConstraints(),fltDistance/2)
            arrExtendedGBAtoms = objGBTree.GetExtendedPoints()
            arrIndices,arrDistances = objGBTree.Pquery_radius(arrGBAtoms,fltDistance) #by default points are returned in distance order
            lstDistances = list(map(lambda x: np.round(x,5),arrDistances))
            arrLengths = np.array(list(map(lambda x: len(x),arrIndices)))
            arrRows = np.where(arrLengths > 1)[0]
            if len(arrRows) > 0:
                lstIndices = list(map(lambda x: arrIndices[x][lstDistances[x] <= lstDistances[x][1]],arrRows)) #every point 0 distance from itself 
                #point at position [1] is then the next closest point. 
                lstUsedIndices = [item for sublist in lstIndices for item in sublist]
                lstTrueIndices = np.unique(objGBTree.GetPeriodicIndices(lstUsedIndices)).tolist()
                arrUnusedIndices = np.unique(list(set(range(len(arrGBAtoms))).difference(lstTrueIndices)))
                lstMergedAtoms = list(map(lambda x: np.mean(arrExtendedGBAtoms[x],axis=0),lstIndices))
                if len(arrUnusedIndices) > 0:
                    lstMergedAtoms.append(arrGBAtoms[arrUnusedIndices])
                arrGBAtoms = np.vstack(lstMergedAtoms)
                arrGBAtoms = self.RemoveRealDuplicates(arrGBAtoms,1e-3)
            else:
                blnStop = True
            i +=1
        if i == intLimit:
            warnings.warn('Merge too close atoms terminated after ' + str(i) + ' iterations')
        self.__NonGrainAtomPositions = arrGBAtoms
        self.__NonGrainAtomTypes = np.ones(len(arrGBAtoms))*intAtomType
    def RemoveRealDuplicates(self, inPoints, fltDistance = 1e-5): #returns the unique points that lie inside the simulation cell
        arrPoints = self.WrapVectorIntoSimulationBox(inPoints)
        arrRows = gf.FindDuplicates(arrPoints, self.__BasisVectors,fltDistance)
        lstUniqueIndices = list(set(range(len(inPoints))).difference(arrRows.tolist()))
        arrUniqueIndices = np.unique(lstUniqueIndices)
        return self.WrapVectorIntoSimulationBox(arrPoints[arrUniqueIndices])
    def PlotSimulationCellAtoms(self):
        self.WrapAllAtomsIntoSimulationCell()
        return tuple(zip(*self.__GrainAtomPositions))
    def RemovePlaneOfAtoms(self, inPlane: np.array, fltTolerance: float):
        arrPointsOnPlane = gf.CheckLinearEquality(np.round(self.__UniqueRealPoints,10), inPlane,fltTolerance)
        self.__UniqueRealPoints = np.delete(self.__UniqueRealPoints,arrPointsOnPlane, axis=0)   
    def GetAtomPoints(self)->np.array:
        if len(self.__GrainAtomPositions) > 0:
            return self.__GrainAtomPositions
        else:
            lstAtoms = []
            for j in self.GrainList:
                lstAtoms.append(j.GetInteriorAtoms(self.__BasisVectors))
            return np.vstack(lstAtoms)
    def GetSimulationCellVolume(self):
        return np.abs(np.dot(self.__BasisVectors[0], np.cross(self.__BasisVectors[1], self.__BasisVectors[2])))
    def GetNumberOfVacancies(self):
        intNumberOfVacancies = 0
        for j in self.GrainList:
            intNumberOfVacancies += self.GetGrain(j).GetNumberOfVacancies()
        return intNumberOfVacancies
    def LAMMPSMinimisePositions(self,strDirectory: str,strFileOutput: str, strTemplate: str, intRange: int, fltDatum: float,fltNearestNeighbour = None):
        if fltNearestNeighbour is None:
            lstDistances = []
            for i in self.GrainList:
                lstDistances.append(self.GetGrain(i).GetNearestNeighbourDistance())
            fltNearestNeighbour = min(lstDistances)
        lstj = []
        lstAtoms = []
        lstPE = []
        lstAdjusted = []
        self.WrapAllAtomsIntoSimulationCell()
        for j in range(0, intRange):
            lstj.append(fltNearestNeighbour*j/(intRange))
            self.RemoveTooCloseAtoms(lstj[-1])
            lstAtoms.append(self.GetUpdatedAtomNumbers())
            self.SetFileHeader('Something')
            self.WrapAllAtomsIntoSimulationCell()
            self.WriteLAMMPSDataFile(strDirectory + 'read.dat')
            objLammps = lammps.PyLammps()
            objLammps.file(strDirectory + strTemplate) #must potential energy
            if len(lstj) ==1: 
                lstAdjusted.append(0)
                lstPE.append(objLammps.eval('pe'))
            elif lstAtoms[-2] == lstAtoms[-1]:
                lstPE.append(lstPE[-1])
                lstAdjusted.append(lstAdjusted[-1])
            else:
                lstPE.append(objLammps.eval('pe'))
                lstAdjusted.append(lstPE[-1] - lstPE[0] + fltDatum*(lstAtoms[0]-lstAtoms[-1]))    
            #arrMins =  np.where(np.array(lstAdjusted) == min(lstAdjusted))[0]
            if lstAdjusted[j] == min(lstAdjusted):
                    objLammps.command('write_data ' + strDirectory + strFileOutput)
                    self.WriteLAMMPSDataFile(strDirectory + 'best.dat')
            objLammps.close()
        return lstAtoms, lstAdjusted,np.argmin(lstAdjusted)/intRange
    def GetRealConstraints(self):
        arrConstraints = gf.FindConstraintsFromBasisVectors(self.__BasisVectors)
        return arrConstraints
    def RemoveAtomsOnOpenBoundaries(self):
        for j in self.GrainList:
            self.GetGrain(j).SetOpenBoundaryPoints(self.GetRealConstraints())
    def GetNumberOfNonGrainAtoms(self):
        return len(self.__NonGrainAtomPositions)
    def GetNonGrainAtomPositions(self):
        return self.__NonGrainAtomPositions
    def GetNonGrainAtomTypes(self):
        return self.__NonGrainAtomTypes
    def RemoveNonGrainAtomPositons(self):
        self.__NonGrainAtomPositions = []
        self.__NonGrainAtomTypes = []
 
class Grain(object):
    def __init__(self, intGrainNumber: int):
        self.__GrainID = intGrainNumber
        self.__GlobalID = -1
        self.__AdjacentGrainBoundaries = []
        self.__AdjacentJunctionLines = []
        self.__AdjacentGrains = []
        self.__GrainCentre = []
    def SetAdjacentGrainBoundaries(self, lstAdjacentGrainBoundaries):
        self.__AdjacentGrainBoundaries
    def GetAdjacentGrainBoundaries(self):
        return self.__AdjacentGrainBoundaries
    def SetAdjacentJunctionLines(self, lstAdjacentJunctionLines):
        self.__AdjacentJunctionLines
    def GetAdjacentJunctionLines(self):
        return self.__AdjacentJunctionLines
    def SetGrainCentre(self, arrGrainCentre):
        self.__GrainCentre = arrGrainCentre
    def GetGrainCentre(self):
        return self.__GrainCentre
        
class DefectMeshObject(object):
    def __init__(self,inMeshPoints: np.array, intID: int):
        self.__MeshPoints = inMeshPoints
        self.__ID = intID
        self.__OriginalID = intID
        self.__AtomIDs = []
        self.__AdjustedMeshPoints = [] #these are general mesh points adjusted to correct for the limited accuracy of the QuantisedCuboid object
        self.__Volume = 0
        self.__PE = 0
        self.__PeriodicVectors = [] #used to mesh points in the periodic directions.
        self.__ExtendedMeshPoints = []
    def GetMeshPoints(self):
        return np.copy(self.__MeshPoints)
    def GetOriginalID(self):
        return self.__OriginalID 
    def GetID(self):
        return self.__ID
    def SetID(self, intID):
        self.__ID = intID
    def GetNumberOfAtoms(self):
        return len(self.__AtomIDs)
    def SetAtomIDs(self, inlstIDs: list):
        self.__AtomIDs = list(map(int,inlstIDs))
    def GetAtomIDs(self)->list:
        return cp.copy(self.__AtomIDs)
    def AddAtomIDs(self, inList):
        inList = set(map(int,inList))
        self.__AtomIDs = list(inList.union(self.__AtomIDs))
    def RemoveAtomIDs(self, inList):
        setAtomIDs = set(self.__AtomIDs)
        self.__AtomIDs = list(setAtomIDs.difference(inList))
    def SetMeshPoints(self, inPoints):
        self.__MeshPoints = inPoints
    def SetAdjustedMeshPoints(self, inPoints: np.array):
        self.__AdjustedMeshPoints = inPoints
    def GetAdjustedMeshPoints(self)->np.array:
        return self.__AdjustedMeshPoints
    def SetVolume(self, fltVolume):
        self.__Volume = fltVolume 
    def GetVolume(self):
        return self.__Volume
    def GetVolumePerAtom(self):
        if self.GetNumberOfAtoms() > 0: 
            return self.__Volume/self.GetNumberOfAtoms()
    def GetAtomicDensity(self):
        if self.__Volume > 0 and len(self.__AtomIDs) > 0:
            return len(self.__AtomIDs)/self.__Volume
        else:
            return 0   
    def SetPeriodicDirections(self, inList):
        self.__PeriodicDirections = inList
    def GetPeriodicDirections(self):
        return cp.copy(self.__PeriodicDirections)
    def GetTotalPE(self, fltPEDatum = None):
        if fltPEDatum is None:
            return self.__PE
        else:
            return self.__PE - fltPEDatum*self.GetNumberOfAtoms()
    def SetTotalPE(self, fltPE):
        self.__PE = fltPE 
    def GetPEPerAtom(self, fltPEDatum = None):
        if self.GetNumberOfAtoms() > 0:
            return self.GetTotalPE(fltPEDatum)/len(self.__AtomIDs)
    def GetPEPerVolume(self, fltPEDatum = None):
        if self.__Volume > 0:
            return self.GetTotalPE(fltPEDatum)/self.__Volume
    def SetPeriodicVectors(self, inVectors):
        self.__PeriodicVectors
    def SetExtraMeshPoints(self, inPoints):
        self.__ExtraMeshPoints = inPoints
    def GetExtraMeshPoints(self):
        return self.__ExtraMeshPoints

class GeneralJunctionLine(DefectMeshObject):
    def __init__(self,inMeshPoints: np.array, intID: int):
        DefectMeshObject.__init__(self,inMeshPoints, intID)
        self.__AdjacentGrains = []
        self.__AdjacentGrainBoundaries = []
        self.__PeriodicDirections = []
        self.__JunctionLength = 0
    def SetAdjacentGrains(self, inList):
        self.__AdjacentGrains = inList
    def GetAdjacentGrains(self)->list:
        return cp.copy(self.__AdjacentGrains)
    def SetAdjacentGrainBoundaries(self, inList):
        self.__AdjacentGrainBoundaries = inList
    def GetAdjacentGrainBoundaries(self)->list:
        return cp.copy(self.__AdjacentGrainBoundaries)
    def FindJunctionLength(self):
        self.__JunctionLength = gf.FindSplineLength(np.append(self.GetMeshPoints(), self.GetExtraMeshPoints(),axis=0))
    def GetJunctionLineLength(self):
        if self.__JunctionLength == 0:
            self.FindJunctionLength()
        return self.__JunctionLength
   
class GeneralGrainBoundary(DefectMeshObject):
    def __init__(self,inMeshPoints: np.array, intID: str):
        DefectMeshObject.__init__(self,inMeshPoints, intID)
        self.__AdjacentGrains = []
        self.__AdjacentJunctionLines = []
        self.__AdjacentGrainBoundaries = []
        self.__PeriodicDirections = [] 
        self.__SurfaceArea= 0  
        self.__SurfaceMesh = []
    def SetAdjacentGrains(self, inList):
        self.__AdjacentGrains = inList
    def GetAdjacentGrains(self)->list:
        return cp.copy(self.__AdjacentGrains) 
    def SetAdjacentJunctionLines(self, inList):
        self.__AdjacentJunctionLines = inList
    def GetAdjacentJunctionLines(self)->list:
        return cp.copy(self.__AdjacentJunctionLines)
    def GetSurfaceArea(self):
        self.__SurfaceArea = gf.FindSurfaceArea(np.append(self.GetMeshPoints(),self.GetExtraMeshPoints(),axis=0)) 
       #self.__SurfaceArea = gf.FindSurfaceArea(self.GetMeshPoints()) 
        return self.__SurfaceArea
    def SetSurfaceMesh(self, lstPoints):
        self.__SurfaceMesh = lstPoints
    def GetSurfaceMesh(self, intIndex = None):
        if intIndex is None:
            return np.vstack(self.__SurfaceMesh)
        else:
            return self.__SurfaceMesh[intIndex]
    def GetGrainBoundaryWidth(self):
        if self.__SurfaceArea == 0:
            self.GetSurfaceArea()
        return self.GetVolume()/self.__SurfaceArea
    def GetEnergyPerSurfaceArea(self,fltDatum: float, blnSIUnits = True):
        self.GetSurfaceArea()
        fltValue =  self.GetTotalPE(fltDatum)/self.__SurfaceArea
        if blnSIUnits:
            fltValue *= 16.0217662 #convert from eV/Anstrom**2 to J/m**2
        return fltValue
   
class DefectObject(object):
    def __init__(self, fltTimeStep = None):
        if fltTimeStep is not None:
            self.__TimeStep = fltTimeStep
        else:
            self.__TimeStep = []
        self.__dctJunctionLines = dict()
        self.__dctGrainBoundaries = dict()
    def AddJunctionLine(self, objJunctionLine: GeneralJunctionLine):
        self.__dctJunctionLines[objJunctionLine.GetID()] = objJunctionLine 
    def AddGrainBoundary(self, objGrainBoundary: GeneralGrainBoundary):
        self.__dctGrainBoundaries[objGrainBoundary.GetID()] = objGrainBoundary
    def GetJunctionLine(self, intLocalKey: int):
        return self.__dctJunctionLines[intLocalKey]
    def GetGrainBoundary(self, intLocalKey):
        return self.__dctGrainBoundaries[intLocalKey]      
    def GetJunctionLineIDs(self):
        return list(self.__dctJunctionLines.keys())
    def GetGrainBoundaryIDs(self):
        return list(self.__dctGrainBoundaries.keys())
    def GetAdjacentGrainBoundaries(self, intJunctionLine: int)->list:
        lstAdjacentGrainBoundaries = []
        for j in self.GetJunctionLine(intJunctionLine).GetAdjacentGrainBoundaries():
            lstAdjacentGrainBoundaries.append(self.GetGrainBoundary(j).GetID())
        return lstAdjacentGrainBoundaries
    def GetAdjacentJunctionLines(self, intGrainBoundary: int)->list:
        lstAdjacentJunctionLines = []
        for j in self.GetGrainBoundary(intGrainBoundary).GetAdjacentJunctionLines():
            lstAdjacentJunctionLines.append(self.GetJunctionLine(j).GetID())
        return lstAdjacentJunctionLines
    def SetTimeStep(self, fltTimeStep):
        self.__TimeStep = fltTimeStep
    def GetTimeStep(self):
        return self.__TimeStep
    def ImportData(self, strFilename: str):
        with open(strFilename,'r') as fdata:
            blnNotEnd = True
            try:
                line = next(fdata).strip()
                objJunctionLine = None
                objGrainBoundary = None
                while blnNotEnd:
                    if line == "Time Step":
                        self.__TimeStep = int(next(fdata).strip())
                        line = next(fdata).strip()
                    elif line == "Junction Line":
                        intJL = int(next(fdata).strip())
                        line = next(fdata).strip()
                        if line == "Mesh Points":
                            line = next(fdata).strip()    
                            arrMeshPoints = np.array(eval(line))
                            objJunctionLine = GeneralJunctionLine(arrMeshPoints, intJL)
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
                            line = next(fdata).strip()
                        if line == "Adjusted Mesh Points":
                            line = next(fdata).strip()
                            arrAdjustedMeshPoints = np.array(eval(line))
                            objJunctionLine.SetAdjustedMeshPoints(arrAdjustedMeshPoints)
                            line = next(fdata).strip()
                        self.AddJunctionLine(objJunctionLine)       
                    elif line == "Grain Boundary":
                        intGB = int(next(fdata).strip())
                        line = next(fdata).strip()
                        if line == "Mesh Points":
                            line = next(fdata).strip()    
                            arrMeshPoints = np.array(eval(line))
                            objGrainBoundary = GeneralGrainBoundary(arrMeshPoints, intGB)
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
                            line = next(fdata).strip()
                        if line == "Adjusted Mesh Points":
                            line = next(fdata).strip()
                            arrAdjustedMeshPoints = np.array(eval(line))
                            objGrainBoundary.SetAdjustedMeshPoints(arrAdjustedMeshPoints)
                            line = next(fdata).strip()
                        self.AddGrainBoundary(objGrainBoundary)
                    else:
                        blnNotEnd = False
            except StopIteration as EndOfFile:
                if objJunctionLine is not None:
                    self.AddJunctionLine(objJunctionLine) #Some fields are optional so end of file may come
                if objGrainBoundary is not None:
                    self.AddGrainBoundary(objGrainBoundary) #before the GB or JL objects have been added        
                blnNotEnd = False

class SigmaCell(object):
    def __init__(self, arrRotationAxis: np.array, inCellNodes: np.array):
        intGCD = np.gcd.reduce(arrRotationAxis)
        self.__RotationAxis = (arrRotationAxis/intGCD).astype('int')
        # if np.all(self.__RotationAxis == np.array([0,0,1])):
        #     self.__LatticeBasis = gf.StandardBasisVectors(3)
        #     self.__CellHeight = 1
        # else:
        #     fltAngle, arrVector = gf.FindRotationVectorAndAngle(arrRotationAxis, np.array([0,0,1]))
        #     self.__LatticeBasis = gf.RotatedBasisVectors(fltAngle,arrVector)
        self.__CellHeight = np.linalg.norm(self.__RotationAxis)
        self.__CellType = inCellNodes
        self.__BasisVectors = []
        self.__CSLPrimitiveVectors = []
        self.__CSLPrimitiveInverse = []
        self.__CSLPoints = []
        self.__BasisVectors = []
        self.__TransformationMatrix = []
        self.__LatticeBases = []
        self.__MedianLattice = []
    def GetCSLPoints(self):
        return self.__CSLPoints
    def GetRotationAxis(self):
        return self.__RotationAxis
    def GetSigmaValues(self, intSigmaMax, blnDisorientation = True):
        return  gf.CubicCSLGenerator(self.__RotationAxis, intSigmaMax,blnDisorientation)
    def MakeCSLCell(self, intSigmaValue: int, arrHorizontalVector = np.array([1,0,0]), blnUnitCell = True):
        blnValidSigma = True
        arrSigma = self.GetSigmaValues(25, True)
        arrRows = np.where(arrSigma[:,0].astype('int') == intSigmaValue)
        if len(arrRows[0]) == 0:
            blnValidSigma = False
        if blnValidSigma:
            arrSigmas = arrSigma[arrRows]
            intMin = np.argmin(np.abs(arrSigmas[:,1]))
            intSigmaValue = int(arrSigmas[intMin,0])
            h = self.__CellHeight
            l = intSigmaValue
            #arrCSLCentre = np.array([l,l,l])
            fltSigma = float(arrSigmas[intMin,1])
            arrBasis1 = gf.StandardBasisVectors(3)
            arrBasis2 = gf.RotateVectors(fltSigma,self.__RotationAxis,gf.StandardBasisVectors(3))
            arrBasisMedian = gf.RotateVectors(fltSigma/2,self.__RotationAxis,gf.StandardBasisVectors(3))
            objFirstLattice = ExtrudedRectangle(l,l,l,arrBasis1, self.__CellType, np.ones(3),np.zeros(3))
            objSecondLattice = ExtrudedRectangle(l,l,l,arrBasis2,self.__CellType,np.ones(3),np.zeros(3))
            arrPoints1 = objFirstLattice.GetRealPoints()
            arrPoints2 = objSecondLattice.GetRealPoints()
            objTree1 = KDTree(arrPoints1)
            arrDistancesOne, arrIndicesOne = objTree1.query(arrPoints2, k=1)
            arrCloseOne = np.where(arrDistancesOne < 1e-5)[0]
            arrIndicesOne = arrIndicesOne.ravel()
            arrIndicesOne = arrIndicesOne[arrCloseOne]
            arrCSLPoints = arrPoints1[arrIndicesOne]
            arrMediod = gf.FindGeometricMediod(arrCSLPoints)
            arrCSLPoints = arrCSLPoints - arrMediod
            self.__CSLPoints = arrCSLPoints
            #arrRows = np.where(np.matmul(arrCSLPoints, np.transpose(self.__RotationAxis)) == 0)[0]
            #arrPlane = arrCSLPoints[arrRows]
            arrDistances = np.round(np.linalg.norm(arrCSLPoints,axis=1),5)
            lstPositions = gf.FindNthSmallestPosition(arrDistances, 1)
            arrVector1 = arrCSLPoints[lstPositions[0]]
            blnFound1 = False
            if len(lstPositions) > 1: #there are atleast two equidistance vectors
                i = 1
                while i < len(lstPositions) and not(blnFound1):
                    arrVector2 = arrCSLPoints[lstPositions[i]]
                    if np.any(np.abs(np.round(np.cross(arrVector2,arrVector1),5)) > 0):
                        blnFound1  = True
                    i +=1
            if not(blnFound1): #equidistant vectors were all parallael
                i = 3
                blnFound2 = False
                while i < len(arrDistances) and not(blnFound2):
                    arrVector2 = arrCSLPoints[gf.FindNthSmallestPosition(arrDistances,i)[0]]
                    if np.any(np.abs(np.round(np.cross(arrVector2,arrVector1),5)) > 0):
                        blnFound2  = True
                    i +=1
            blnFound3 = False
            while i < len(arrDistances) and not(blnFound3):
                lstPositions = gf.FindNthSmallestPosition(arrDistances,i)
                k = 0
                while k < len(lstPositions) and not(blnFound3):
                    arrVector3 = arrCSLPoints[lstPositions[k]]
                    if abs(np.linalg.det(np.array([arrVector1,arrVector2,arrVector3]))) >1e-5:
                        k +=1
                        blnFound3  = True
                    k +=1
                i += 1
            lstVectors = []
            lstVectors.append(arrVector2)
            lstVectors.append(arrVector1)
            lstVectors.append(arrVector3)
            arrPrimitiveVectors = np.vstack(lstVectors)
            arrAllVectors = arrPrimitiveVectors
            lstAllVectors = []
            self.__CSLPrimitiveVectors = arrPrimitiveVectors
            self.__CSLPrimitiveInverse = np.linalg.inv(arrPrimitiveVectors)
            for a in arrPrimitiveVectors:
                lstAllVectors.append(arrAllVectors + a)
                lstAllVectors.append(arrAllVectors -a)
                arrAllVectors = np.vstack(lstAllVectors)
            arrAllVectors = np.unique(arrAllVectors, axis=0)
            arrDeleteRows = np.where(np.all(arrAllVectors == np.zeros(3),axis=1))[0]
            arrPlane = np.delete(arrAllVectors, arrDeleteRows, axis=0)
            arrRows = np.where(np.abs(np.matmul(arrPlane, np.transpose(self.__RotationAxis)))< 1e-5)[0]
            arrPlane = arrPlane[arrRows]
            if self.__CheckIsCSLVectors(self.__RotationAxis):
                arrReturnVectors = np.zeros([3,3])
                arrPlaneDistances = np.linalg.norm(arrPlane, axis=1)
                lstPositions = gf.FindNthSmallestPosition(arrPlaneDistances,0)
                arrNextVector = arrPlane[lstPositions[0]]
                arrRows = np.where(np.abs(np.matmul(arrPlane, np.transpose(arrNextVector)))< 1e-5)[0]
                if len(arrRows) > 0:
                    arrPlane = arrPlane[arrRows]
                    arrNextDistances = np.linalg.norm(arrPlane, axis=1)
                    lstPositions = gf.FindNthSmallestPosition(arrNextDistances,0)
                    arrLastVector = arrPlane[lstPositions[0]]
                    arrReturnVectors[0] = arrLastVector
                    arrReturnVectors[1] = arrNextVector
                    arrReturnVectors[-1] =  self.__RotationAxis
                else:
                    arrReturnVectors[1] = arrNextVector
                    arrReturnVectors[-1] = self.__RotationAxis
                    blnFound4 = False
                    i = 0
                    while i < len(arrPlaneDistances) and not(blnFound4):
                        lstPositions = gf.FindNthSmallestPosition(arrPlaneDistances,i)
                        k = 0
                        while k < len(lstPositions) and not(blnFound4): 
                            arrReturnVectors[0] = arrPlane[lstPositions[k]]
                            if np.round(np.linalg.det(arrReturnVectors),10) > 0 and not(blnFound4):
                                blnFound4 = True  
                            k += 1
                        i += 1    
            if blnUnitCell:
                for k in range(len(arrReturnVectors)):
                    if arrReturnVectors[k,k] < 0:
                        arrReturnVectors[k] = - arrReturnVectors[k]
                arrReturnVectors, arrTransformation = gf.ConvertToLAMMPSBasis(arrReturnVectors)
            else: 
                arrReturnVectors, arrTransformation = gf.ConvertToLAMMPSBasis(arrPrimitiveVectors)

            self.__BasisVectors = np.round(arrReturnVectors,10)
            self.__TransformationMatrix = arrTransformation
            lstLatticeBasis = []
            lstLatticeBasis.append(np.matmul(arrBasis1,arrTransformation))
            lstLatticeBasis.append(np.matmul(arrBasis2,arrTransformation))
            self.__LatticeBases = lstLatticeBasis 
            self.__MedianLattice = np.matmul(arrBasisMedian,arrTransformation)   
        else:
            warnings.warn("Invalid sigma value for axis " + str(self.__RotationAxis))
    def GetLatticeBases(self):
        return self.__LatticeBases
    def GetMedianLattice(self):
        return self.__MedianLattice
    def GetTransformationMatrix(self):
        return self.__TransformationMatrix
    def GetBasisVectors(self):
        return self.__BasisVectors
    def GetLatticeRotations(self):
        return self.__LatticeRotations
    def GetPrimitiveCoefficients(self,inVector: np.array):
        return np.round(np.matmul(np.transpose(inVector),self.__CSLPrimitiveInverse),10)
    def __CheckIsCSLVectors(self,inVector: np.array):
        blnIsCSL = False
        arrCoefficients =  self.GetPrimitiveCoefficients(inVector)        
        if np.all(np.mod(arrCoefficients, np.ones(3))) == 0:
            blnIsCSL = True
        return blnIsCSL

class CSLTripleLine(object):
    def __init__(self,arrRotationAxis: np.array, inCellNodes: np.array) -> None:
        intGCD = np.gcd.reduce(arrRotationAxis)
        self.__RotationAxis = (arrRotationAxis/intGCD).astype('int')
        if np.all(self.__RotationAxis == np.array([0,0,1])):
            self.__LatticeBasis = gf.StandardBasisVectors(3)
            self.__CellHeight = 1
        else:
            fltAngle, arrVector = gf.FindRotationVectorAndAngle(arrRotationAxis, np.array([0,0,1]))
            self.__LatticeBasis = gf.RotatedBasisVectors(fltAngle,arrVector)
            self.__CellHeight = np.linalg.norm(self.__RotationAxis)
        self.__CellType = inCellNodes
        self.__CSLBasisVectors = []
        self.__TripleValues = []
    def FindTripleLineSigmaValues(self,  intSigmaMax: int, intIterations = 50):
        arrSigma = gf.CubicCSLGenerator(self.__RotationAxis,intIterations)
        arrSigmaValues = arrSigma[:,0].astype('int')
        arrRows = np.where(arrSigmaValues <= intSigmaMax)
        intLength = np.max(arrRows)
        lstIndices = []
        for i in range(intLength):
            for j in range(i,intLength):
                for k in range(j,intLength):
                    fltAngle = arrSigma[i,1]+arrSigma[j,1]+arrSigma[k,1] -2*np.pi
                    if abs(fltAngle) < 1e-5:
                        if sorted([i,j,k]) not in lstIndices:
                            lstIndices.append(sorted([i,j,k]))      
        arrTripleValues = arrSigma[np.array(lstIndices)]
        arrTJSigmaValues = np.zeros([len(arrTripleValues)])
        n = 0
        for a in arrTripleValues:
            arrTJSigmaValues[n] = self.GetTJSigmaValue(a)
            n += 1
        self.__TJSigmaValues = arrTJSigmaValues
        self.__TripleValues = arrTripleValues
        self.__RotationAngles = np.zeros(len(arrTripleValues))
        return arrTripleValues
    def GetTripleLineSigmaValues(self):
        return self.__TJSigmaValues
    def GetTripleLineValues(self):
        return self.__TripleValues
    def GetGCD(self, arrSigmaArray: np.array, intIndex1: int, intIndex2: int):
        arrQ1 = gf.GetMatrixFromAxisAngle(self.__RotationAxis, arrSigmaArray[intIndex1,1])
        arrQ2 = gf.GetMatrixFromAxisAngle(self.__RotationAxis, arrSigmaArray[intIndex2,1])
        arrProduct = np.round(np.matmul(arrQ1,arrQ2)*arrSigmaArray[intIndex1,0]*arrSigmaArray[intIndex2,0],10).astype('int')
        intGCD = np.gcd.reduce(np.gcd.reduce(arrProduct))
    def GetTJSigmaValue(self, arrSigmaArray: np.array):
        #intSigma = np.sqrt(arrSigmaArray[2,0]**2*intGCD)
        intSigma = np.sqrt(np.product(arrSigmaArray[:,0]))
        return intSigma
    def GetTJBasisVectors(self, intTJSigmaValueIndex: int, blnUnitCell = False):
        arrTripleValues = self.__TripleValues[intTJSigmaValueIndex]
        arrBasisVectors = gf.StandardBasisVectors(3)
        flth = 2*np.linalg.norm(self.__RotationAxis) 
        l = self.__TJSigmaValues[intTJSigmaValueIndex]
        arrBasis1 = arrBasisVectors
        arrBasis2 = gf.RotateVectors(arrTripleValues[0,1],self.__RotationAxis,arrBasisVectors)
        arrBasis3 = gf.RotateVectors(arrTripleValues[0,1] + arrTripleValues[1,1],self.__RotationAxis,arrBasisVectors)
        objFirstLattice = ExtrudedRectangle(l,l,l,arrBasis1, self.__CellType, np.ones(3),np.zeros(3))
        objSecondLattice = ExtrudedRectangle(l,l,l,arrBasis2, self.__CellType, np.ones(3),np.zeros(3))
        objThirdLattice = ExtrudedRectangle(l,l,l,arrBasis3, self.__CellType, np.ones(3),np.zeros(3))
        arrPoints1 = objFirstLattice.GetRealPoints()
        arrPoints2 = objSecondLattice.GetRealPoints()
        arrPoints3 = objThirdLattice.GetRealPoints()
        objTree1 = KDTree(arrPoints1)
        arrDistancesOne, arrIndicesOne = objTree1.query(arrPoints2,k=1)
        arrIndicesOne = arrIndicesOne.ravel()
        arrCloseOne = np.where(arrDistancesOne < 1e-5)[0]
        arrOneAndTwo = arrPoints1[arrIndicesOne[arrCloseOne]] 
        objTree2 = KDTree(arrOneAndTwo)
        arrDistancesTwo,arrIndicesTwo = objTree2.query(arrPoints3, k=1)
        arrIndicesTwo = arrIndicesTwo.ravel()
        arrCloseTwo = np.where(arrDistancesTwo< 1e-5)[0]
        arrCloseAll = arrOneAndTwo[arrIndicesTwo[arrCloseTwo]]
        arrMean = np.mean(arrCloseAll,axis=0)
        arrDistances = np.linalg.norm(arrCloseAll-arrMean, axis=1)
        arrMediod = arrCloseAll[np.argmin(arrDistances)]
        arrCentredPoints = arrCloseAll - arrMediod 
        arrDistances = np.linalg.norm(arrCentredPoints, axis=1)
        #arrVector1 = self.__RotationAxis
        arrVector1 = arrCentredPoints[gf.FindNthSmallestPosition(arrDistances,1)[0]]
        i = 2
        blnFoundVector2 = False
        while i < len(arrCentredPoints)  and not(blnFoundVector2):
            lstPositions = gf.FindNthSmallestPosition(arrDistances, i)
            k = 0
            while k < len(lstPositions) and not(blnFoundVector2):
                arrVector2 = arrCentredPoints[lstPositions[k]] 
                if np.all(np.abs(np.cross(arrVector1,arrVector2)) <1e-5):
                    k +=1
                elif np.all(np.mod(arrVector2, np.ones(3))==np.zeros(3)) or not(blnUnitCell):
                    blnFoundVector2 = True
                else:
                    k +=1
            i +=1
        j = i
        blnFoundVector3 = False
        while j < len(arrCentredPoints)  and not(blnFoundVector3):
            lstPositions = gf.FindNthSmallestPosition(arrDistances, j)
            k = 0
            while k < len(lstPositions) and not(blnFoundVector3):
                arrVector3 = arrCentredPoints[lstPositions[k]] 
                if abs(np.linalg.det(np.array([arrVector1,arrVector2,arrVector3]))) <1e-5:
                    k +=1
                elif np.all(np.mod(arrVector3, np.ones(3))==np.zeros(3)) or not(blnUnitCell):
                    blnFoundVector3 = True
                else:
                    k +=1
            j +=1
        lstVectors= [arrVector2,arrVector3, arrVector1]
        arrVectors = np.vstack(lstVectors)
       # arrVectors[:2,:] = arrVectors[:2,:][np.argsort(np.abs(arrVectors[:,0]))]
        for k in range(len(arrVectors)):
            if arrVectors[k,k] < 0:
                arrVectors[k] = -arrVectors[k]    
        arrReturn = arrVectors
        self.__CSLBasisVectors = arrReturn
        arrRealBasis, arrTransformationMatrix  = gf.ConvertToLAMMPSBasis(arrReturn)
        self.__SimulationCellBasis = arrRealBasis
        self.__RotationMatrix = arrTransformationMatrix
        lstLatticeBasis = []
        lstLatticeBasis.append(np.matmul(arrBasis1,arrTransformationMatrix))
        lstLatticeBasis.append(np.matmul(arrBasis2,arrTransformationMatrix))
        lstLatticeBasis.append(np.matmul(arrBasis3,arrTransformationMatrix))
        self.__LatticeBases = lstLatticeBasis
    def GetCSLBasisVectors(self):
        return self.__CSLBasisVectors                 
    def GetSimulationCellBasis(self):
        return self.__SimulationCellBasis
    def GetRotationMatrix(self):
        return self.__RotationMatrix
    def GetLatticeBasis(self, intIndex: int):
        return self.__LatticeBases[intIndex]
  
    