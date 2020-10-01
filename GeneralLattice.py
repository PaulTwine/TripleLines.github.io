import numpy as np
import GeometryFunctions as gf
#import LatticeShapes as ls
import LatticeDefinitions as ld
import scipy as sc
from sympy.parsing.sympy_parser import parse_expr
from sympy import lambdify
from sympy.abc import x,y,z
from datetime import datetime
import copy as cp

class PureCell(object):
    def __init__(self,inCellNodes: np.array): 
        self.__CellNodes = inCellNodes
        self.__NumberOfCellNodes = len(inCellNodes)
        self.__Dimensions = len(inCellNodes[0])
    def UnitVector(self, intNumber: int)->np.array:
        arrVector = np.zeros(self.Dimensions())
        if intNumber >= 0:
            arrVector[intNumber] = 1
        else:
            arrVector[-intNumber] = -1
        return np.array(arrVector)
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
    def CellDirectionalMotif(self, intBasisVector: int, intSign = 1)->np.array:
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
        arrDistanceMatrix = sc.spatial.distance_matrix(self.__RealCellVectors,self.__RealCellVectors)
        self.__NearestNeighbourDistance = np.min(arrDistanceMatrix[arrDistanceMatrix > 0])
    def GetCellVectors(self)->np.array:
        return self.__CellVectors
    def GetRealCellVectors(self)->np.array:
        return self.__RealCellVectors
    def GetLatticeParameters(self):
        return self.__LatticeParameters
    def GetNearestNeighbourDistance(self):
        return self.__NearestNeighbourDistance
    
class GeneralLattice(RealCell):
    def __init__(self,inBasisVectors:np.array,inCellNodes: np.array,inLatticeParameters:np.array,inOrigin: np.array,inCellBasis = None):
        RealCell.__init__(self, inCellNodes,inLatticeParameters, inCellBasis)
        self.__UnitBasisVectors  = inBasisVectors # Cartesian basis vectors for the lattice
        self.__RealPoints = []
        self.__LatticePoints = []
        self.__CellPoints = []
        self.__AtomType = 1
        self.__Origin = inOrigin
        self.__LatticeParameters = inLatticeParameters
        self.__RealBasisVectors = np.zeros([self.Dimensions(),self.Dimensions()])
        for j in range(self.Dimensions()):
            self.__RealBasisVectors[j] = inLatticeParameters[j]*inBasisVectors[j]
    def GetUnitBasisVectors(self)->np.array:
        return self.__UnitBasisVectors
    def GetRealBasisVectors(self)->np.array:
        return self.__RealBasisVectors
    def __DeletePoints(self,lstDeletedIndices: list):
        self.__RealPoints = np.delete(self.__RealPoints, lstDeletedIndices, axis=0)
        self.__LatticePoints  = np.delete(self.__LatticePoints, lstDeletedIndices, axis=0) 
    def GetRealPoints(self)->np.array:
        return self.__RealPoints
    def MakeRealPoints(self, inConstraints):
        self.__LinearConstraints = inConstraints
        self.GenerateLatticeConstraints(inConstraints)
        arrBounds = self.FindBoxConstraints(self.__LatticeConstraints)
        arrBounds[:,0] = np.floor(arrBounds[:,0])
        arrBounds[:,1] = np.ceil(arrBounds[:,1])
        arrCellPoints = np.array(gf.CreateCuboidPoints(arrBounds))
        arrLatticePoints = self.MakeLatticePoints(arrCellPoints)
        arrLatticePoints = np.delete(arrLatticePoints, self.CheckLatticeConstraints(arrLatticePoints), axis = 0)
        self.GenerateRealPoints(arrLatticePoints)
    def MakeLatticePoints(self, inCellPoints):
        arrLatticePoints = np.empty([self.GetNumberOfCellNodes()*len(inCellPoints), self.Dimensions()])
        for i, position in enumerate(inCellPoints):
            for j, cell in enumerate(self.GetCellNodes()):
                arrLatticePoints[j+i*self.GetNumberOfCellNodes()] = np.add(position,cell)
        return np.unique(arrLatticePoints, axis = 0)
    def GenerateRealPoints(self, inLatticePoints):
        self.__LatticePoints = inLatticePoints
        arrRealPoints = np.round(np.matmul(inLatticePoints, self.GetRealCellVectors()),10)
        self.__RealPoints = np.round(np.matmul(arrRealPoints, self.GetUnitBasisVectors()),10)
        self.__RealPoints = np.add(self.__Origin, self.__RealPoints)
    def GenerateLatticeConstraints(self, inConstraints: np.array):
        rtnArray = np.zeros([len(inConstraints),len(inConstraints[0])])
        tmpArray = np.zeros([3])
        for i in range(len(inConstraints)):
            arrVector = inConstraints[i,:-1]
            fltLength = np.linalg.norm(arrVector)
            arrVector = arrVector/fltLength
            arrConstraint = inConstraints[i,3]*arrVector/fltLength**2
            arrVector = np.matmul(arrVector, np.linalg.inv(self.GetRealBasisVectors()))
            arrConstraint = np.matmul(arrConstraint, np.linalg.inv(self.GetRealBasisVectors()))
            for k in range(3):
                rtnArray[i,k] = np.dot(arrVector,self.GetCellVectors()[k]) # generally a non-Carteisan Basis
                tmpArray[k] = np.dot(arrConstraint, self.GetCellVectors()[k])
            fltLength = np.linalg.norm(rtnArray[i,:3])
            rtnArray[i,:3] = rtnArray[i,:3]/fltLength 
            rtnArray[i,3] = gf.InnerProduct(rtnArray[i,:3], tmpArray,np.linalg.inv(self.GetCellVectors()))
        self.__LatticeConstraints = rtnArray
    def CheckLinearConstraints(self,inPoints: np.array)-> np.array: #returns indices to delete for real coordinates  
        arrPositions = np.subtract(np.matmul(inPoints, np.transpose(self.__LinearConstraints[:,:-1])), np.transpose(self.__LinearConstraints[:,-1])) #if it fails any constraint then the point is put in the deleted list
        arrPositions = np.argwhere(np.round(arrPositions,10) > 0)[:,0]        
        return arrPositions
    def CheckLatticeConstraints(self,inPoints: np.array)-> np.array: #returns indices to delete   
         arrPositions = np.subtract(np.matmul(inPoints, np.transpose(self.__LatticeConstraints[:,:-1])), np.transpose(self.__LatticeConstraints[:,-1]))
         arrPositions = np.argwhere(np.round(arrPositions,10) > 0)[:,0]        
         return np.unique(arrPositions)
    def GetNumberOfAtoms(self)->int:
        return len(self.__RealPoints)
    def GetAtomType(self)->int:
        return self.__AtomType
    def GetOrigin(self)->np.array:
        return self.__Origin
    def SetOrigin(self, inArray: np.array):
        self.__Origin = inArray
    def MatLabPlot(self):
        return zip(*self.GetRealPoints())
    def LinearConstrainRealPoints(self, inConstraint: np.array):
        lstDeletedIndices = gf.CheckLinearConstraint(self.__RealPoints, inConstraint)
        self.__DeletePoints(lstDeletedIndices)
    def RemovePlaneOfAtoms(self, inPlane: np.array):
        lstDeletedIndices = gf.CheckLinearEquality(self.__RealPoints, inPlane, 0.01)
        self.__DeletePoints(lstDeletedIndices)
    #FindBoxConstraint only works for linear constraints. Searches for all the vertices where three constraints #simultaneously apply and then finds the points furthest from the origin.
    def FindBoxConstraints(self,inConstraints: np.array, fltTolerance = 0.0001)->np.array:
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
    def ApplyGeneralConstraint(self,strFunction, strVariables='[x,y,z]'): #default scalar value is less than or equal to 0 if "inside" the region
        lstVariables = parse_expr(strVariables)
        fltFunction = lambdify(lstVariables,parse_expr(strFunction))
        arrFunction = lambda X : fltFunction(X[0],X[1],X[2])
        arrLess = np.array(list(map(arrFunction, self.__RealPoints)))
        lstDeletedIndices = np.where(arrLess > 0)[0]
        self.__DeletePoints(lstDeletedIndices)
    def GetQuaternionOrientation(self)->np.array:
        return gf.FCCQuaternionEquivalence(gf.GetQuaternionFromBasisMatrix(np.transpose(self.GetUnitBasisVectors())))     
    def GetLinearConstraints(self):
        return self.__LinearConstraints
    def GetLatticeConstraints(self):
        return self.__LatticeConstraints
class ExtrudedRectangle(GeneralLattice):
    def __init__(self, fltLength: float, fltWidth: float, fltHeight: float, inBasisVectors: np.array, inCellNodes: np.array ,inLatticeParameters: np.array, inOrigin: np.array, inCellBasis= None):
        arrConstraints = np.zeros([6,4])
        arrConstraints[0] = np.array([1,0,0,fltLength])
        arrConstraints[1] = np.array([-1,0,0,0])
        arrConstraints[2] = np.array([0,1,0,fltWidth])
        arrConstraints[3] = np.array([0,-1,0,0])
        arrConstraints[4] = np.array([0,0,1,fltHeight])
        arrConstraints[5] = np.array([0,0,-1,0])
        GeneralLattice.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
        self.MakeRealPoints(arrConstraints)
    
class ExtrudedRegularPolygon(GeneralLattice):
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
        GeneralLattice.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters,inOrigin, inCellBasis)
        self.MakeRealPoints(arrConstraints)



class SimulationCell(object):
    def __init__(self, inBoxVectors: np.array):
        self.Dimensions = len(inBoxVectors[0])
        self.BoundaryTypes = ['p']*self.Dimensions #assume periodic boundary conditions as a default
        self.SetOrigin(np.zeros(self.Dimensions))
        self.dctGrains = dict() #dictionary of RealGrain objects which form the simulation cell
        self.SetParallelpipedVectors(inBoxVectors)
        self.blnPointsAreWrapped = False
        self.__FileHeader = ''
        self.GrainList = []
    def AddGrain(self,inGrain, strName = None):
        if strName is None:
            strName = str(len(self.dctGrains.keys())+1)
        self.dctGrains[strName] = inGrain
        self.GrainList = list(self.dctGrains.keys())
    def GetGrain(self, strName: str):
        return self.dctGrains[strName]
    def GetNumberOfGrains(self)->int:
        return len(self.GrainList)
    def GetTotalNumberOfAtoms(self):
        if self.blnPointsAreWrapped:
            intNumberOfAtoms = len(self.__UniqueRealPoints)
        else: 
            intNumberOfAtoms = 0
            for j in self.GrainList:
                intNumberOfAtoms += self.GetGrain(j).GetNumberOfAtoms()
        return intNumberOfAtoms
    def GetRealBasisVectors(self):
        return self.__BasisVectors
    def SetBoundaryTypes(self,inList: list): #boundaries can be periodic 'p' or fixed 'f'
        self.BoundaryTypes = inList    
    def GetNumberOfAtomTypes(self):
        lstAtomTypes = []
        for j in self.GrainList:
            intCurrentAtomType = self.GetGrain(j).GetAtomType()
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
    def SetFileHeader(self, inString: str):
        self.__FileHeader = inString
    def WrapVectorIntoSimulationBox(self, inVector: np.array)->np.array:
        return gf.WrapVectorIntoSimulationCell(self.__BasisVectors, self.__InverseBasis, inVector)
    def WriteLAMMPSDataFile(self,inFileName: str):        
        now = datetime.now()
        strDateTime = now.strftime("%d/%m/%Y %H:%M:%S")
        with open(inFileName, 'w') as fdata:
            fdata.write('## ' + strDateTime + ' ' + self.__FileHeader + '\n')
            fdata.write('{} atoms\n'.format(self.GetTotalNumberOfAtoms()))
            fdata.write('{} atom types\n'.format(self.GetNumberOfAtomTypes()))
            fdata.write('{} {} xlo xhi\n'.format(self.__xlo,self.__xhi))
            fdata.write('{} {} ylo yhi\n'.format(self.__ylo,self.__yhi))
            if self.Dimensions == 3:
                fdata.write('{} {} zlo zhi\n'.format(self.__zlo,self.__zhi))
                fdata.write('{}  {} {} xy xz yz \n'.format(self.__xy,self.__xz,self.__yz))
            elif self.Dimensions ==2:
                fdata.write('{}  xy \n'.format(self.__xy))
            fdata.write('\n')
            fdata.write('Atoms\n\n')
            if self.blnPointsAreWrapped:
                for j in range(len(self.__UniqueRealPoints)):
                    fdata.write('{} {} {} {} {}\n'.format(j+1,self.__AtomTypes[j], *self.__UniqueRealPoints[j]))
            else:
                count = 1
                for j in self.GrainList:
                    for position in self.GetGrain(j).GetRealPoints():
                        fdata.write('{} {} {} {} {}\n'.format(count,self.GetGrain(j).GetAtomType(), *position))
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
            self.__BasisVectors = np.array([[self.__xhi,0,0],[self.__xy, self.__yhi,0],[self.__xz, self.__yz,self.__zhi]])
            lstBasisVectors = []
            for j in self.__BasisVectors:
                lstBasisVectors.append(gf.NormaliseVector(j))
            self.__UnitBasisVectors = np.array(lstBasisVectors)
            self.__InverseUnitBasis = np.linalg.inv(self.__UnitBasisVectors)
            self.__InverseBasis = np.linalg.inv(self.__BasisVectors)
    def WrapAllPointsIntoSimulationCell(self, intRound = 1)->np.array:
        lstUniqueRowindices = []
        arrAllAtoms = np.zeros([self.GetTotalNumberOfAtoms(),self.Dimensions])
        arrAllAtomTypes = np.ones([self.GetTotalNumberOfAtoms()],dtype=np.int8)
        i = 0
        for j in self.GrainList:
            for fltPoint in self.GetGrain(j).GetRealPoints():
                arrAllAtomTypes[i] = self.GetGrain(j).GetAtomType()
                arrAllAtoms[i] = fltPoint
                i = i + 1
        arrAllAtoms = np.round(self.WrapVectorIntoSimulationBox(arrAllAtoms),intRound)
        self.__UniqueRealPoints,lstUniqueRowindices = np.unique(arrAllAtoms,axis=0,return_index=True)
        self.__AtomTypes = arrAllAtomTypes[lstUniqueRowindices]  
        self.blnPointsAreWrapped = True
    def PlotSimulationCellAtoms(self):
        if self.blnPointsAreWrapped:
            return tuple(zip(*self.__UniqueRealPoints))
    def RemovePlaneOfAtoms(self, inPlane: np.array, fltTolerance: float):
        arrPointsOnPlane = gf.CheckLinearEquality(np.round(self.__UniqueRealPoints,10), inPlane,fltTolerance)
        self.__UniqueRealPoints = np.delete(self.__UniqueRealPoints,arrPointsOnPlane, axis=0)   
    def GetRealPoints(self)->np.array:
        if self.blnPointsAreWrapped:
            return self.__UniqueRealPoints
        else:
            raise("Error: Points need to be wrapped into simulation cell")

class DefectMeshObject(object):
    def __init__(self,inMeshPoints: np.array, intID: int):
        self.__MeshPoints = inMeshPoints
        self.__ID = intID
        self.__AtomIDs = []
        self.__AdjustedMeshPoints = [] #these are general adjusted to correct for the limited accuracy of the QuantisedCuboid object
        self.__Volume = 0
        self.__PE = 0
        self.__WrappedMeshPoints = []
    def GetMeshPoints(self):
        return np.copy(self.__MeshPoints)
    def GetWrappedMeshPoints(self):
        return np.copy(self.__WrappedMeshPoints)
    def SetWrappedMeshPoints(self, inPoints):
        self.__WrappedMeshPoints = inPoints
    def SetGlobalID(self, intID):
        self.__GlobalID = intID
    def GetGlobalID(self):
        return self.__GlobalID 
    def GetID(self):
        return self.__ID
    def GetNumberOfAtoms(self):
        return len(self.__AtomIDs)
    def SetAtomIDs(self, inlstIDs: list):
        self.__AtomIDs = list(map(int,inlstIDs))
    def GetAtomIDs(self)->list:
        return cp.copy(self.__AtomIDs)
    def AddAtomIDs(self, inList):
        inList = list(map(int,inList))
        self.__AtomsIDs = list(np.unique(self.__AtomIDs.extend(inList)))
    def RemoveAtomIDs(self, inList):
        setAtomIDs = set(self.__AtomIDs)
        self.__AtomIDs = list(setAtomIDs.difference(inList))
    def SetMeshPoints(self, inPoints):
        self.__MeshPoints = inPoints
    def SetAdjustedMeshPoints(self, inPoints: np.array):
        self.__AdjustedMeshPoints = inPoints
    def GetAdjustedMeshPoints(self)->np.array:
        return np.copy(self.__AdjustedMeshPoints)
    def SetVolume(self, fltVolume):
        self.__Volume = fltVolume 
    def GetVolume(self):
        return self.__Volume
    def GetAtomicDensity(self):
        if self.__Volume > 0 and len(self.__AtomIDs) > 0:
            return len(self.__AtomIDs)/self.__Volume
        else:
            return 0   
    def SetPeriodicDirections(self, inList):
        self.__PeriodicDirections = inList
    def GetPeriodicDirections(self):
        return cp.copy(self.__PeriodicDirections)
    def GetTotalPE(self):
        return self.__PE
    def SetTotalPE(self, fltPE):
        self.__PE = fltPE 
    def GetPEPerAtom(self):
        if self.GetNumberOfAtoms() > 0:
            return self.__PE/len(self.__AtomIDs)
    def GetPEPerVolume(self):
        if self.__Volume > 0:
            return self.__PE/self.__Volume
    

class GeneralJunctionLine(DefectMeshObject):
    def __init__(self,inMeshPoints: np.array, intID: int):
        DefectMeshObject.__init__(self,inMeshPoints, intID)
        self.__AdjacentGrains = []
        self.__AdjacentGrainBoundaries = []
        self.__PeriodicDirections = []
    def SetAdjacentGrains(self, inList):
        self.__AdjacentGrains = inList
    def GetAdjacentGrains(self)->list:
        return cp.copy(self.__AdjacentGrains)
    def SetAdjacentGrainBoundaries(self, inList):
        self.__AdjacentGrainBoundaries = inList
    def GetAdjacentGrainBoundaries(self)->list:
        return cp.copy(self.__AdjacentGrainBoundaries)
   
class GeneralGrainBoundary(DefectMeshObject):
    def __init__(self,inMeshPoints: np.array, intID: str):
        DefectMeshObject.__init__(self,inMeshPoints, intID)
        self.__AdjacentGrains = []
        self.__AdjacentJunctionLines = []
        self.__AdjacentGrainBoundaries = []
        self.__PeriodicDirections = [] 
        self.__GlobalAdjacentJunctionLines = []   
    def SetAdjacentGrains(self, inList):
        self.__AdjacentGrains = inList
    def GetAdjacentGrains(self)->list:
        return cp.copy(self.__AdjacentGrains) 
    def SetAdjacentJunctionLines(self, inList):
        self.__AdjacentJunctionLines = inList
    def GetAdjacentJunctionLines(self)->list:
        return cp.copy(self.__AdjacentJunctionLines)
   
class DefectObject(object):
    def __init__(self, fltTimeStep: float):
        self.__TimeStep = fltTimeStep
        self.__dctJunctionLines = dict()
        self.__dctGrainBoundaries = dict()
        self.__dctGlobalJunctionLines = dict()
        self.__dctGlobalGrainBoundaries =dict()
        self.__MapToGlobalID = []
    def SetMapToGlobalID(self, inList):
        self.__MapToGlobalID = inList
    def GetMapToGlobalID(self):
        return cp.copy(self.__MapToGlobalID)
    def AddJunctionLine(self, objJunctionLine: GeneralJunctionLine):
        self.__dctJunctionLines[objJunctionLine.GetID()] = objJunctionLine 
    def AddGrainBoundary(self, objGrainBoundary: GeneralGrainBoundary):
        self.__dctGrainBoundaries[objGrainBoundary.GetID()] = objGrainBoundary
    def AddGlobalJunctionLine(self, objJunctionLine: GeneralJunctionLine):
        self.__dctGlobalJunctionLines[objJunctionLine.GetGlobalID()] = objJunctionLine
    def AddGlobalGrainBoundary(self, objGrainBoundary: GeneralGrainBoundary):
        self.__dctGlobalGrainBoundaries[objGrainBoundary.GetGlobalID()] = objGrainBoundary    
    def GetGlobalJunctionLine(self, intGlobalKey: int):
        return self.__dctGlobalJunctionLines[intGlobalKey]
    def GetJunctionLine(self, intLocalKey: int):
        return self.__dctJunctionLines[intLocalKey]
    def GetGrainBoundary(self, intLocalKey):
        return self.__dctGrainBoundaries[intLocalKey]
    def GetGlobalGrainBoundary(self, intGlobalKey: int):
        return self.__dctGlobalGrainBoundaries[intGlobalKey]      
    def GetJunctionLineIDs(self):
        return list(self.__dctJunctionLines.keys())
    def GetGrainBoundaryIDs(self):
        return list(self.__dctGrainBoundaries.keys())
    def GetGlobalJunctionLineIDs(self):
        return list(self.__dctGlobalJunctionLines.keys()) 
    def GetGlobalGrainBoundaryIDs(self):
        return list(self.__dctGlobalGrainBoundaries.keys())
    def GetAdjacentGlobalGrainBoundaries(self, intGlobalJunctionLine: int)->list:
        lstGlobalAdjacentGrainBoundaries = []
        for j in self.GetGlobalJunctionLine(intGlobalJunctionLine).GetAdjacentGrainBoundaries():
            lstGlobalAdjacentGrainBoundaries.append(self.GetGrainBoundary(j).GetGlobalID())
        return lstGlobalAdjacentGrainBoundaries
    def GetAdjacentGlobalJunctionLines(self, intGlobalGrainBoundary: int)->list:
        lstGlobalAdjacentJunctionLines = []
        for j in self.GetGlobalGrainBoundary(intGlobalGrainBoundary).GetAdjacentJunctionLines():
            lstGlobalAdjacentJunctionLines.append(self.GetJunctionLine(j).GetGlobalID())
        return lstGlobalAdjacentJunctionLines