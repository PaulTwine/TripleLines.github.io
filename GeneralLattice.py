import numpy as np
import GeometryFunctions as gf
import LatticeShapes as ls
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
fig = plt.figure()
ax = fig.gca(projection='3d')

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
    def NumberOfCellNodes(self):
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
            arrDistances = np.zeros(self.NumberOfCellNodes)
            for j,arrNode in enumerate(self.__CellNodes):
                arrDistances[j] = gf.RealDistance(inNodePoint, arrNode)
            intMinIndex = np.argmin(arrDistances)
        else:
            raise Exception('Cell co-ordinates must range from 0 to 1 inclusive')
        return self.__CellNodes[intMinIndex] 
class RealCell(PureCell):
    def __init__(self, inBasisVectors: np.array,inCellNodes: np.array, inLatticeParameters: np.array):
        PureCell.__init__(self, inCellNodes)
        self.__UnitBasisVectors = inBasisVectors
        self.__LatticeParameters = inLatticeParameters
        self.__RealBasisVectors = np.matmul(np.diag(inLatticeParameters), inBasisVectors)
    def RealDirectionalMotif(self, intBasisVector: int, intSign = +1)->np.array:
        return np.matmul(self.CellDirectionalMotif(intBasisVector, intSign), self.__UnitBasisVectors)
    def GetUnitBasisVectors(self)->np.array:
        return self.__UnitBasisVectors
    def GetRealBasisVectors(self)->np.array:
        return self.__RealBasisVectors

class GeneralLattice(RealCell):
    def __init__(self,inBasisVectors:np.array,inCellNodes: np.array,inLatticeParameters:np.array):
        RealCell.__init__(self,inBasisVectors, inCellNodes,inLatticeParameters)
        self.__RealPoints = []
        self.__LatticePoints = []
        self.__CellPoints = []
        self.__AtomType = 1
        self.__Origin = np.zeros(self.Dimensions())
        self.__LatticeParameters = inLatticeParameters
    def GetRealPoints(self)->np.array:
        return self.__RealPoints
    def MakeRealPoints(self, inConstraints):
        self.__LinearConstraints = inConstraints
        self.GenerateLatticeConstraints(inConstraints)
        arrBounds = self.FindBoundingBox(self.__LatticeConstraints)
        arrLatticePoints = np.array(gf.CreateCuboidPoints(arrBounds))
        for j in self.GetCellNodes():
            arrLatticePoints =np.add(j,arrLatticePoints)
        arrRealPoints = np.matmul(arrLatticePoints, self.GetUnitBasisVectors())
        arrRealPoints = np.delete(arrRealPoints,self.CheckLinearConstraints(arrRealPoints),axis=0)
        self.__RealPoints = np.add(self.__Origin,arrRealPoints)
        #arrLatticePoints = np.delete(arrLatticePoints,self.CheckLatticeConstraints(arrLatticePoints),axis=0)
       # self.__LatticePoints = np.unique(arrLatticePoints, axis=0)
        #self.GenerateRealPoints(self.__LatticePoints)
    # def GeneratePoints(self, inConstraints: np.array):
    #     self.__LinearConstraints = inConstraints
    #     self.GenerateLatticeConstraints(inConstraints)
    #     setUsedCellPoints = set()
    #     setNewCellPoints = {(0,0,0)}
    #     counter = 0
    #     blnFirstTime = True
    #    # while (len(lstCellPoints) > 0 and counter < max(1,len(self.__CellPoints))):
    #     while (len(setNewCellPoints) > 0):
           
    #         arrPoint = setNewCellPoints.pop()
    #             #arrCellsRemaining = self.__RemoveUsedCellPoints(lstNewCellPoints, lstUsedCellPoints)
    #         # for arrPoint in lstNewCellPoints:
    #         #     lstUsedCellPoints.extend(self.ExtendPoints(arrPoint,0,1))
    #         #     lstUsedCellPoints.extend(self.ExtendPoints(arrPoint,0,-1))
    #         # if len(lstUsedCellPoints) > 0:
    #         #     lstUsedCellPoints = np.unique(lstUsedCellPoints,axis=0)
    #         #for arrPoint in lstUsedCellPoints:
    #         # for arrPoint in lstUsedCellPoints:
    #         #     lstUsedCellPoints.extend(self.ExtendPoints(arrPoint,0,1))
    #         #     lstUsedCellPoints.extend(self.ExtendPoints(arrPoint,0,-1))
    #         # # if l
    #         self.ExtendPoints(arrPoint,setUsedCellPoints,0,1)
    #         self.ExtendPoints(arrPoint,setUsedCellPoints,0,-1)
    #         for j in (setUsedCellPoints-setNewCellPoints):
    #             for intDirection in range(1,self.Dimensions()):
    #                 self.ExtendPoints(j,setNewCellPoints, intDirection,1)
    #                 self.ExtendPoints(j,setNewCellPoints,intDirection,-1)
    #         setNewCellPoints = setNewCellPoints - setUsedCellPoints
    #         #if blnFirstTime:
    #         #    lstCellPoints = self.__CellPoints
    #         #    blnFirstTime = False
    #         # lstNewCellPoints = list(self.__RemoveUsedCellPoints(lstNewCellPoints, lstUsedCellPoints))
    #         # lstUsedCellPoints = []
    #         counter += 1
    #     self.GenerateRealPoints(self.__LatticePoints)
    def __RemoveUsedCellPoints(self,lstCellPoints: list, lstUsedCellPoints: list):
        lstDeletedIndices = []
        for j in lstUsedCellPoints:
            arrIndex = np.argwhere(np.all(lstCellPoints == j, axis = 1))
            if len(arrIndex) > 0:
                lstDeletedIndices.extend(arrIndex[0])
        return np.delete(lstCellPoints, lstDeletedIndices, axis=0)
    def GenerateRealPoints(self, inLatticePoints):
        self.__RealPoints = np.matmul(inLatticePoints, self.GetRealBasisVectors())
        self.__RealPoints = np.add([self.__Origin], self.__RealPoints)
    def ExtendPoints(self,inCellPoint:np.array,setCellPoints: set(), intCoordinateDirection: int, intSign: int)->list:
            lstRealPoints = []
            lstLatticePoints = []
            arrIndicesToDelete = []
            counter = 0
            arrMotif = self.CellDirectionalMotif(intCoordinateDirection,intSign)
            blnEnd = False
            while (not blnEnd):
                #lstCellPoints.extend([np.add(inCellPoint,intSign*counter*self.UnitVector(intCoordinateDirection))])
                arrPoint = np.add(inCellPoint,intSign*counter*self.UnitVector(intCoordinateDirection))
                setCellPoints.add(tuple(arrPoint))
                arrPointsToAdd = np.add(arrPoint, arrMotif)
                arrIndicesToDelete = self.CheckLinearConstraints(np.matmul(arrPointsToAdd,self.GetRealBasisVectors())) 
                #arrIndicesToDelete = self.CheckLatticeConstraints(arrPointsToAdd)
                arrReturnPoints = np.delete(arrPointsToAdd,arrIndicesToDelete, axis=0)
                if len(arrReturnPoints) > 0:
                   # lstRealPoints.extend(np.matmul(arrReturnPoints, self.GetRealBasisVectors()))
                    lstLatticePoints.extend(arrReturnPoints)
                    counter += 1
                else:
                    blnEnd = True
                    #lstCellPoints = lstCellPoints[:-1]
            #self.__RealPoints = self.__AppendNewItems(list(self.__RealPoints), lstRealPoints)[0]
            #self.__LatticePoints = self.__AppendNewItems(list(self.__LatticePoints), lstLatticePoints)[0]
            #self.__CellPoints, intNumberOfCellPointsAdded = self.__AppendNewItems(list(self.__CellPoints), lstCellPoints)          
            # if len(self.__RealPoints) == 0:
            #     self.__RealPoints = np.array(lstRealPoints)
            # elif len(lstRealPoints) > 0:
            #     self.__RealPoints = np.unique(np.append(self.__RealPoints,np.array(lstRealPoints), axis=0),axis=0)
            if len(self.__LatticePoints) == 0:
                 self.__LatticePoints = np.array(lstLatticePoints)
            elif len(lstLatticePoints) > 0:
                 self.__LatticePoints = np.unique(np.append(self.__LatticePoints, np.array(lstLatticePoints), axis=0),axis=0)
            # if len(self.__CellPoints) ==0:
            #     self.__CellPoints = np.array(lstCellPoints)
            # elif len(lstCellPoints)> 0:
            #     self.__CellPoints = np.unique(np.append(self.__CellPoints, np.array(lstCellPoints), axis=0),axis=0)
            #return setCellPoints
    def GenerateLatticeConstraints(self, inConstraints: np.array):
        rtnArray = np.zeros([len(inConstraints),len(inConstraints[0])])
        for k in range(len(inConstraints)):
            arrVector = np.matmul(inConstraints[k,:-1], np.linalg.inv(self.GetUnitBasisVectors()))
            for j in range(len(arrVector)):
                rtnArray[k,j] = arrVector[j]
            #rtnArray[k, 3] = inConstraints[k,3]
            rtnArray[k,3] = np.linalg.norm(np.matmul(inConstraints[k,:-1]*inConstraints[k,3],np.linalg.inv(np.diag(self.__LatticeParameters))),axis=0)
        self.__LatticeConstraints = rtnArray
    def CheckLinearConstraints(self,inPoints: np.array)-> np.array: #returns indices to delete for real coordinates  
        arrPositions = np.subtract(np.matmul(inPoints, np.transpose(self.__LinearConstraints[:,:-1])), np.transpose(self.__LinearConstraints[:,-1]))
        arrPositions = np.argwhere(arrPositions > 0)[:,0]        
        return arrPositions
    def CheckLatticeConstraints(self,inPoints: np.array)-> np.array: #returns indices to delete   
        arrPositions = np.subtract(np.matmul(inPoints, np.transpose(self.__LatticeConstraints[:,:-1])), np.transpose(self.__LatticeConstraints[:,-1]))
        arrPositions = np.argwhere(arrPositions > 0)[:,0]        
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
        self.__RealPoints = np.delete(self.__RealPoints, lstDeletedIndices, axis=0)
        self.__LatticePoints  = np.delete(self.__LatticePoints, lstDeletedIndices, axis=0)  
    def RemovePlaneOfAtoms(self, inPlane: np.array):
        arrPointsOnPlane = gf.CheckLinearEquality(self.__RealPoints, inPlane)
        self.__RealPoints = arrPointsOnPlane
        return arrPointsOnPlane  
    #FindBoundingBox only works for linear constraints. Searches for all the vertices where three constraints #simultaneously apply and then finds the points furthers from the origin.
    def FindBoundingBox(self,inConstraints: np.array)->np.array:
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
                    if abs(np.linalg.det(arrMatrix[:,:-1])) > 0.0001:
                        arrPoints[counter] = np.matmul(np.linalg.inv(arrMatrix[:,:-1]),arrMatrix[:,-1])
                        counter += 1
        for j in range(len(arrRanges)):
            arrRanges[j,0] = np.min(arrPoints[:,j])
            arrRanges[j,1] = np.max(arrPoints[:,j])
        return(arrRanges)
                    
class ExtrudedRegularPolygon(GeneralLattice):
    def __init__(self, fltSideLength: float, fltHeight: float, intNumberOfSides: int, inBasisVectors: np.array, inCellNodes: np.array, inLatticeParameters: np.array, inOrigin = None):
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
        GeneralLattice.__init__(self,inBasisVectors, inCellNodes, inLatticeParameters)
        if not(inOrigin is None):
            self.SetOrigin(inOrigin)
        self.MakeRealPoints(arrConstraints)



class SimulationCell(object):
    def __init__(self, inBoxVectors: np.array):
        self.Dimensions = len(inBoxVectors[0])
        self.BoundaryTypes = ['p']*self.Dimensions #assume periodic boundary conditions as a default
        self.SetOrigin(np.zeros(self.Dimensions))
        self.GrainList = [] #list of RealGrain objects which form the simulation cell
        self.SetParallelpipedVectors(inBoxVectors)
        self.blnPointsAreWrapped = False
    def AddGrain(self,inGrain):
        self.GrainList.append(inGrain)
    def GetGrain(self, intGrainIndex: int):
        return self.GrainList[intGrainIndex]
    def GetNumberOfGrains(self)->int:
        return len(self.GrainList)
    def GetTotalNumberOfAtoms(self):
        if self.blnPointsAreWrapped:
            intNumberOfAtoms = len(self.__UniqueRealPoints)
        else: 
            intNumberOfAtoms = 0
            for j in self.GrainList:
                intNumberOfAtoms += j.GetNumberOfAtoms()
        return intNumberOfAtoms
    def GetRealBasisVectors(self):
        return self.__BasisVectors
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
    def WrapVectorIntoSimulationBox(self, inVector: np.array)->np.array:
        return gf.WrapVectorIntoSimulationCell(self.__BasisVectors, self.__InverseBasis, inVector)
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
            if self.blnPointsAreWrapped:
                for j in range(len(self.__UniqueRealPoints)):
                    fdata.write('{} {} {} {} {}\n'.format(j+1,self.__AtomTypes[j], *self.__UniqueRealPoints[j]))
            else:
                count = 1
                for j in self.GrainList:
                    for position in j.GetRealPoints():
                        fdata.write('{} {} {} {} {}\n'.format(count,j.GetAtomType(), *position))
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
            self.__InverseBasis = np.linalg.inv(self.__BasisVectors)
    def WrapAllPointsIntoSimulationCell(self)->np.array:
        lstUniqueRowindices = []
        arrAllAtoms = np.zeros([self.GetTotalNumberOfAtoms(),self.Dimensions])
        arrAllAtomTypes = np.ones([self.GetTotalNumberOfAtoms()],dtype=np.int8)
        i = 0
        for objGrain in self.GrainList:
            for fltPoint in objGrain.GetRealPoints():
                arrAllAtomTypes[i] = objGrain.GetAtomType()
                arrAllAtoms[i] = fltPoint
                i = i + 1
        self.__UniqueRealPoints,lstUniqueRowindices = np.unique(self.WrapVectorIntoSimulationBox(arrAllAtoms),axis=0,return_index=True)
        self.__AtomTypes = arrAllAtomTypes[lstUniqueRowindices]  
        self.blnPointsAreWrapped = True
    def ApplySimulationCellConstraint(self):
        lstPlanes = []
        if self.Dimensions == 3:
            lstPlanes.append(gf.FindPlane(self.__BoxVectors[0], self.__BoxVectors[2], self.__Origin))
            lstPlanes.append(gf.FindPlane(-self.__BoxVectors[0], self.__BoxVectors[2], self.__BoxVectors[1]))
            lstPlanes.append(gf.FindPlane(-self.__BoxVectors[1], self.__BoxVectors[2], self.__Origin))
            lstPlanes.append(gf.FindPlane(self.__BoxVectors[1], self.__BoxVectors[2], self.__BoxVectors[0]))
            lstPlanes.append(gf.FindPlane(self.__BoxVectors[0], self.__BoxVectors[1], self.__BoxVectors[2]))
            lstPlanes.append(gf.FindPlane(-self.__BoxVectors[0], self.__BoxVectors[1], self.__BoxVectors[0]))
            for j in lstPlanes:
                for k in self.GrainList:
                    k.LinearConstrainRealPoints(j)
    def PlotSimulationCellAtoms(self):
        if self.blnPointsAreWrapped:
            return zip(*self.__UniqueRealPoints)

#a = 4.05 ##lattice parameter
a = 4.05*np.sqrt(3) #periodic cell repeat multiple
l = 8
h= 2
z = a*np.array([0,0,h])
#MyLattice = GeneralLattice(gf.RotateVectors(2,z,gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)),z)),ld.FCCCell,np.array([a,a,a]))
#MyLattice.SetOrigin(np.array([70,15,0]))
#MyLattice.GeneratePoints(np.array([[1,0,0,2*l*a],[-1,0,0,0],[0,1,0,l*a],[0,-1,0,0],[0,0,1,h*a],[0,0,-1,0]]))
#MyLattice.MakeRealPoints(np.array([[1,0,0,2*l*a],[-1,0,0,0],[0,1,0,l*a],[0,-1,0,0],[0,0,1,h*a],[0,0,-1,0]]))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
arr111BasisVectors = gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2))
arrHorizontalVector = np.array([l*a,0,0])
arrDiagonalVector =  np.array([a*l/2, a*l*np.sqrt(3)/2,0])
MySimulationCell = SimulationCell(np.array([3*arrHorizontalVector,3*arrDiagonalVector, z])) 
MyTri1 = ExtrudedRegularPolygon(l*a, h*a, 6, arr111BasisVectors, ld.FCCCell, np.array([a,a,a]),np.array([0,0,0]))
MyTri2 = ExtrudedRegularPolygon(l*a, h*a, 6, gf.RotateVectors(1.5,z, arr111BasisVectors), ld.FCCCell, np.array([a,a,a]),arrDiagonalVector+arrHorizontalVector)
MyTri3 = ExtrudedRegularPolygon(l*a, h*a, 6, arr111BasisVectors, ld.FCCCell, np.array([a,a,a]), arrHorizontalVector)
MySimulationCell.AddGrain(MyTri1)
MySimulationCell.AddGrain(MyTri2)
MySimulationCell.AddGrain(MyTri3)
#MySimulationCell.AddGrain(MyLattice)
MySimulationCell.WrapAllPointsIntoSimulationCell()
MySimulationCell.WriteLAMMPSDataFile('new.data')
ax.scatter(*MySimulationCell.PlotSimulationCellAtoms())
plt.show()
#print(MyLattice.GetNumberOfAtoms())
