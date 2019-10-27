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
    def GetRealPoints(self)->np.array:
        return self.__RealPoints
    def GeneratePoints(self, inConstraints: np.array):
        self.__LinearConstraints = inConstraints
        lstCellPoints = [np.zeros([self.Dimensions()])] #this is always an initial lattice point
        lstUsedCellPoints = []
        counter = 0
        while (len(lstCellPoints) > 0 and counter < max(1,len(self.__CellPoints))):
            arrInitialPoint = lstCellPoints[0]
            lstUsedCellPoints.extend(self.ExtendPoints(arrInitialPoint,0,1))
            lstUsedCellPoints.extend(self.ExtendPoints(arrInitialPoint,0,-1))
            for arrPoint in lstUsedCellPoints:
                    for intDirection in range(1,self.Dimensions()):
                        self.ExtendPoints(arrPoint,intDirection,1)
                        self.ExtendPoints(arrPoint,intDirection,-1)
            lstUsedCellPoints = np.unique(lstUsedCellPoints,axis=0)
            if counter == 0:
                lstCellPoints = self.__CellPoints
            lstCellPoints = self.__RemoveUsedCellPoints(lstCellPoints, lstUsedCellPoints)
            lstUsedCellPoints = []
            counter += 1
        self.__RealPoints = np.add([self.__Origin], self.__RealPoints)
    def __RemoveUsedCellPoints(self,lstCellPoints: list, lstUsedCellPoints: list):
        lstDeletedIndices = []
        for j in lstUsedCellPoints:
            arrIndex = np.argwhere(np.all(lstCellPoints == j, axis = 1))
            if len(arrIndex) > 0:
                lstDeletedIndices.extend(arrIndex[0])
        return np.delete(lstCellPoints, lstDeletedIndices, axis=0)
    def ExtendPoints(self,inCellPoint:np.array, intCoordinateDirection: int, intSign: int)->list:
            lstRealPoints = []
            lstCellPoints = []
            lstLatticePoints = []
            arrIndicesToDelete = []
            counter = 0
            arrMotif = self.CellDirectionalMotif(intCoordinateDirection,intSign)
            blnEnd = False
            while (not blnEnd):
                lstCellPoints.extend([np.add(inCellPoint,intSign*counter*self.UnitVector(intCoordinateDirection))])
                arrPointsToAdd = np.add(lstCellPoints[-1], arrMotif)
                arrIndicesToDelete = self.CheckLinearConstraints(np.matmul(arrPointsToAdd,self.GetRealBasisVectors())) 
                arrReturnPoints = np.delete(arrPointsToAdd,arrIndicesToDelete, axis=0)
                if len(arrReturnPoints) > 0:
                    lstRealPoints.extend(np.matmul(arrReturnPoints, self.GetRealBasisVectors()))
                    lstLatticePoints.extend(arrPointsToAdd)
                    counter += 1
                else:
                    blnEnd = True
                    lstCellPoints = lstCellPoints[:-1]
            if len(self.__RealPoints) == 0:
                self.__RealPoints = np.array(lstRealPoints)
            elif len(lstRealPoints) > 0:
                self.__RealPoints = np.unique(np.append(self.__RealPoints,np.array(lstRealPoints), axis=0),axis=0)
            if len(self.__LatticePoints) == 0:
                self.__LatticePoints = np.array(lstLatticePoints)
            elif len(lstLatticePoints) > 0:
                self.__LatticePoints = np.unique(np.append(self.__LatticePoints, np.array(lstLatticePoints), axis=0),axis=0)
            if len(self.__CellPoints) ==0:
                self.__CellPoints = np.array(lstCellPoints)
            elif len(lstCellPoints)> 0:
                self.__CellPoints = np.unique(np.append(self.__CellPoints, np.array(lstCellPoints), axis=0),axis=0)
            return lstCellPoints
    def CheckLinearConstraints(self,inPoints: np.array)-> np.array: #returns indices to delete   
        arrPositions = np.subtract(np.matmul(inPoints, np.transpose(self.__LinearConstraints[:,:-1])), np.transpose(self.__LinearConstraints[:,-1]))
        arrPositions = np.argwhere(arrPositions > 0)[:,0]        
        return arrPositions
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

class SimulationCell(object):
    def __init__(self, inBoxVectors: np.array):
        self.Dimensions = len(inBoxVectors[0])
        self.BoundaryTypes = ['p']*self.Dimensions #assume periodic boundary conditions as a default
        self.SetOrigin(np.zeros(self.Dimensions))
        self.GrainList = [] #list of RealGrain objects which form the simulation cell
        self.SetParallelpipedVectors(inBoxVectors)
        self.NoDuplicates = False
    def AddGrain(self,inGrain):
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
            if self.NoDuplicates:
                for j in range(len(self.__UniqueRealPoints)):
                    fdata.write('{} {} {} {} {}\n'.format(j+1,self.__AtomTypes[j], *self.WrapVectorIntoSimulationBox(self.__UniqueRealPoints[j])))
            else:
                count = 1
                for j in self.GrainList:
                    for position in j.GetRealPoints():
                        fdata.write('{} {} {} {} {}\n'.format(count,j.GetAtomType(), *self.WrapVectorIntoSimulationBox(position)))
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
    def RemoveDuplicateAtoms(self)->np.array:
        lstUniqueRowindices = []
        arrAllAtoms = np.zeros([self.GetTotalNumberOfAtoms(),self.Dimensions])
        arrAllAtomTypes = np.ones([self.GetTotalNumberOfAtoms()],dtype=np.int8)
        i = 0
        for objGrain in self.GrainList:
            for fltPoint in objGrain.GetRealPoints():
                arrAllAtomTypes[i] = objGrain.GetAtomType()
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
            lstPlanes.append(gf.FindPlane(self.__BoxVectors[0], self.__BoxVectors[1], self.__BoxVectors[2]))
            lstPlanes.append(gf.FindPlane(-self.__BoxVectors[0], self.__BoxVectors[1], self.__BoxVectors[0]))
            for j in lstPlanes:
                for k in self.GrainList:
                    k.LinearConstrainRealPoints(j)

#a = 4.05 ##lattice parameter
a = 4.05*np.sqrt(3) #periodic cell repeat multiple
h= 4
l=5
MyLattice = GeneralLattice(gf.RotatedBasisVectors(np.arccos(1/np.sqrt(3)), np.array([1,-1,0])/np.sqrt(2)),ld.FCCCell,np.array([a,a,a]))
MyLattice.SetOrigin(np.array([0,0,0]))
MyLattice.GeneratePoints(np.array([[1,0,0,5*a],[-1,0,0,0],[0,1,0,5*a],[0,-1,0,0],[0,0,1,(h-1)*a],[0,0,-1,0]]))
MyLattice.RemovePlaneOfAtoms([0,0,1,h*a])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
z = a*np.array([0,0,h])
T0 = np.array([-l/2, -l*np.sqrt(3)/2,0]) #down and left
T1 = np.array([l,0,0])
T2 = np.array([-l/2, l*np.sqrt(3)/2, 0]) #up and left
arrDiagonalVector =  -a*T0
MySimulationCell = SimulationCell(np.array([2*a*T1,3*arrDiagonalVector, z-0.5*np.array([0,0,a])])) 
MySimulationCell.AddGrain(MyLattice)
#MySimulationCell.ApplySimulationCellConstraint()
#MySimulationCell.RemoveDuplicateAtoms()
print(MySimulationCell.GetRealBasisVectors())
ax.scatter(*MyLattice.MatLabPlot())
#ax.quiver(*np.zeros([3,3]),*gf.RotatedBasisVectors(-np.pi/4, np.array([1,-1,0])),length=0.1)
plt.show()
MySimulationCell.WriteLAMMPSDataFile('new.data')
