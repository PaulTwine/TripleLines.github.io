import LAMMPSDump as  Ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import GeometryFunctions as gf
from scipy import spatial
class OVITOSPostProcess(object):
    def __init__(self,arrGrainQuaternions: np.array, objTimeStep: Ld.LAMMPSTimeStep, intLatticeType: int):
        self.__GrainOrientations = arrGrainQuaternions
        self.__NumberOfGrains = len(arrGrainQuaternions)
        self.__LAMMPSTimeStep = objTimeStep
        self.__Dimensions = objTimeStep.GetNumberOfDimensions()
        self.__LatticeStructure = intLatticeType #lattice structure type as defined by OVITOS
        self.__intStructureType = objTimeStep.GetColumnNames().index('StructureType')
        self.__intPositionX = objTimeStep.GetColumnNames().index('x')
        self.__intPositionY = objTimeStep.GetColumnNames().index('y')
        self.__intPositionZ = objTimeStep.GetColumnNames().index('z')
        self.__intQuarternionW = objTimeStep.GetColumnNames().index('OrientationW')
        self.__intQuarternionX = objTimeStep.GetColumnNames().index('OrientationX')
        self.__intQuarternionY = objTimeStep.GetColumnNames().index('OrientationY')
        self.__intQuarternionZ = objTimeStep.GetColumnNames().index('OrientationZ')
       # self.__PeriodicTranslations = objTimeStep.GetPeriodicTranslations()
        lstUnknownAtoms = []
        dctLatticeAtoms = dict()
        dctGrainPointsTree = dict()
        for j in range(self.__NumberOfGrains):
            dctLatticeAtoms[str(j)] = []    
        lstGBAtoms = []
        for j in range(objTimeStep.GetNumberOfAtoms()):
            arrCurrentRow = objTimeStep.GetRow(j)
            intGrainNumber = self.ReturnGrainIndex(arrCurrentRow)
            if (int(arrCurrentRow[self.__intStructureType]) == self.__LatticeStructure):    
                if  int(intGrainNumber) == -1:
                    lstUnknownAtoms.append(arrCurrentRow)
                else:
                    dctLatticeAtoms[str(intGrainNumber)].append(arrCurrentRow)
            else:
                lstGBAtoms.append(arrCurrentRow)
        self.__UnknownAtoms = lstUnknownAtoms
        self.__LatticeAtoms = dctLatticeAtoms
        self.__GBAtoms = lstGBAtoms
        for strGrainKey in dctLatticeAtoms.keys():
            dctGrainPointsTree[strGrainKey] = spatial.KDTree(list(zip(*self.PlotGrain(strGrainKey))))
        self.__dctGrainPointsTree = dctGrainPointsTree
    def ReturnUnknownAtoms(self):
        return self.__UnknownAtoms
    def ReturnGrainIndex(self, lstAtomRow: list)->int: #returns -1 if the atoms orientation doesn't match any lattice
        fltTest = 0
        arrAtom = np.array([lstAtomRow[self.__intQuarternionW],lstAtomRow[self.__intQuarternionX],lstAtomRow[self.__intQuarternionY],lstAtomRow[self.__intQuarternionZ]])
        arrAtom = gf.NormaliseVector(arrAtom)
        blnFound = False
        j = 0
        intIndex = -1 #will return an error if lstatom doesn't belong to any grain
        while  (j < self.__NumberOfGrains and not blnFound): 
            objQuarternion = self.__GrainOrientations[j]
            fltTest = np.dot(objQuarternion,gf.QuaternionConjugate(arrAtom)) 
            if 0.999 < abs(fltTest) < 1.001:
                intIndex = j
                blnFound = True
            j = j + 1
        return intIndex
    def PlotTripleLine(self):
        arrPoints = self.__TripleLine
        return arrPoints[:,0],arrPoints[:,1], arrPoints[:,2]
    def PlotGrain(self, strGrainNumber: str):
        return self.__PlotList(self.__LatticeAtoms[strGrainNumber])
    def PlotUnknownAtoms(self):
        return self.__PlotList(self.__UnknownAtoms)
    def PlotGBAtoms(self):
        return self.__PlotList(self.__GBAtoms)
    def PlotTriplePoints(self):
        arrPoints = self.__TriplePoints
        return arrPoints[:,:,0],arrPoints[:,:,1], arrPoints[:,:,2]
    def __PlotList(self, strList: list):
        arrPoints = np.array(strList)
        return arrPoints[:,self.__intPositionX], arrPoints[:,self.__intPositionY], arrPoints[:,self.__intPositionZ]
    def __GetCoordinates(self, strList: list):
        arrPoint = np.array(strList)
        return arrPoint[self.__intPositionX], arrPoint[self.__intPositionY], arrPoint[self.__intPositionZ]
    def FindClosestGrainPoint(self, arrPoint: np.array,strGrainKey: str)->np.array:
        arrPeriodicPoints = self.__LAMMPSTimeStep.PeriodicEquivalents(arrPoint)
        fltDistances, intIndices =  self.__dctGrainPointsTree[strGrainKey].query(arrPeriodicPoints)
        intMin = np.argmin(fltDistances)
        intDataIndex = intIndices[intMin]
        return self.__dctGrainPointsTree[strGrainKey].data[intDataIndex]
    def NumberOfGBAtoms(self)->int:
        return len(self.__GBAtoms)
    def FindTriplePoints(self):
        arrTriplePoints =np.zeros([self.NumberOfGBAtoms(),self.__NumberOfGrains,self.__Dimensions ])
        arrTripleLine = np.zeros([self.NumberOfGBAtoms(), self.__Dimensions])
        lstGrainBoundaryLength = []
        for n,GBAtom in enumerate(self.__GBAtoms):
            fltLengths = []
            for j,strGrainKey  in enumerate(self.__LatticeAtoms.keys()):
                arrGBAtom = self.__GetCoordinates(GBAtom)
                arrGrainPoint = self.FindClosestGrainPoint(arrGBAtom, strGrainKey)
                arrTriplePoints[n,j] = self.__LAMMPSTimeStep.PeriodicShiftCloser(arrGBAtom,arrGrainPoint)
            arrCentre = gf.EquidistantPoint(*arrTriplePoints[n])
            arrTripleLine[n] = arrCentre
            for m in range(len(arrTriplePoints[n])):
                for k in range(m+1,len(arrTriplePoints[n])):
                    fltLengths.append(gf.RealDistance(arrTriplePoints[n,m],arrTriplePoints[n,k]))   
            lstGrainBoundaryLength.append(min(fltLengths))
        fltMeanGrainBoundaryLength = np.mean(lstGrainBoundaryLength)
        lstIndicesToDelete = []
        for j in range(len(arrTripleLine)):
            lstSpacing = []
            for k in range(len(arrTriplePoints[j])):
                lstSpacing.append(gf.RealDistance(arrTriplePoints[j,k],arrTripleLine[j]))
            fltSpacing = max(lstSpacing)
            fltDistance = gf.RealDistance(arrTripleLine[j], self.__GetCoordinates(self.__GBAtoms[j]))
            if fltSpacing > fltMeanGrainBoundaryLength or fltDistance > fltMeanGrainBoundaryLength:
                lstIndicesToDelete.append(j)
        arrTripleLine = np.delete(arrTripleLine,lstIndicesToDelete, axis = 0 )
        arrTriplePoints = np.delete(arrTriplePoints,lstIndicesToDelete, axis = 0 )
        self.__TriplePoints = arrTriplePoints
        self.__TripleLine = arrTripleLine
    def __FindInitialPoint(self, inAtomCoordinates: np.array)->np.array:
        arrTriplePoints = np.zeros([self.__NumberOfGrains, self.__Dimensions])
        arrDistances = np.zeros([self.__Dimensions])
        for j,strGrainKey  in enumerate(self.__LatticeAtoms.keys()):
                arrGrainPoint = self.FindClosestGrainPoint(inAtomCoordinates, strGrainKey)
                arrTriplePoints[j] = self.__LAMMPSTimeStep.PeriodicShiftCloser(inAtomCoordinates,arrGrainPoint)
                arrDistances[j] = gf.RealDistance(inAtomCoordinates,arrTriplePoints[j])
                intMaxIndex = np.argmax(arrDistances)
        return arrTriplePoints[intMaxIndex],str(intMaxIndex) 
    def GetTriplePoints(self):
        return self.__TriplePoints
    def PeriodicMinimumDistance(self, inVector1: np.array, inVector2: np.array)->float:
        arrVector1Periodic = self.__LAMMPSTimeStep.PeriodicEquivalents(inVector1)
        arrDistances = np.zeros(len(arrVector1Periodic))
        for j, vctCurrent in enumerate(arrVector1Periodic):
            arrDistances[j] = gf.RealDistance(vctCurrent, inVector2)
        return np.min(arrDistances)

objData = Ld.LAMMPSData('data/30and40.eamPM')
objTimeStep = objData.GetTimeStepByIndex(0)

arrQuart1 = gf.GetQuaternion(np.array([0,0,1]),0)
arrQuart2 = gf.GetQuaternion(np.array([0,0,1]),30)
arrQuart3 = gf.GetQuaternion(np.array([0,0,1]),40)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
objPostProcess = OVITOSPostProcess(np.array([arrQuart1,arrQuart2,arrQuart3]), objTimeStep, 1)
objPostProcess.FindTriplePoints()
ax.scatter(*objPostProcess.PlotTripleLine(),c='red')
#ax.scatter(*objPostProcess.PlotGBAtoms(),c='red')
#ax.scatter(*objPostProcess.PlotTriplePoints(),c='black')
#ax.scatter(*objPostProcess.PlotUnknownAtoms(), c='blue')
#ax.scatter(*objPostProcess.PlotGBAtoms(),c='red')
#ax.scatter(*objPostProcess.PlotGrain('0'))
#ax.scatter(*objPostProcess.PlotGrain('1'))
#ax.scatter(*objPostProcess.PlotGrain('2'))
plt.show()
