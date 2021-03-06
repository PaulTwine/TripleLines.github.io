# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:38:14 2019

@author: twine
"""
import numpy as np
import itertools as it
import scipy as sc
from scipy import spatial
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import warnings
from decimal import Decimal
#import shapely as sp
#import geopandas as gpd
#All angles are assumed to be in radians
def DegreesToRadians(inDegrees: float)->float:
        return inDegrees/180*np.pi
def RealDistance(inPointOne, inPointTwo)->float:
        return np.linalg.norm(inPointOne-inPointTwo)
def RotateVector(vctInVector: np.array, vctAxis: np.array, inAngle: float)->np.array:
        vctAxis = NormaliseVector(vctAxis)
        return np.add(np.add(np.multiply(np.cos(inAngle), vctInVector),
                         np.multiply(np.sin(inAngle),np.cross(vctAxis,vctInVector))),
        np.multiply(1-np.cos(inAngle),np.multiply(np.dot(vctInVector,vctAxis),vctAxis)))
def NormaliseVector(vctInVector: np.array)-> np.array:
        if np.any(vctInVector != 0):
                rtnVector =  np.multiply(1/(np.linalg.norm(vctInVector)),vctInVector)
        else:  
                rtnVector = vctInVector
        return rtnVector
def Frange(inStart: float, inEnd: float)->list:
    lstNumbers = []
    i=0
    while (i+inStart) <= inEnd:
        lstNumbers.append(inStart + i)
        i += 1
    return lstNumbers 
def CylindrialToCartesian(inArray: np.array)->np.array:
    x = inArray[0]*np.cos(inArray[1])
    y = inArray[0]*np.sin(inArray[1])
    if len(inArray) == 2:
        return np.array([x,y])
    elif len(inArray) == 3:
        return np.array([x,y,inArray[2]])
def CartesianProduct(inList: list)->np.array:
    return np.array(list(it.product(*inList)))
def CreateCuboidLatticePoints(inBoxDimensions:np.array)->np.array:
        arrDimensions = np.zeros([len(inBoxDimensions),2]) 
        for j in range(len(arrDimensions)):
                arrDimensions[j][1] = inBoxDimensions[j]
        return CreateCuboidPoints(arrDimensions)
def CreateCuboidPoints(inBoxDimensions: np.array)->np.array:
    lstPoints = []
    for j in inBoxDimensions:
        lstPoints.append(list(Frange(j[0],j[1])))
    return CartesianProduct(lstPoints)
def FindBoundingBox(inVectors: np.array)->np.array:
        intDimensions = len(inVectors[0])
        vctDimensions = np.empty([intDimensions,2])
        for j in range(intDimensions):
                lstCoordinate = []
                fltCoordinate = 0
                for k in inVectors:
                        fltCoordinate += k[j]
                        lstCoordinate.append(fltCoordinate)
                vctDimensions[j] = [min(lstCoordinate), max(lstCoordinate)]
        return vctDimensions
def VectorToConstraint(inVector)->np.array:
        rtnVector = np.zeros([len(inVector),4])
        for j in range(len(inVector)):
                rtnVector[j,:3] = NormaliseVector(inVector[j])
                rtnVector[j,3] = np.linalg.norm(inVector[j])
        return rtnVector
def CheckLinearConstraint(inPoints: np.array, inConstraint: np.array)-> np.array:
        lstIndicesToDelete = []
        intDimensions = len(inConstraint)-1
        for j in range(len(inPoints)):
               if ((np.dot(inPoints[j],inConstraint[:-1]) > inConstraint[intDimensions])):
                   lstIndicesToDelete.append(j)
        return lstIndicesToDelete
def StandardBasisVectors(inDimensions: int): #generates standard basis vectors [1 0 0],[0 1 0] etc for any dimension
        arrBasisVectors = np.zeros([inDimensions, inDimensions])
        for j in range(inDimensions):
                arrBasisVectors[j][j] = 1
        return arrBasisVectors
def FindPlane(inVector1: np.array, inVector2: np.array, inPointOnPlane: np.array)->np.array:
        vctNormal = NormaliseVector(np.cross(inVector1, inVector2))
        fltConstant = np.dot(vctNormal,inPointOnPlane)
        return np.append(vctNormal, fltConstant)
def RotatedBasisVectors(inAngle: float, inAxis: np.array)->np.array:
        return RotateVectors(inAngle, inAxis, StandardBasisVectors(len(inAxis)))
def RotateVectors(inAngle: float, inAxis: np.array, inVectors: np.array)->np.array:
        arrReturnVector = np.zeros([len(inVectors), len(inVectors[0])])
        for j in range(len(inVectors)):
                arrReturnVector[j] = RotateVector(inVectors[j],inAxis,inAngle)
        return arrReturnVector
def OverlappedPoints(in2dArray1: np.array,in2dArray2: np.array)->list:
        lstOverlappedPoints = []
        for count, j in enumerate(in2dArray1):
                if any((in2dArray2[:]==j).all(1)):
                        lstOverlappedPoints.append(count)
        return lstOverlappedPoints
def GetQuaternionFromBasisMatrix(inBasis: np.array)-> np.array:
#         arrQuaternion = np.zeros(4)
#         eValues, eVectors = np.linalg.eig(np.transpose(inBasis))
#         intIndex = np.argwhere(np.isreal(eValues))[0,0]
#         vctAxis = NormaliseVector(np.real(eVectors[:,intIndex]))
#         fltAngle = np.arccos((np.trace(inBasis)-1)/2)
#         arrQuaternion[-1] = np.cos(fltAngle/2)
#         for j in range(3):
#                 arrQuaternion[j] = np.sin(fltAngle/2)*vctAxis[j]
#         return NormaliseVector(arrQuaternion) 
       r = 1/2*np.sqrt(1+np.trace(inBasis))
       return np.array([1/(4*r)*(inBasis[2,1]-inBasis[1,2]),1/(4*r)*(inBasis[0,2]-inBasis[2,0]),1/(4*r)*(inBasis[1,0]-inBasis[0,1]),r])      
def GetQuaternionFromVector(inVector: np.array, inAngle)->np.array: #angle first then axis
        vctAxis = NormaliseVector(inVector)
        #lstQuarternion  = []
        C = np.cos(inAngle/2)
        S = np.sin(inAngle/2)
        #lstQuarternion.append(vctAxis[0]*S)
        #lstQuarternion.append(vctAxis[1]*S)
        #lstQuarternion.append(vctAxis[2]*S)
        #lstQuarternion.append(C)
        return np.array([C,vctAxis[0]*S,vctAxis[1]*S,vctAxis[2]*S])
def QuaternionProduct(inVectorOne: np.array, inVectorTwo:np.array )->np.array: #angle first then axis
        if len(inVectorOne) != 4 or len(inVectorTwo) != 4:
                raise "Error quarternions must be 4 dimensional arrays"
        else:
                r1 = inVectorOne[0]
                r2 = inVectorTwo[0]
                v1 = inVectorOne[1:]
                v2 = inVectorTwo[1:]
                r = r1*r2 - np.dot(v1,v2)
                v  =  r1*v2 + r2*v1 + np.cross(v1,v2)
                return np.array([r,v[0],v[1],v[2]])
def QuaternionConjugate(inVector: np.array)->np.array: #takes a vector of quarternions
        rtnVector = -inVector
        rtnVector[0] = inVector[0]
        return rtnVector
def FCCQuaternionEquivalence(inVector: np.array)->np.array:
        arrVector = np.zeros([3,4])
        arrVector[0] = np.sort(np.abs(inVector))
        arrVector[1] = np.array([arrVector[0,0] - arrVector[0,1],arrVector[0,1]+ arrVector[0,0], arrVector[0,2]
        -arrVector[0,3],arrVector[0,3]+ arrVector[0,2]])*1/np.sqrt(2)
        arrVector[2] = np.array([arrVector[0,0]-arrVector[0,3]-arrVector[0,1] +arrVector[0,2],arrVector[0,1]
        -arrVector[0,3] - arrVector[0,2] + arrVector[0,0],arrVector[0,2]- arrVector[0,3] - arrVector[0,0] 
        + arrVector[0,1],arrVector[0,0]+arrVector[0,1]+arrVector[0,2]+arrVector[0,3]])*1/2
        intMax = np.argmax(arrVector[:,0])
        return arrVector[intMax]       
def EquidistantPoint(inVector1: np.array, inVector2: np.array, inVector3: np.array)->np.array: #3 dimensions only
        arrMatrix = np.zeros([3,3])
        arrMatrix[0] = inVector3-inVector2
        arrMatrix[1] = np.cross(inVector2-inVector1,inVector3-inVector1)
        arrMatrix[2] = inVector2-inVector1
        if np.linalg.det(arrMatrix) != 0:
                invMatrix = np.linalg.inv(arrMatrix)
                vctDirection = np.matmul(invMatrix, np.array([np.dot(arrMatrix[0],0.5*(inVector2+inVector3) - inVector1),0,0.5*np.dot(arrMatrix[2],arrMatrix[2])]))+inVector1
        else:
                vctDirection= np.mean(np.unique(np.array([inVector1, inVector2, inVector3]),axis=0), axis=0)
        return vctDirection    
def CheckLinearEquality(inPoints: np.array, inPlane: np.array, fltTolerance: float)-> np.array: #returns indices to delete for real coordinates  
        arrPositions = np.subtract(np.matmul(inPoints, np.transpose(inPlane[:,:-1])), np.transpose(inPlane[:,-1]))
        arrPositions = np.argwhere(np.abs(arrPositions) < fltTolerance)[:,0]        
        return arrPositions    
def MergePeriodicClusters(inPoints: np.array, inCellVectors: np.array, inBasisConversion: np.array, inBoundaryList: list, fltMinDistance = 0.5):
        lstPoints = []
        clustering = DBSCAN(fltMinDistance, min_samples=1).fit(inPoints)
        arrValues = clustering.labels_
        arrUniqueValues, arrCounts = np.unique(arrValues, return_counts=True)
        intMaxClusterValue = arrUniqueValues[np.argmax(arrCounts)] # find the largest cluster
        arrFixedPoint = np.mean(inPoints[arrValues ==intMaxClusterValue],axis=0) 
        lstPoints.append(inPoints[arrValues ==intMaxClusterValue])
        for j in arrValues:
                if j != intMaxClusterValue:
                        arrPointsToMove = inPoints[arrValues ==j]
                        arrPointsToMove = PeriodicShiftAllCloser(arrFixedPoint, arrPointsToMove, inCellVectors, inBasisConversion, inBoundaryList)
                        lstPoints.append(arrPointsToMove)
        return np.concatenate(lstPoints)
def IsVectorOutsideSimulationCell(inMatrix: np.array, invMatrix: np.array, inVector: np.array):
        arrCoefficients = np.matmul(inVector, invMatrix)
        if np.any(arrCoefficients >= 1) or np.any(arrCoefficients < 0):
                return True
        else:
                return False

def RemoveVectorsOutsideSimulationCell(inBasis: np.array, inVector: np.array, blnReturnCellCoordinates = False)->np.array:
        # if len(np.shape(inVector)) > 1:
        #         if np.shape(inVector)[1] == 2:
        #                 inMatrix = inMatrix[:2,:2]
        #                 invMatrix = invMatrix[:2,:2]
        # elif np.shape(inVector)[0]==2:
        #         inMatrix = inMatrix[:2,:2]
        #         invMatrix = invMatrix[:2,:2]  
        arrUnitBasis = inBasis/np.linalg.norm(inBasis, ord=2, axis=1, keepdims=True)
        invMatrix = np.linalg.inv(arrUnitBasis) 
        arrMod = np.linalg.norm(inBasis, axis=1)           
        arrCoefficients = np.matmul(inVector, invMatrix) #find the coordinates in the simulation cell basis
        lstRows = np.where(np.any(arrCoefficients >= arrMod, axis=0) | np.any(arrCoefficients <0 , axis=0))[0]
        return lstRows
        # arrCoefficients = arrCoefficients[lstRows]
        # # for j in range(3):
        # #         arrRows = np.where(np.abs(arrCoefficients[:,j]) < 0.1)
        # #         if len(arrRows) > 0:
        # #                 arrCoefficients[arrRows,j] = np.zeros(len(arrRows))
        # # arrCoefficients = np.mod(arrCoefficients, arrMod) #move so that they lie inside cell 
        # # #arrCoefficients = np.unique(arrCoefficients, axis= 0)
        # if blnReturnCellCoordinates:
        #         return np.matmul(arrCoefficients, arrUnitBasis),arrCoefficients #return the wrapped vector in the standard basis
        # else:        
        #         return np.matmul(arrCoefficients, arrUnitBasis)

def WrapVectorIntoSimulationCell(inBasis: np.array, invMatrix: np.array, inVector: np.array, blnReturnCellCoordinates = False)->np.array:
        # if len(np.shape(inVector)) > 1:
        #         if np.shape(inVector)[1] == 2:
        #                 inMatrix = inMatrix[:2,:2]
        #                 invMatrix = invMatrix[:2,:2]
        # elif np.shape(inVector)[0]==2:
        #         inMatrix = inMatrix[:2,:2]
        #         invMatrix = invMatrix[:2,:2]  
        arrUnitBasis = inBasis/np.linalg.norm(inBasis, ord=2, axis=1, keepdims=True)
        invMatrix = np.linalg.inv(arrUnitBasis) 
        arrMod = np.linalg.norm(inBasis, axis=1)           
        arrCoefficients = np.matmul(inVector, invMatrix) #find the coordinates in the simulation cell basis
        # for j in range(3):
        #         arrRows = np.where(np.abs(arrCoefficients[:,j]) < 0.1)
        #         if len(arrRows) > 0:
        #                 arrCoefficients[arrRows,j] = np.zeros(len(arrRows))
        arrCoefficients = np.mod(arrCoefficients, arrMod) #move so that they lie inside cell 
        #arrCoefficients = np.unique(arrCoefficients, axis= 0)
        if blnReturnCellCoordinates:
                return np.matmul(arrCoefficients, arrUnitBasis),arrCoefficients #return the wrapped vector in the standard basis
        else:        
                return np.matmul(arrCoefficients, arrUnitBasis)
def CheckVectorIsInSimulationCell(inMatrix: np.array, invMatrix: np.array, inVector: np.array)->np.array:
        arrCellCoordinates = np.matmul(inVector, invMatrix)
        arrModCoordinates = np.mod(arrCellCoordinates,np.ones(np.shape(arrCellCoordinates)))
        return np.all(arrCellCoordinates == arrModCoordinates)
def ExtendQuantisedVector(inVector: np.array, intAmount)->np.array: #extends 
        rtnVector = np.zeros([2])
        intMaxCol =np.argmax(np.abs(inVector))
        intMinCol = 1- intMaxCol
        fltRatio = inVector[intMinCol]/inVector[intMaxCol]
        rtnVector[intMaxCol] = inVector[intMaxCol]+ intAmount
        rtnVector[intMinCol] = rtnVector[intMaxCol]*fltRatio
        return QuantisedVector(rtnVector)
def QuantisedVector(inVector: np.array)->np.array: #2D only for now!
        intCol = np.argmax(np.abs(inVector))
        arrReturn = np.zeros([np.round(inVector[intCol]+1).astype('int'),2])
        fltRatio = 1
        if inVector[intCol] != 0:
                fltRatio = inVector[intCol-1]/inVector[intCol]
        for j in range(len(arrReturn)):
                arrReturn[j,intCol] = j
                arrReturn[j, 1-intCol] = np.round(j*fltRatio)   
        return arrReturn.astype('int')
def PeriodicEquivalents(inPositionVector: np.array, inCellVectors:np.array, inBasisConversion: np.array, inBoundaryList: list, blnInsideCell=False)->np.array: 
        arrCoefficients = np.matmul(inPositionVector, inBasisConversion) #find the coordinates in the simulation cell basis
        arrVector = np.copy(inPositionVector)
        lstOfArrays = []  
        for i,strBoundary in enumerate(inBoundaryList):
                if strBoundary == 'pp':
                        lstOfArrays.append(arrVector)
                        if blnInsideCell: #limits the search to points within +/- 0.5 of each periodic vector only use if you pass a single vector
                                if  arrCoefficients[i] > 0.5:
                                        lstOfArrays.append(arrVector - inCellVectors[i])
                                elif arrCoefficients[i] <= 0.5:
                                        lstOfArrays.append(arrVector+ inCellVectors[i])
                        else:
                                lstOfArrays.append(arrVector+inCellVectors[i])
                                lstOfArrays.append(arrVector-inCellVectors[i])
                        arrVector = np.vstack(lstOfArrays)
                        lstOfArrays = []
        return arrVector
def PeriodicShiftAllCloser(inFixedPoint: np.array, inAllPointsToShift: np.array, inCellVectors:np.array, inBasisConversion: np.array, inBoundaryList: list, blnNearyBy = False)->np.array:
        arrPoints = np.array(list(map(lambda x: PeriodicShiftCloser(inFixedPoint, x, inCellVectors, inBasisConversion, inBoundaryList, blnNearyBy), inAllPointsToShift)))
        return arrPoints
def PeriodicShiftCloser(inFixedPoint: np.array, inPointToShift: np.array, inCellVectors:np.array, inBasisConversion: np.array, inBoundaryList: list, blnNearyBy=False)->np.array:
        arrPeriodicVectors = PeriodicEquivalents(inPointToShift, inCellVectors, inBasisConversion, inBoundaryList, blnNearyBy)
        fltDistances = list(map(np.linalg.norm, np.subtract(arrPeriodicVectors, inFixedPoint)))
        return arrPeriodicVectors[np.argmin(fltDistances)]
def MakePeriodicDistanceMatrix(inVectors1: np.array, inVectors2: np.array, inCellVectors: np.array, inBasisConversion: np.array, inBoundaryList: list)->np.array:
        arrPeriodicDistance = np.zeros([len(inVectors1), len(inVectors2)])
        for j in range(len(inVectors1)):
            for k in range(len(inVectors2)):
                arrPeriodicDistance[j,k] = PeriodicMinimumDistance(inVectors1[j],inVectors2[k], inCellVectors, inBasisConversion, inBoundaryList)
        return arrPeriodicDistance
def AddPeriodicWrapper(inPoints: np.array,inCellVectors: np.array, fltDistance: float, blnRemoveOriginalPoints = False, lstPeriodic = ['p','p','p']):
        arrInverseMatrix = np.linalg.inv(inCellVectors)
        arrCoefficients = np.matmul(inPoints, arrInverseMatrix)
        arrProportions = np.zeros(3)
        for i in range(len(inCellVectors)):
                arrUnitVector = np.zeros(3)
                arrUnitVector[i] = 1
                fltComponent = np.dot(NormaliseVector(inCellVectors[i]),arrUnitVector)
                if fltComponent != 0:
                        arrProportions[i] = fltDistance/(np.linalg.norm(inCellVectors[i])*fltComponent)
        lstNewPoints = []
        if not blnRemoveOriginalPoints:
                lstNewPoints.append(arrCoefficients)
        for j in range(3):
                if lstPeriodic[j] == 'p':
                        arrVector = np.zeros(3)
                        arrVector[j] = 1
                        arrRows = np.where((arrCoefficients[:,j] >= 0) & (arrCoefficients[:,j] <= arrProportions[j]))[0]
                        arrNewPoints = arrCoefficients[arrRows] + arrVector
                        lstNewPoints.append(arrNewPoints)
                        arrRows = np.where((arrCoefficients[:,j] >= 1-arrProportions[j]) & (arrCoefficients[:,j] <= 1))[0]
                        arrNewPoints = arrCoefficients[arrRows] - arrVector
                        lstNewPoints.append(arrNewPoints)
                        arrCoefficients = np.concatenate(lstNewPoints)
        return np.matmul(arrCoefficients, inCellVectors)
def PeriodicMinimumDistance(inVector1: np.array, inVector2: np.array,inCellVectors: np.array, inBasisConversion: np.array, inBoundaryList: list)->float:
        inVector2 = PeriodicShiftCloser(inVector1, inVector2,inCellVectors,inBasisConversion,inBoundaryList)
        return np.linalg.norm(inVector2-inVector1, axis=0)
def PeriodicEquivalentMovement(inVector1, inVector2,inCellVectors, inBasisConversion, inBoundaryList, intAccuracy = 5):
        intDim = len(inVector1)
        arrTranslation = inVector2- inVector1
        arrCoefficients = np.matmul(arrTranslation,inBasisConversion)
        arrIntegers = np.round(arrCoefficients,0)
        arrDecimals = arrCoefficients - arrIntegers
        arrOriginalDecimals = np.copy(arrDecimals)
        lstOfArrays = []
        arrChange = np.zeros([intDim**3,intDim])
        for i, value in enumerate(inBoundaryList):
                lstOfArrays.append(arrDecimals)
                arrChange =np.zeros(intDim)
                arrChange[i] = 1
                if value =='pp':
                        lstOfArrays.append(arrDecimals - arrChange)
                        lstOfArrays.append(arrDecimals + arrChange)
                arrDecimals = np.vstack(lstOfArrays)
                lstOfArrays = []
        arrDistances = np.array(list(map(lambda x: np.sqrt(InnerProduct(x,x,inCellVectors)),arrDecimals)))
        intMin = np.argmin(arrDistances)
        arrIntegerMove = arrIntegers + (arrDecimals[intMin] - arrOriginalDecimals) #returns real distance, shortest vector, periodic part of movement
        return arrDistances[intMin], np.matmul(arrIntegers + arrDecimals[intMin], inCellVectors), np.matmul(arrIntegerMove, inCellVectors)
def PowerRule(r, a,b):
        return b*r**a
def LinearRule(r,m,c):
        return r*m+c 
def AsymptoticLinear(r,a,b):
        return a*r - b*np.log(r+b/a) + b*np.log(b/a)
def SortInDistanceOrder(inArray: np.array)->np.array:
        intLength = np.shape(inArray)[0]
        setIndices =set()
        if intLength > 2:
                myMatrix = np.round(sc.spatial.distance.cdist(inArray,inArray),4)
                arrSortedColumn = np.sort(myMatrix, axis =1)[:,2]
                intStart = np.argwhere(arrSortedColumn==np.max(arrSortedColumn)) #assumes an endpoint is furthest from 2nd neighbour
                intStart = intStart[0][0]
                j = 1
                lstIndices = [intStart]
                while (len(lstIndices) < intLength and j < intLength):
           # intStart = np.argwhere(myMatrix[intStart] == np.partition(myMatrix[intStart],j)[j])[0][0]
                        intStarts = np.argwhere(myMatrix[intStart] == np.sort(myMatrix[intStart])[j])
                        setIndices = set(set(intStarts[:,0])).difference(lstIndices)
                        if len(setIndices)>0:
                                intStart = setIndices.pop() 
                                lstIndices.append(intStart) #a new point so add to the list
                                j=1 #also now need to begin with the nearest neighbour
                        else: 
                                j +=1  
                return inArray[lstIndices], myMatrix[lstIndices,1]
        else:
                return inArray, np.array([0])
def ArcSegment(arrPoints: np.array, arrCentre: np.array, arrVector1: np.array, arrVector2: np.array, fltRadius: float, fltHeight: float)->list:
        #sector of a cylinder bounded by the two vectors and part of a cylindrical curved surface
        #for semicircular sections use arrVector1 == arrVector2
        arrMovedPoints = arrPoints[:,0:2] - arrCentre[0:2]
        arrUnit1 = NormaliseVector(arrVector1[0:2])
        arrUnit2 = NormaliseVector(arrVector2[0:2]) 
        lstIndices = np.where((np.dot(arrMovedPoints,arrUnit1) >= 0)  & (np.dot(arrMovedPoints,arrUnit2) >= 0)  & 
                                (np.linalg.norm(arrMovedPoints[:,0:2],axis=1)  <= fltRadius)
                                & (np.abs(arrPoints[:,2]) <= fltHeight/2))[0]
        return list(lstIndices)
def CylindricalVolume(arrPoints: np.array, arrCentre: np.array, fltRadius: float, fltHeight: float)->list:
        arrPointsNew = arrPoints - arrCentre
        lstIndices = np.where((np.linalg.norm(arrPointsNew[:,0:2],axis=1) <= fltRadius)
                               & (np.abs(arrPointsNew[:,2]) <= fltHeight/2))[0]
        return list(lstIndices)
def SphericalVolume(arrPoints: np.array, arrCentre: np.array, fltRadius: float)->list:
        arrPointsNew = arrPoints - arrCentre
        lstIndices = np.where(np.linalg.norm(arrPointsNew,axis=1) <= fltRadius)[0]
        return list(lstIndices)
def ParallelopipedVolume(arrPoints: np.array, arrStartPoint: np.array, arrAlong: np.array, arrAcross: np.array, arrUp:np.array)->list:
        arrPointsNew = arrPoints - arrStartPoint
        fltHeight = np.linalg.norm(arrUp, axis = 0)
        fltLength = np.linalg.norm(arrAlong, axis = 0)
        fltWidth = np.linalg.norm(arrAcross, axis = 0)
        lstIndices = np.where((np.abs(np.dot(arrPointsNew, NormaliseVector(arrUp))) <= fltHeight/2) &  (np.abs(np.dot(arrPointsNew, NormaliseVector(arrAcross))) <= fltWidth/2) & (np.dot(arrPointsNew, NormaliseVector(arrAlong)) <= fltLength)
        & (np.dot(arrPointsNew, NormaliseVector(arrAlong)) >= 0))[0]
        return list(lstIndices)
def AngleGenerator(intJobArray: int, fltIncrement: float, fltSymmetry: float): #updated to keep the angles in
    intN = int(fltSymmetry/fltIncrement -1) #ascending numerical order. intJobArray starts at 1 but Python is zero based 
    lstOfValues = []
    arrOfValues = np.zeros([intN*(intN-1),2])
    for i in range(intN):
        lstOfValues.append(fltIncrement*(i+1))
    counter = 0 
    for j in lstOfValues:
        for k in lstOfValues:
            if j!=k:
                arrOfValues[counter,0] = j
                arrOfValues[counter,1] = k
                counter +=1
    return arrOfValues[intJobArray-1, 0],arrOfValues[intJobArray-1, 1]
def IndexFromAngles(fltAngle1, fltAngle2, intLength, fltIncrement, fltLimit):
        lstOfAngles = []
        for j in range(1,intLength+1):
                lstOfAngles.append(AngleGenerator(j, fltIncrement, fltLimit))
        return lstOfAngles.index(tuple(fltAngle1, fltAngle2))
def FindNthSmallestPosition(inArray: np.array, intN :int)->list:
        fltValue = np.sort(inArray)[intN]
        lstPosition = np.where(inArray == fltValue)[0]
        return list(lstPosition)
def FindNthLargestPosition(inArray: np.array, intN: int)->list:
        fltValue = np.sort(inArray)[len(inArray)-intN-1]
        lstPosition = np.where(inArray == fltValue)[0]
        return list(lstPosition)
def FindRotationVectorAndAngle(arrStartVector: np.array, arrEndVector: np.array):
        arrUnitStart  = NormaliseVector(arrStartVector)
        arrUnitEnd  = NormaliseVector(arrEndVector)
        fltDot = np.dot(arrUnitStart,arrUnitEnd)
        if np.abs(fltDot) != 1:
                arrAxis = NormaliseVector(np.cross(arrUnitStart, arrUnitEnd))
                fltAngle = np.arccos(fltDot)
                return fltAngle, arrAxis
        else:
                warnings.warn("Two vectors are paralell")
def FindGeometricMediod(inPoints: np.array,bln2D = False, blnSquaring = True)-> np.array:
        if bln2D:
                inPoints = inPoints[:,0:2]
        arrDistanceMatrix = sc.spatial.distance_matrix(inPoints, inPoints)
        if blnSquaring:
                arrDistanceMatrix  = np.vectorize(lambda x: x**2)(arrDistanceMatrix)
        arrRowSums = np.sum(arrDistanceMatrix, axis = 0)
        intPosition = np.argmin(arrRowSums)
        return inPoints[intPosition]
def InnerProduct(inVector1 :np.array, inVector2: np.array, inBasisVectors: np.array )->float:
        arrMatrix = np.matmul(np.transpose(inBasisVectors), inBasisVectors)
        return np.matmul(np.transpose(inVector1), np.matmul(arrMatrix, inVector2))
def WrapAroundSlice(inSlices:np.array, inModArray:np.array)->slice:
        lstPoints = []
        for i in range(len(inSlices)):
                lstRange = list(range(inSlices[i][0],inSlices[i][1]))
                lstRange = list(np.mod(lstRange, inModArray[i]))  
                lstPoints.append(lstRange)
        lstPoints = CartesianProduct(lstPoints).astype('int')
        return lstPoints[:,0],lstPoints[:,1],lstPoints[:,2]
def EqualAxis3D(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
def ParseConic(lstCentre: list, lstScaling: list, lstPower:list, lstVariables = ['x','y','z'])->str:
        strReturn = ''
        for i in range(len(lstCentre)):
                strReturn += '((' + str(lstVariables[i]) + '-' + str(lstCentre[i]) + ')/' + str(lstScaling[i]) + ')**' + str(lstPower[i]) + '+'
        strReturn = strReturn[:-1] + '-1'
        return strReturn
def InvertRegion(strInput: str)->str: #changes an "inside closed surface to outside closed surface"
        return '-(' + strInput + ')'

def CubicCSLGenerator(inAxis: np.array, intIterations=5)->list: #usually five iterations is find the first 5 sigma values
        intGCD = np.gcd.reduce(inAxis)
        inAxis = inAxis*1/intGCD
        intSquared = np.sum(inAxis*inAxis).astype('int')
        dctSigma =dict()
        for j in range(2,intIterations+1):
                n = j
                m = 1
                while  (m < n):
                        intSigma = n**2 + m**2*intSquared
                        while np.mod(intSigma,2) == 0:
                                intSigma = intSigma/2
                        if intSigma > 2:
                                fltAngle = 2*np.arctan2(m*np.sqrt(intSquared),n)
                                if intSigma in dctSigma.keys():
                                        if abs(fltAngle) < abs(dctSigma[intSigma]):
                                                dctSigma[intSigma] = fltAngle
                                else:
                                        dctSigma[intSigma] = fltAngle  
                        m +=1                      
        arrReturn = np.ones([len(dctSigma.keys()),3])
        lstKeys = sorted(list(dctSigma.keys()))
        for k in range(len(lstKeys)):
                arrReturn[k, 0] = lstKeys[k]
                arrReturn[k,1] = dctSigma[lstKeys[k]]
                arrReturn[k,2] = 180*arrReturn[k,1]/np.pi
        return arrReturn
def GetBoundaryPoints(inPoints, intNumberOfNeighbours: int, fltRadius: float,inCellVectors = None):
        intLength = len(inPoints) #assumes a lattice configuration with fixed number of neighbours
        if intLength > 0:
                if inCellVectors is not(None):
                        inPoints = AddPeriodicWrapper(inPoints, inCellVectors, 4*fltRadius)
                objSpatial = KDTree(inPoints)
                arrCounts = objSpatial.query_radius(inPoints, fltRadius, count_only=True)
                arrBoundaryIndices = np.where(arrCounts < intNumberOfNeighbours +1)[0]
                arrBoundaryIndices = arrBoundaryIndices[arrBoundaryIndices < intLength]
                return arrBoundaryIndices
        else:
                return [] 
def GetPeriodicDuplicatePoints(inPoints, intNumberOfNeighbours: int, fltRadius: float,inCellVectors):
        intLength = len(inPoints) #assumes a lattice configuration with fixed number of neighbours
        if intLength > 0:
                arrWrappedPoints = AddPeriodicWrapper(inPoints, inCellVectors, 4*fltRadius)
                objSpatial = KDTree(arrWrappedPoints)
                arrCounts = objSpatial.query_radius(inPoints, fltRadius, count_only=True)
                arrBoundaryIndices = np.where(arrCounts > intNumberOfNeighbours+1)[0]
                arrBoundaryIndices = arrBoundaryIndices[arrBoundaryIndices < intLength]
                return arrBoundaryIndices
        else:
                return [] 
# def FindSurfaceArea(inPoints: np.array):
#         cloud = pv.PolyData(inPoints)
#         surf = cloud.delaunay_2d()
#         return surf.area
# def FindSplineLength(inPoints: np.array):
#         arrPoints = SortInDistanceOrder(inPoints)[0]
#         objSpline = pv.Spline(arrPoints)
#         return objSpline.length  
def DecimalArray(inArray: np.array):
        if len(np.shape(inArray)) ==2:
                lstDecimals = [[Decimal(str(x)) for x in y] for y in inArray]
        elif len(np.shape(inArray)) ==1:
                lstDecimals = [Decimal(str(x)) for x in inArray]
        return np.array(lstDecimals)

def EqualRows(arrOne: np.array, arrTwo: np.array, intRound = 5):
        blnEqual = True
        i=0
        arrRoundOne = np.round(arrOne,intRound)
        arrRoundTwo = np.round(arrTwo, intRound)
        while blnEqual and i < len(arrOne):
                if not(np.any(np.all(arrRoundOne[i]==arrRoundTwo, axis=1),axis=0)):
                        blnEqual = False
                        i += 1
        return blnEqual
def CubicQuaternions():
        lstRows=[0,1,-1]
        lstQuaternions = []
        for i in range(3):
                for j in range(3):
                        for k in range(3):
                                arrDirection = np.array([lstRows[k],lstRows[j],lstRows[i]])
                                fltLength = np.round(np.linalg.norm(arrDirection),5)
                                if  fltLength == 1:
                                        for a in range(1,4):
                                                lstQuaternions.append(GetQuaternionFromVector(arrDirection,np.pi/4*a))
                                elif fltLength == np.round(np.sqrt(2),5):
                                        lstQuaternions.append(GetQuaternionFromVector(arrDirection,np.pi))
                                elif fltLength ==np.round(np.sqrt(3),5):
                                        for b in range(1,3):
                                                lstQuaternions.append(GetQuaternionFromVector(arrDirection,2*np.pi/3*a))
        arrValues = np.vstack(lstQuaternions)
        arrRows = np.unique(np.round(arrValues,3),axis=0, return_index=True)[1]                              
        return arrValues[arrRows]

                               

       