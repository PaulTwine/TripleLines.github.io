# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:38:14 2019

@author: twine
"""

import numpy as np
import itertools as it
from numpy.linalg.linalg import det
import scipy as sc
from scipy import spatial
import sympy as sy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import warnings
from decimal import Decimal
import scipy.stats as stats
import LatticeDefinitions as ld
from scipy import optimize
import MiscFunctions as mf
import cmath as cm
from fractions import Fraction
from functools import reduce

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
        if len(np.shape(inVector)) ==1:
                inVector = np.array([inVector])
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
       return np.array([r,1/(4*r)*(inBasis[2,1]-inBasis[1,2]),1/(4*r)*(inBasis[0,2]-inBasis[2,0]),1/(4*r)*(inBasis[1,0]-inBasis[0,1])])      
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
def MergePeriodicClusters(inPoints: np.array, inCellVectors: np.array, inBoundaryList: list, fltMin = 4.05):
        inConstraints = FindConstraintsFromBasisVectors(inCellVectors)
        lstPoints = []
        clustering = DBSCAN(fltMin).fit(inPoints)
        arrValues = clustering.labels_
        arrUniqueValues, arrCounts = np.unique(arrValues, return_counts=True)
        arrUniqueValues = arrUniqueValues[np.argsort(arrCounts)[::-1]]
        lstTranslations = []
        lstPosition = []
        lstUniqueValues = list(arrUniqueValues)
        lstMergedPoints = []
        if -1 in lstUniqueValues:
                lstUniqueValues.remove(-1)
        while  len(lstUniqueValues) > 0:
                lstEquivalentPoints = []
                arrPoints = inPoints[arrValues == lstUniqueValues[0]]
                lstUniqueValues.remove(lstUniqueValues[0])
                lstEquivalentPoints.append(arrPoints)
                objTree = PeriodicWrapperKDTree(arrPoints, inCellVectors, inConstraints, fltMin,inBoundaryList)
                k = 0
                while k < len(lstUniqueValues):
                        arrNextPoints = inPoints[arrValues == lstUniqueValues[k]]
                        arrDistances, arrIndices = objTree.Pquery(arrNextPoints, 1)
                        arrDistances = np.array([x[0] for x in arrDistances])
                        arrIndices = np.array([x[0] for x in arrIndices])
                        arrRows = np.where(arrDistances <= fltMin)[0]
                        if len(arrRows) > 0:
                                arrCloseIndices = arrIndices[arrRows]
                                arrTranslation = arrPoints[objTree.GetPeriodicIndices(arrCloseIndices)]-objTree.GetExtendedPoints()[arrCloseIndices]
                                intMin = np.argmin(np.linalg.norm(arrTranslation,axis=1))
                                lstEquivalentPoints.append(arrNextPoints + arrTranslation[intMin])
                                lstUniqueValues.remove(lstUniqueValues[k])
                        else:
                                k +=1
                if len(lstEquivalentPoints) == 1:
                        lstMergedPoints.append(lstEquivalentPoints[0])
                else:
                        lstMergedPoints.append(np.concatenate(lstEquivalentPoints, axis= 0))
        return lstMergedPoints        
        
        
   
def IsVectorOutsideSimulationCell(inMatrix: np.array, invMatrix: np.array, inVector: np.array):
        arrCoefficients = np.matmul(inVector, invMatrix)
        if np.any(arrCoefficients >= 1) or np.any(arrCoefficients < 0):
                return True
        else:
                return False
def FindConstraintsFromBasisVectors(inBasisVectors: np.array): #must be in order of a right handed set
        i = len(inBasisVectors)
        arrConstraints = np.zeros([i,i+1])
        for j in range(i):
            arrConstraints[j,:i] = NormaliseVector(np.cross(inBasisVectors[np.mod(j+1,i)], inBasisVectors[np.mod(j+2,i)]))
            arrConstraints[j,i] = np.dot(arrConstraints[j,:i],inBasisVectors[np.mod(j,i)])
        return arrConstraints
def RemoveVectorsOutsideSimulationCell(inBasis: np.array, inVectors: np.array,blnIncludeAllBoundaries = False)->np.array:      
        arrUnitBasis = inBasis/np.linalg.norm(inBasis, ord=2, axis=1, keepdims=True)
        invMatrix = np.linalg.inv(arrUnitBasis) 
        arrMod = np.linalg.norm(inBasis, axis=1)           
        arrCoefficients = np.round(np.matmul(inVectors, invMatrix),10) #find the coordinates in the simulation cell basis
        if blnIncludeAllBoundaries:
                arrRows = np.where(np.all(arrCoefficients <= arrMod, axis=1) & np.all(arrCoefficients >= 0 , axis=1))[0]
        else:
                arrRows = np.where(np.all(arrCoefficients < arrMod, axis=1) & np.all(arrCoefficients >= 0 , axis=1))[0]
        return arrRows   
def WrapVectorIntoSimulationCell(inCellVectors: np.array, inVector: np.array, fltTolerance = 1e-5)->np.array: 
        arrUnitBasis = inCellVectors/np.linalg.norm(inCellVectors, ord=2, axis=1, keepdims=True)
        invMatrix = np.linalg.inv(arrUnitBasis) 
        arrMod = np.linalg.norm(inCellVectors, axis=1)           
        arrCoefficients = np.matmul(inVector, invMatrix) #find the coordinates in the simulation cell basis
        arrCoefficients = np.mod(arrCoefficients, arrMod) #move so that they lie inside cell 
        arrRowsAndCols = np.where((arrCoefficients > (arrMod -fltTolerance*np.ones(3))) & (arrCoefficients < (arrMod + fltTolerance*np.ones(3))))
        if len(arrRowsAndCols) > 0:
                arrCoefficients[arrRowsAndCols] = 0      
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
def InnerProducts(inVectors: np.array, inBasis: np.array)->np.array:
        vFunction = np.vectorize(GeneralLength, signature='(n),(m,p)->()')
        return vFunction(inVectors, inBasis)
def GeneralLength(inVector, inBasis)->np.array:
        return InnerProduct(inVector,inVector,inBasis)
def PeriodicAllMinDisplacement(arrDisplacements, inCellVectors, inPeriodicDirections):
        arrPoints = np.array(list(map(lambda x: PeriodicMinDisplacement(x, inCellVectors,inPeriodicDirections),arrDisplacements)))
        return arrPoints
def PeriodicMinDisplacement(arrDisplacements, inPeriodicVectors: np.array, inPeriodicDirections):
       intLength = len(arrDisplacements)
       arrWrapped = WrapVectorIntoSimulationCell(inPeriodicVectors,arrDisplacements)
       arrRows = np.array(range(intLength))
       arrPoints = PeriodicExtension(arrWrapped, inPeriodicVectors)
       arrDistances = np.linalg.norm(arrPoints, axis=1)
       arrStackedDistances = np.reshape(arrDistances, (2**(len(inPeriodicVectors)),intLength))
       arrMinColumns = np.argmin(arrStackedDistances, axis =0)
       arrMinPoints = intLength*arrMinColumns + arrRows
       return arrPoints[arrMinPoints], arrDistances[arrMinPoints]
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
        intSize = np.shape(inPoints)[1]
        arrProportions = np.zeros(intSize)
        for i in range(len(inCellVectors)):
                arrUnitVector = np.zeros(intSize)
                arrUnitVector[i] = 1
                fltComponent = np.dot(NormaliseVector(inCellVectors[i]),arrUnitVector)
                if fltComponent != 0:
                        arrProportions[i] = fltDistance/(np.linalg.norm(inCellVectors[i])*fltComponent)
        lstNewPoints = []
        if not blnRemoveOriginalPoints:
                lstNewPoints.append(arrCoefficients)
        for j in range(intSize):
                if lstPeriodic[j] == 'p':
                        arrVector = np.zeros(intSize)
                        arrVector[j] = 1
                        arrRows = np.where((arrCoefficients[:,j] >= 0) & (arrCoefficients[:,j] <= arrProportions[j]))[0]
                        arrNewPoints = arrCoefficients[arrRows] + arrVector
                        lstNewPoints.append(arrNewPoints)
                        arrRows = np.where((arrCoefficients[:,j] >= 1-arrProportions[j]) & (arrCoefficients[:,j] <= 1))[0]
                        arrNewPoints = arrCoefficients[arrRows] - arrVector
                        lstNewPoints.append(arrNewPoints)
                        arrCoefficients = np.concatenate(lstNewPoints)
        return np.matmul(arrCoefficients, inCellVectors)
def PeriodicExtension(inPoints: np.array, inCellVectors: np.array):
        lstPoints = []
        arrPoints = np.copy(inPoints)
        for i in inCellVectors:
                lstPoints.append(arrPoints)
                lstPoints.append(arrPoints-i)
                arrPoints = np.vstack(lstPoints)
                lstPoints = [] 
        return arrPoints

def AddPeriodicWrapperAndIndices(inPoints: np.array,inCellVectors,inConstraints: np.array, fltDistance: float, lstPeriodic = ['p','p','p']):
        lstNewIndices = []
        lstNewPoints = []
        intLength = len(inPoints)
        lstNewPoints.append(inPoints)
        arrAllIndices = np.array(list(range(len(inPoints))))
        arrAllPoints = np.copy(inPoints)
        lstNewIndices.append(arrAllIndices)
        for i in range(3):
                if lstPeriodic[i] == 'p':
                        blnAdd = False
                        j = inConstraints[i]
                        k = inCellVectors[i]
                        arrPositions1 =  np.round(np.subtract(np.matmul(arrAllPoints, np.transpose(j[:-1])), fltDistance),5)
                        arrRows1 = np.where(arrPositions1 < 1e-5)[0]
                        if len(arrRows1) > 0:
                                blnAdd = True
                                lstNewIndices.append(arrAllIndices[arrRows1])
                                lstNewPoints.append(arrAllPoints[arrRows1] + k)
                        arrPositions2 =  np.round(np.subtract(np.matmul(arrAllPoints, np.transpose(j[:-1])), j[-1]-fltDistance),5)
                        arrRows2 = np.where(arrPositions2 > -1e-5)[0]
                        if len(arrRows2) > 0:
                                blnAdd = True
                                lstNewIndices.append(arrAllIndices[arrRows2])
                                lstNewPoints.append(arrAllPoints[arrRows2] - k)
                        #if len(lstNewPoints) > 0:
                        if blnAdd:
                                arrAllPoints = np.concatenate(lstNewPoints)
                                arrAllIndices = np.concatenate(lstNewIndices)                
        if len(arrAllPoints) > intLength:
                objSpatial = KDTree(arrAllPoints[intLength:])
                arrDuplicates = objSpatial.query_radius(inPoints,1e-5,count_only=False, return_distance = False)
                lstDuplicates = list(map(lambda x: x, arrDuplicates))
                lstDuplicates = [item for sublist in lstDuplicates for item in sublist]
                arrRows = np.unique(lstDuplicates)
                if len(arrRows) > 0:
                        arrDelete = arrRows + np.ones(len(arrRows))*intLength
                        arrDelete = arrDelete.astype('int')
                        arrAllPoints = np.delete(arrAllPoints, arrDelete, axis=0)
                        arrAllIndices = np.delete(arrAllIndices, arrDelete, axis=0)
        return arrAllPoints, arrAllIndices
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
def FindReflectionMatrix(inPlaneNormal: np.array):
        fltDot = np.dot(inPlaneNormal,inPlaneNormal)
        arrNormal = np.array([inPlaneNormal])
        return np.identity(len(inPlaneNormal)) - 2*np.matmul(np.transpose(arrNormal), arrNormal)/fltDot
def FindMediod(inPoints: np.array):
        arrMean = np.mean(inPoints, axis=0)
        if len(np.shape(inPoints)) > 1:
                arrDistances = np.linalg.norm(inPoints-arrMean, axis=1)
        elif len(np.shape(inPoints)) == 1:
                arrDistances = np.linalg.norm(inPoints-arrMean)
        return inPoints[np.argmin(arrDistances)]
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

def GeneralCylinder(arrAxis: np.array, arrBasePoint: np.array, fltRadius: float, lstVariables = ['x','y','z']):
        arrUnitAxis = NormaliseVector(arrAxis)
        strReturn = '(a2*z-a3*y)**2 + (a3*x-a1*z)**2 +(a1*y-a2*x)**2 - ' + str(fltRadius**2)
        strReturn = strReturn.replace('x', '(' + lstVariables[0] + '-' + str(arrBasePoint[0]) + ')')
        strReturn = strReturn.replace('y', '(' + lstVariables[1] + '-' + str(arrBasePoint[1]) + ')')
        strReturn = strReturn.replace('z', '(' + lstVariables[2] + '-' + str(arrBasePoint[2]) + ')')
        strReturn = strReturn.replace('a2', str(arrUnitAxis[1]))
        strReturn = strReturn.replace('a2', str(arrUnitAxis[1]))
        strReturn = strReturn.replace('a3', str(arrUnitAxis[2]))
        return strReturn
def ParsePlane(arrNormal: np.array, arrPointOnPlane: np.array, lstVariables = ['x','y','z']):
        arrUnit = NormaliseVector(arrNormal)
        strReturn = 'x*n1 + y*n2 + z*n3'
        strReturn = strReturn.replace('x', '(' + lstVariables[0] + '-' + str(arrPointOnPlane[0]) + ')')
        strReturn = strReturn.replace('y', '(' + lstVariables[1] + '-' + str(arrPointOnPlane[1]) + ')')
        strReturn = strReturn.replace('z', '(' + lstVariables[2] + '-' + str(arrPointOnPlane[2]) + ')')
        strReturn = strReturn.replace('n1', str(arrUnit[0]))
        strReturn = strReturn.replace('n2', str(arrUnit[1]))
        strReturn = strReturn.replace('n3', str(arrUnit[2]))
        return strReturn        
        
def ParseConic(lstCentre: list, lstScaling: list, lstPower:list, lstVariables = ['x','y','z'])->str:
        strReturn = ''
        for i in range(len(lstCentre)):
                strReturn += '((' + str(lstVariables[i]) + '-' + str(lstCentre[i]) + ')/' + str(lstScaling[i]) + ')**' + str(lstPower[i]) + '+'
        strReturn = strReturn[:-1] + '-1'
        return strReturn
def InvertRegion(strInput: str)->str: #changes an "inside closed surface to outside closed surface"
        return '-(' + strInput + ')'
def GetMatrixFromAxisAngle(inAxis, inAngle):
        inAxis = NormaliseVector(inAxis)
        c = np.cos(inAngle)
        s = np.sin(inAngle)
        x = inAxis[0]
        y = inAxis[1]
        z = inAxis[2]
        arrMatrix = np.array([[c+x**2*(1-c), x*y*(1-c) - z*s,x*z*(1-c) +y*s],
                                [y*x*(1-c) + z*s,c+y**2*(1-c), y*z*(1-c) -x*s],
                                [ x*z*(1-c)-y*s,y*z*(1-c)+x*s,c+z**2*(1-c)]])
        return arrMatrix

def CubicCSLGenerator(inAxis: np.array, intIterations=5, blnDisorientation = False)->list: #usually five iterations is find the first 5 sigma values
        intGCD = np.gcd.reduce(inAxis)
        inAxis = inAxis*1/intGCD
        intSquared = np.sum(inAxis*inAxis).astype('int')
        dctSigma =dict()
        intLimit = int(intIterations + 1)
        for a in range(0,intLimit):
                n = a
                for b in range(0,intLimit):
                        m = b
                        i = np.max([np.gcd(n,m),1]) 
                        intSigma = (n**2 + m**2*intSquared)/(i**2)
                        if intSigma == 0:
                                intSigma = 1
                        while np.mod(intSigma,2) == 0:
                                intSigma = intSigma/2
                        if intSigma > 2:       
                                fltAngle = 2*np.arctan2(m*np.sqrt(intSquared),n)
                                if blnDisorientation:
                                        if intSigma in dctSigma.keys():
                                                if abs(fltAngle) < abs(dctSigma[intSigma]):
                                                        dctSigma[(m/i,n/i)] = fltAngle
                                        else:
                                                dctSigma[(m/i,n/i)] = (intSigma,fltAngle)
                                else:
                                        dctSigma[(m/i,n/i)] = (intSigma,fltAngle)  
                        m +=1                      
        arrReturn = np.ones([len(dctSigma.keys()),3])
        p = 0
        for k in dctSigma.keys():
                arrReturn[p,0] = dctSigma[k][0]
                arrReturn[p,1] = dctSigma[k][1]
                arrReturn[p,2] = 180*arrReturn[p,1]/np.pi
                p +=1
        return arrReturn[np.argsort(arrReturn[:,0])]
def FindAxesFromSigmaValues(intSigma :int, intLimit: int): #cubic only
        lstAxes = []
        lstPossibleAxes = list(it.combinations_with_replacement(list(range(intSigma)),3))
        lstPossibleAxes.remove((0,0,0))
        lstReduce = list(map(lambda x: x/np.gcd.reduce(x),lstPossibleAxes))
        lstReduce = np.unique(lstReduce,axis=0).astype('int')
        for a in lstReduce:
                blnNotFound = True
                n = 0
                m = 0
                intSquared = np.sum(a*a).astype('int')
                while m <= intLimit and blnNotFound:
                        n = 0
                        while n <= intLimit and blnNotFound:
                                intTest = (n**2 + m**2*(intSquared))
                                intTest = np.max([intTest,1])
                                while np.mod(intTest,2) == 0:
                                        intTest = intTest/2
                                if int(intTest) == int(intSigma):
                                        blnNotFound =False
                                        lstAxes.append(a)
                                n +=1
                        m +=1
        return np.unique(lstAxes,axis=0)
def GetBoundaryPoints(inPoints, intNumberOfNeighbours: int, fltRadius: float, inCellVectors = None):
        intLength = len(inPoints) #assumes a lattice configuration with fixed number of neighbours
        inConstraints = FindConstraintsFromBasisVectors(inCellVectors)
        if intLength > 0:
                if inCellVectors is not(None):
                        arrWrappedPoints,arrIndices = AddPeriodicWrapperAndIndices(inPoints, inCellVectors,inConstraints, 2*fltRadius)
                else:
                        arrWrappedPoints = inPoints
                        arrIndices = np.array(list(range(len(inPoints))))
                objSpatial = KDTree(arrWrappedPoints)
                #arrIndices = objSpatial.query_radius(inPoints,1e-5,count_only=False, return_distance = False)
                arrCounts = objSpatial.query_radius(inPoints, fltRadius, count_only=True)
                arrBoundaryIndices = np.where(arrCounts < intNumberOfNeighbours+1)[0]
                return np.unique(arrIndices[arrBoundaryIndices])
        else:
                return [] 
def GetPeriodicDuplicatePoints(inPoints, intNumberOfNeighbours: int, fltRadius: float,inCellVectors):
        intLength = len(inPoints) #assumes a lattice configuration with fixed number of neighbours
        if intLength > 0:
                arrWrappedPoints = AddPeriodicWrapper(inPoints, inCellVectors, 4*fltRadius)
                objSpatial = KDTree(arrWrappedPoints)
                arrCounts = objSpatial.query_radius(inPoints, fltRadius, count_only=True)
                arrBoundaryIndices = np.where(arrCounts < intNumberOfNeighbours+1)[0]
                arrBoundaryIndices = arrBoundaryIndices[arrBoundaryIndices < intLength]
                return arrBoundaryIndices
        else:
                return [] 
def NormaliseMatrixAlongRows(inArray: np.array):
        return inArray/np.linalg.norm(inArray, axis=1)[:,np.newaxis]
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
                                        for a in range(0,4):
                                                lstQuaternions.append(GetQuaternionFromVector(arrDirection,np.pi/4*a))
                                elif fltLength == np.round(np.sqrt(2),5):
                                        lstQuaternions.append(GetQuaternionFromVector(arrDirection,np.pi))
                                elif fltLength ==np.round(np.sqrt(3),5):
                                        for b in range(0,3):
                                                lstQuaternions.append(GetQuaternionFromVector(arrDirection,2*np.pi/3*b))
        arrValues = np.vstack(lstQuaternions)
        arrRows = np.unique(np.round(arrValues,3),axis=0, return_index=True)[1]                              
        return arrValues[arrRows]

def MergeTooCloseAtoms(inPoints, inBasisVectors, fltDistance, intLimit =50):
        if fltDistance == 0:
                fltDistance = 1e-5
        blnStop = False
        i = 0
        arrPoints = np.copy(inPoints)
        inConstraints = FindConstraintsFromBasisVectors(inBasisVectors)
        while not(blnStop) and i < intLimit:
                lstMergedAtoms = []
                objGBTree = PeriodicWrapperKDTree(arrPoints,inBasisVectors,inConstraints,1.05*fltDistance/2)
                arrExtendedGBAtoms = objGBTree.GetExtendedPoints()
                arrIndices,arrDistances = objGBTree.Pquery_radius(arrPoints,fltDistance) #by default points are returned in distance order
                lstDistances = list(map(lambda x: np.round(x,5),arrDistances))
                arrLengths = np.array(list(map(lambda x: len(x),arrIndices)))
                arrRows = np.where(arrLengths > 1)[0]
                if len(arrRows) > 0:
                        lstIndices = list(map(lambda x: arrIndices[x][lstDistances[x] <= lstDistances[x][1]],arrRows)) #every point 0 distance from itself 
                        #point at position [1] is then the next closest point. 
                        lstUsedIndices = [item for sublist in lstIndices for item in sublist]
                        lstTrueIndices = np.unique(objGBTree.GetPeriodicIndices(lstUsedIndices)).tolist()
                        arrUnusedIndices = np.unique(list(set(range(len(arrPoints))).difference(lstTrueIndices)))
                        lstMergedAtoms = list(map(lambda x: np.mean(arrExtendedGBAtoms[x],axis=0),lstIndices))
                        if len(arrUnusedIndices) > 0:
                                lstMergedAtoms.append(arrPoints[arrUnusedIndices])
                        arrPoints = np.vstack(lstMergedAtoms)
                        arrRows = FindDuplicates(arrPoints, inBasisVectors,fltDistance) 
                        lstUniqueIndices = list(set(range(len(arrPoints))).difference(arrRows.tolist()))
                        arrUniqueIndices = np.unique(lstUniqueIndices)
                        arrPoints = arrPoints[arrUniqueIndices]

                else:
                        blnStop = True
                i +=1
        if i == intLimit:
                warnings.warn('Merge too close atoms terminated after ' + str(i) + ' iterations')
        
        return arrPoints
def FindReciprocalVectors(inRealVectors: np.array): 
        # V = np.linalg.det(inRealVectors)
        # #rtnMatrix= np.matmul(np.transpose(inRealVectors),np.linalg.inv(np.matmul(inRealVectors,np.transpose(inRealVectors))))
        # #return rtnMatrix
        # lstVectors = []
        # for k in range(len(inRealVectors)):
        #         intFirst = np.mod(k+1,3)
        #         intSecond = np.mod(k+2,3)
        #         lstVectors.append(np.cross(inRealVectors[intFirst],inRealVectors[intSecond])/V)
        return np.linalg.inv(np.transpose(inRealVectors))

def FindDuplicates(inPoints, inCellVectors, fltDistance, lstBoundaryType = ['p','p','p']):
        arrConstraints = FindConstraintsFromBasisVectors(inCellVectors)
        objPeriodicTree = PeriodicWrapperKDTree(inPoints, inCellVectors,arrConstraints,2*fltDistance, lstBoundaryType)
        arrIndices = objPeriodicTree.Pquery_radius(inPoints,fltDistance)[0]
        lstIndices = objPeriodicTree.GetPeriodicIndices(arrIndices)
        arrLengths = np.array(list(map(lambda x: len(x),lstIndices)))
        arrRows = np.where(arrLengths >1)[0]
        arrRepeatedIndices = np.array([])
        if len(arrRows) > 0:
                lstDuplicates = list(map(lambda x: np.sort(lstIndices[x])[1:], arrRows))
                lstDuplicates = [item for sublist in lstDuplicates for item in sublist]
                arrRepeatedIndices = np.unique(lstDuplicates)
               # arrRepeatedIndices = np.unique(np.vstack(list(map(lambda x: np.sort(lstIndices[x])[1:],arrRows))))
        return arrRepeatedIndices
def AffineTransformationMatrix(inMatrix: np.array, inTranslation: np.array):
        tupShape = np.shape(inMatrix)
        rtnMatrix = np.zeros([tupShape[0]+1,tupShape[1]+1])
        rtnMatrix[:-1,-1] = inTranslation
        rtnMatrix[:tupShape[0],:tupShape[1]] = inMatrix
        rtnMatrix[-1,-1] = 1
        return rtnMatrix 

class PeriodicKDTree(object):
    def __init__(self, inPoints, inPeriodicVectors, lstBoundaryType = ['pp','pp','pp']):
        self.__PeriodicVectors = inPeriodicVectors
        arrInverse = np.linalg.inv(inPeriodicVectors)
        self.__InverseBasis = arrInverse
        self.__ModValue = len(inPoints)
        arrScaling = np.linalg.inv(np.diag(np.linalg.norm(inPeriodicVectors, axis=1)))  
        self__UnitVectors = np.matmul(arrScaling, inPeriodicVectors)
        self.__UnitPeriodicVectors = np.matmul(arrScaling,inPeriodicVectors)
        self.__ExtendedPoints = PeriodicEquivalents(inPoints, inPeriodicVectors,arrInverse,lstBoundaryType)
        self.__PeriodicTree = KDTree(self.__ExtendedPoints)
    def Pquery_radius(self, inPoints: np.array, fltRadius: float):
        arrIndices = self.__PeriodicTree.query_radius(inPoints, fltRadius)
        lstIndices = list(map(lambda x: np.unique(np.mod(x,self.__ModValue)),arrIndices))
        x = 1
      #  for j in arrIndices:
      #      lstIndices.append(np.unique(np.mod(j, self.__ModValue)))
        return lstIndices
    def Pquery(self,inPoints: np.array, k :int, fltDistance: float):
        arrDistances, arrIndices = self.__PeriodicTree.query(inPoints, k)
        arrDistances = np.round(arrDistances, 5)
        arrClose =  np.where(arrDistances <= fltDistance)
        arrDistances = arrDistances[arrClose]
        arrIndices = arrIndices[arrClose]
        arrIndices = np.mod(arrIndices, self.__ModValue)
        arrPositions = np.unique(arrIndices,axis=0, return_index=True)[1]
        arrDistances = arrDistances[np.argsort(arrPositions)]
        arrIndices = arrIndices[np.argsort(arrPositions)] 
        return arrDistances,arrIndices

class PeriodicWrapperKDTree(object):
    def __init__(self, inPoints,inPeriodicVectors,inConstraints, fltWrapperLength, lstBoundaryType = ['p','p','p']):
        self.__OriginalPoints = np.copy(inPoints)
        self.__PeriodicVectors = np.copy(inPeriodicVectors)
        arrInverse = np.linalg.inv(inPeriodicVectors)
        self.__InverseBasis = arrInverse
        self.__ModValue = len(inPoints)
        self.__WrapperWidth = fltWrapperLength
        arrScaling = np.linalg.inv(np.diag(np.linalg.norm(inPeriodicVectors, axis=1)))  
        self.__UnitPeriodicVectors = np.matmul(arrScaling,inPeriodicVectors)
        arrExtendedPoints, arrUniqueIndices = AddPeriodicWrapperAndIndices(inPoints, inPeriodicVectors,inConstraints,fltWrapperLength,lstBoundaryType)
        if len(arrUniqueIndices) == 0:
                self.__UniqueIndices = np.array(list(range(len(inPoints))))
        else:        
                self.__UniqueIndices = arrUniqueIndices
        
        if len(arrExtendedPoints) == 0:
                self.__ExtendedPoints = np.copy(inPoints)
        else:
                self.__ExtendedPoints = arrExtendedPoints
        if len(self.__ExtendedPoints) > 0:
                self.__PeriodicTree = KDTree(self.__ExtendedPoints)
    def Pquery_radius(self, inPoints: np.array, fltRadius: float,blnReturnDistance=True, blnSortResults=True):
        arrIndices,arrDistances = self.__PeriodicTree.query_radius(inPoints, fltRadius,return_distance=blnReturnDistance,sort_results=blnSortResults)
        return arrIndices, arrDistances
    def GetExtendedPoints(self):
        return self.__ExtendedPoints
    def GetPeriodicIndices(self, inRealIndices: list)->list:
        return list(map(lambda x: self.__UniqueIndices[x],inRealIndices))
    def Pquery(self,inPoints:np.array,k=1):
        arrDistances, arrIndices = self.__PeriodicTree.query(inPoints,k)
        return arrDistances, arrIndices
    def GetWrapperLength(self):
        return self.__WrapperWidth
    def GetOriginalPoints(self):
        return self.__OriginalPoints

class PeriodicFullKDTree(object):
    def __init__(self, inPoints: np.array,inPeriodicVectors: np.array):
        self.__OriginalPoints = np.copy(inPoints)
        self.__intNumberOfPoints = len(inPoints)
        self.__PeriodicVectors = np.copy(inPeriodicVectors)
        self.__intPeriodicDirections = len(inPeriodicVectors)
        arrInverse = np.linalg.inv(inPeriodicVectors)
        arrExtendedPoints, arrUniqueIndices = PeriodicExtension(inPoints, inPeriodicVectors)
        self.__ExtendedPoints = arrExtendedPoints
        self.__UniqueIndices = arrUniqueIndices
        self.__PeriodicTree = KDTree(self.__ExtendedPoints)
    def Pquery_radius(self, inPoints: np.array, fltRadius: float,blnReturnDistance=True, blnSortResults=True):
        arrIndices,arrDistances = self.__PeriodicTree.query_radius(inPoints, fltRadius,return_distance=blnReturnDistance,sort_results=blnSortResults)
        return arrIndices, arrDistances
    def GetExtendedPoints(self):
        return self.__ExtendedPoints
    def GetPeriodicIndices(self, inRealIndices: list)->list:
        return list(map(lambda x: self.__UniqueIndices[x],inRealIndices))
    def Pquery(self,inPoints:np.array,k=1):
        arrDistances, arrIndices = self.__PeriodicTree.query(inPoints,k)
        return arrDistances, arrIndices
    def GetOriginalPoints(self):
        return self.__OriginalPoints
    def MinimumDisplacement(self):
        arrDistances, arrIndices = self.Pquery(np.array([[0,0,0]]),k=len(self.__ExtendedPoints))
        arrDistances = arrDistances[np.argsort(arrIndices)]
        arrStackedDistances = np.reshape(arrDistances, (self.__intNumberOfPoints,2**self.__intPeriodicDirections))
        arrStackedIndices = np.reshape(arrIndices, (self.__intNumberOfPoints,2**self.__intPeriodicDirections))
        arrMinimum = np.argmin(arrStackedDistances, axis=1)
        arrMinimumIndices = arrStackedIndices[arrMinimum]
        return self.__ExtendedPoints[arrMinimumIndices], arrDistances[arrMinimum]            

def FindPrimitiveVectors( inLatticePoints):
        arrMediod = FindMediod(inLatticePoints)
        inLatticePoints = inLatticePoints - arrMediod
        arrDistances = np.round(np.linalg.norm(inLatticePoints,axis=1),5)
        lstPositions = FindNthSmallestPosition(arrDistances, 1)
        arrVector1 = inLatticePoints[lstPositions[0]]
        blnFound1 = False
        i = 1
        if len(lstPositions) > 1: #there are atleast two equidistance vectors
                j = 1
                while j < len(lstPositions) and not(blnFound1):
                        arrVector2 = inLatticePoints[lstPositions[j]]
                        if np.any(np.abs(np.round(np.cross(arrVector2,arrVector1),5)) > 0):
                                blnFound1  = True
                        j +=1
        if not(blnFound1): #equidistant vectors were all parallael
                blnFound2 = False
                while i < len(arrDistances) and not(blnFound2):
                        lstPositions = FindNthSmallestPosition(arrDistances,i)
                        j=0
                        while j < len(lstPositions) and not(blnFound2):
                                arrVector2 = inLatticePoints[lstPositions[j]]
                                if np.any(np.abs(np.round(np.cross(arrVector2,arrVector1),5)) > 0):
                                        blnFound2  = True
                                j +=1
                        i +=1
        blnFound3 = False
        while i < len(arrDistances) and not(blnFound3):
                lstPositions = FindNthSmallestPosition(arrDistances,i)
                k = 0
                while k < len(lstPositions) and not(blnFound3):
                        arrVector3 = inLatticePoints[lstPositions[k]]
                        if abs(np.linalg.det(np.array([arrVector1,arrVector2,arrVector3]))) >1e-5:
                                blnFound3  = True
                        k +=1
                i += 1
        lstVectors = []
        lstVectors.append(arrVector2)
        lstVectors.append(arrVector1)
        lstVectors.append(arrVector3)
        arrPrimitiveVectors = np.vstack(lstVectors)
        return arrPrimitiveVectors
def PrimitiveToOrthogonalVectorsGrammSchmdit(inPrimitiveVectors,arrPrimitiveCells):
        arrLengths = np.linalg.norm(inPrimitiveVectors,axis=1)
        arrRows = np.argsort(arrLengths)
        arrPrimitiveVectors = 2*inPrimitiveVectors[arrRows]
        intL = len(arrPrimitiveVectors)
        lstVectors = []
        for i in range(0,intL):
                intVector = 0
                arrProjections = np.zeros(3)
                intVector = 0
                v= 2*arrPrimitiveVectors[i]
                while intVector < len(lstVectors):
                        u = lstVectors[intVector]
                        arrProjections += np.dot(u,v)/np.dot(u,u)*u
                        intVector +=1
                arrVector = (v -arrProjections)
                if len(lstVectors):
                       u = lstVectors[-1]
                       arrVector = arrVector*(np.dot(u,u))
                lstVectors.append(arrVector)
        arrVectors = np.vstack(lstVectors)
        for a in range(len(arrVectors)):
                arrP = np.matmul(arrVectors[a],np.linalg.inv(arrPrimitiveCells)).astype('int')
                arrP = arrP/np.gcd.reduce(arrP)
                arrVectors[a] = np.matmul(np.transpose(arrP), arrPrimitiveCells)
        arrRows = np.argsort(arrLengths)[::-1]
        return arrVectors[arrRows]

                       
                



def PrimitiveToOrthogonalVectors(inPrimitiveVectors, inAxis): #trys to find orthogonal vectors from primitive vectors
        lstAllVectors = [] 
        arrAllVectors = np.copy(inPrimitiveVectors)       
        for a in inPrimitiveVectors:
                lstAllVectors.append(arrAllVectors + a)
                lstAllVectors.append(arrAllVectors -a)
                arrAllVectors = np.vstack(lstAllVectors)
        arrAllVectors = np.unique(arrAllVectors, axis=0)
        arrDeleteRows = np.where(np.all(arrAllVectors == np.zeros(3),axis=1))[0]
        arrPlane = np.delete(arrAllVectors, arrDeleteRows, axis=0)
        arrRows = np.where(np.abs(np.matmul(arrPlane, np.transpose(inAxis)))< 1e-5)[0]
        arrPlane = arrPlane[arrRows]
        lstPlaneVectors = []
        arrAllPlane = np.copy(arrPlane)
        for b in arrPlane:
                lstPlaneVectors.append(arrAllPlane + b)
                lstPlaneVectors.append(arrAllPlane -b)
                arrAllPlane = np.vstack(lstPlaneVectors)
        arrAllPlane = np.unique(arrAllPlane, axis=0)
        arrDeleteRows = np.where(np.all(arrAllPlane == np.zeros(3),axis=1))[0]
        arrAllPlane = np.delete(arrAllPlane, arrDeleteRows, axis=0)
        arrRows, arrCols = np.where(np.abs(np.matmul(arrAllPlane,np.transpose(arrAllPlane))) < 1e-5)
        arrReturnVectors = np.zeros([3,3])
        if len(arrRows) > 1:
                arrRowVectors = arrAllPlane[arrRows]
                arrRowDistances = np.linalg.norm(arrRowVectors, axis=1)
                intPosition = FindNthSmallestPosition(arrRowDistances,0)[0]
                arrRowVector = arrRowVectors[intPosition]
                arrColPositions = np.where(arrRows == arrRows[intPosition])[0]
                arrColVectors = arrAllPlane[arrCols[arrColPositions]]
                arrColDistances = np.linalg.norm(arrColVectors, axis=1)
                arrColVector = arrColVectors[np.argmin(arrColDistances)]
                arrReturnVectors[1] = arrRowVector
                arrReturnVectors[0] = arrColVector
                arrReturnVectors[2] = inAxis
        else:
                arrPlaneDistances = np.linalg.norm(arrAllPlane, axis=1)
                lstPositions = FindNthSmallestPosition(arrPlaneDistances,0)
                if len(lstPositions) > 1:
                        arrReturnVectors[1] = arrAllPlane[lstPositions[0]]
                        arrReturnVectors[0] = arrAllPlane[lstPositions[1]]
                        arrReturnVectors[2] = inAxis
                else:
                        arrReturnVectors[1] = arrAllPlane[lstPositions[0]]
                        arrReturnVectors[0] = arrAllPlane[FindNthSmallestPosition(arrPlaneDistances,1)]
                        arrReturnVectors[2] = inAxis
        return arrReturnVectors


class OLattice(object):
        def __init__(self,arrOTransformation: np.array, arrOPoint: np.array):
                self.__OTransformation = arrOTransformation
                self.__BasisVectors = []
                self.__OVectors = []
                self.__OPoint = arrOPoint
                x,y,z = sy.symbols('x y z')
                arrOArray = np.identity(3) - np.linalg.inv(arrOTransformation)
                symMatrix = sy.Matrix(arrOArray.tolist())
                lstOVectors = []
                lstDiscretePoints = []
                for i in range(3):
                        arrDir = np.zeros(3)
                        arrDir[i] = 1
                        system = A, b=symMatrix, arrDir.tolist()
                        rtnValue = sy.linsolve(system,x,y,z)
                        if rtnValue != sy.EmptySet:
                                lstValue = list(*rtnValue)
                                k = 0
                                for j in lstValue:
                                        if type(j) is  sy.core.symbol.Symbol:
                                                lstValue[k] = 0
                                        k += 1
                                lstDiscretePoints.append(lstValue)
                arrEValues, arrEVectors = np.linalg.eig(self.__OTransformation)
                arrRows = np.where(np.round(arrEValues,10) ==1)[0]
                for k in arrRows:
                        arrVector = arrEVectors[:,k]
                        if np.all(np.isreal(arrVector)): #strips off 0j part 
                                arrVector = np.real(arrVector)
                        lstOVectors.append(arrVector)
                if len(lstOVectors)> 0:                        
                        self.__OVectors = np.unique(np.vstack(lstOVectors),axis=0)
                arrTranslation = arrOPoint -np.matmul(arrOTransformation, arrOPoint)
                self.__Translation = arrTranslation - np.round(arrTranslation,0)
                self.__DiscretePoints = np.array(lstDiscretePoints)
        def GetBasisVectors(self):
                return self.__BasisVectors
        def GetOVectors(self):
                return self.__OVectors
        def GetTranslation(self): #this is the vector translation from the base lattice to the transformed lattice given the OPoint
                return self.__Translation
        def GetDiscretePoints(self):
                return self.__DiscretePoints

def TripleLineTensor(arrTripleLine: np.array, lstOfBases: list, lstOfGBAngles: list):
        arrTensor= np.zeros([3,3])
        arrUnit = NormaliseVector(arrTripleLine)
        for k in range(len(lstOfGBAngles)):
                arrTensor += np.matmul(lstOfBases[np.mod(k+1,3)] - lstOfBases[np.mod(k,3)], RotatedBasisVectors(lstOfGBAngles[k],arrUnit)) 
        return arrTensor

def ConvertToLAMMPSBasis(arrBasisVectors: np.array):   #takes a general 3d Basis and writes in the form [x 0 0], [y, yx , 0] [zx zy z] where x > 0, yx >0, z>0
        arrIdent = np.identity(3)
        if np.linalg.det(arrBasisVectors) < 0:
                arrIdent[0,0] = -1
                arrBasisVectors = np.matmul(arrIdent,arrBasisVectors)
        if np.round(np.dot(NormaliseVector(arrBasisVectors[0]),np.array([1,0,0])),10) < 1:
                fltAngle1, arrAxis1 = FindRotationVectorAndAngle(arrBasisVectors[0], np.array([1,0,0])) #align first row vector with x axis
                arrBasisVectors2 = RotateVectors(fltAngle1, arrAxis1, arrBasisVectors)
                arrTransform1 = RotatedBasisVectors(fltAngle1, arrAxis1)
        else:
                arrBasisVectors2 = arrBasisVectors
                arrTransform1 = StandardBasisVectors(3)
        arrVector2 = NormaliseVector(arrBasisVectors2[1])
        if np.abs(arrVector2[2]) > 1e-5:
                arrInYZ = np.zeros(3)
                arrInYZ[1:] =arrVector2[1:]
                arrFinal = np.array([0, np.linalg.norm(arrInYZ),0])
                fltAngle2, arrAxis2 = FindRotationVectorAndAngle(arrInYZ,arrFinal) 
                arrTransform2 = RotateVectors(fltAngle2,arrAxis2,arrTransform1)
                arrReturn = RotateVectors(fltAngle2,arrAxis2,arrBasisVectors2)
        else:
                arrReturn = arrBasisVectors2
                arrTransform2 = arrTransform1
        
        return arrReturn, arrTransform2    

def ConfidenceAndPredictionBands(x,y, fltPercent):


        slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment
        y_model = np.polyval([slope, intercept], x)   # modeling...
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        n = x.size                        # number of samples
        m = 2                             # number of parameters
        dof = n - m                       # degrees of freedom
        t = stats.t.ppf(fltPercent, dof)       # Students statistic of interval confidence
        residual = y - y_model
        std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error

        numerator = np.sum((x - x_mean)*(y - y_mean))
        denominator = ( np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2) )**.5
        correlation_coef = numerator / denominator
        r2 = correlation_coef**2
        MSE = 1/n * np.sum( (y - y_model)**2 )
        x_line = np.linspace(np.min(x), np.max(x), 100)
        y_line = np.polyval([slope, intercept], x_line)
# confidence interval
        ci = t * std_error * (1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5
# predicting interval
        pi = t * std_error * (1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5  

        return x_line, y_line, ci, pi

        #objAxis.plot(x, y, 'o', color = 'royalblue')
        #objAxis.plot(x_line, y_line, color = 'royalblue')
        #objAxis.fill_between(x_line, y_line + pi, y_line - pi, color = 'lightcyan', label = '95% prediction interval')
        #objAxis.fill_between(x_line, y_line + ci, y_line - ci, color = 'skyblue', label = '95% confidence interval')

#def CSLMultipleJunction(self, inAxis: np.array, #intSigmaMax = 300):


def OrthogonalVectorsFromPrimitiveVectors(inArray: np.array):
        arrCol = inArray[:,0]
        arrTemp = np.copy(inArray)
        intMin = np.argmin(np.abs(arrCol))
        if intMin != 0:
                arrTemp[:,[0,intMin]] = arrTemp[:, [intMin,0]]
        for i in range(len(inArray)):
                arrRow = arrTemp[i]
                for k in range(i+1,3):
                        if arrRow[i] !=0:
                                fltRatio = arrTemp[k,i]/arrRow[i]
                                arrTemp[k,i:] = arrTemp[k,i:] -fltRatio*arrRow[i:]
        arrPrimitive = np.matmul(arrTemp, np.linalg.inv(ld.FCCPrimitive))
        arrPrimitive = arrPrimitive.astype('int')
        # for a in range(len(arrPrimitive)):
        #         arrPrimitive[a] = arrPrimitive[a]/np.gcd.reduce(arrPrimitive[a])
        arrOut = np.matmul(arrPrimitive, ld.FCCPrimitive)
                
                        
        return arrOut
# arrExample = 2*np.array([[-1,  -1, -1, ],
#  [-3.,  -8.5, -5.5],
#  [ 0. ,  0.5 , 0.5]])
# arrVectors = OrthogonalVectorsFromPrimitiveVectors(arrExample)
# print(arrVectors,np.linalg.det(arrVectors), np.matmul(arrVectors,ld.FCCPrimitive))
class CSLMobility(object):
    def __init__(self, arrCellVectors: np.array, arrLogValues: np.array, arrVolumeSpeed: np.array, strType: str, fltTemp: float, fltUPerVolume: float):
        self.__LogValues = arrLogValues
        self.__CellVectors = arrCellVectors
        self.__VolumeSpeed = arrVolumeSpeed
        self.__Temp = fltTemp
        self.__UValue = fltUPerVolume
        self.__Type = strType
        self.__Volume = np.abs(np.linalg.det(arrCellVectors))
        self.__Area = np.linalg.norm(
            np.cross(arrCellVectors[0], arrCellVectors[2]))
        self.__Mobility = 0
        self.__PEPerVolume = 0
        self.__Scale = len(arrLogValues[:,1]-1)/len(arrVolumeSpeed[1,:]-1)
        self.__MaxRows = min([len(arrLogValues[:,0]),len(arrVolumeSpeed[0])])
        self.__LinearRange = slice(0, self.__MaxRows,1)
    def SetLinearRange(self, intStart, intFinish):
        intStart = min([intStart, self.__MaxRows])
        intFinish = min([intFinish, self.__MaxRows])
        self.__LinearRange = slice(intStart,intFinish,1)
        
    def GetLinearRange(self):
        return self.__LinearRange
    def FitLine(self, x, a, b):
        return a*x + b
    def GetLogValues(self):
        return self.__LogValues
    def GetCellVectors(self):
        return self.__CellVectors
    def GetNormalSpeed(self,intStage: int,fltNormalDistance: float):
        intFinish = self.GetLowVolumeCutOff(intStage, fltNormalDistance)
        popt,pop = optimize.curve_fit(
            self.FitLine, self.__VolumeSpeed[0, self.__LinearRange], self.__VolumeSpeed[2,self.__LinearRange])
        return popt[0]
    def GetPEPerVolume(self,intStage: int,fltNormalDistance: float):
        arrRows = self.GetOverlapRows(intStage)
        arrPEValues = self.__LogValues[arrRows]      
        popt,pop = optimize.curve_fit(
            self.FitLine, self.__VolumeSpeed[1, self.__LinearRange], arrPEValues[self.__LinearRange, 2])
        return popt[0]
    def GetVolumeSpeed(self, intColumn=None):
        if intColumn is None:
            return self.__VolumeSpeed
        else:
            return self.__VolumeSpeed[:, intColumn]
    def GetType(self):
        return self.__Type
    def GetTemp(self):
        return self.__Temp
    def GetPEParameter(self):
        return self.__UValue
    def GetPEString(self):
        return str(self.__UValue).split('.')[1]
    def GetOverlapRows(self, intStage: int):
        arrValues = self.__LogValues
        arrRows = np.where(np.isin(arrValues[:, 0], self.__VolumeSpeed[0, :]))[0]
        return arrRows
    def GetLowVolumeCutOff(self, intStage: int, fltDistance: float):
        arrCellVectors = self.GetCellVectors()
        fltArea = np.linalg.norm(
            np.cross(arrCellVectors[0], arrCellVectors[2]))
        arrValues = self.__VolumeSpeed[1, :]
        arrRows = np.where(arrValues < fltDistance*fltArea)[0]
        if len(arrRows) > 0:
            intReturn = np.min(arrRows)
        else:
            intReturn = len(arrValues)
        return intReturn
    def SetMobility(self, inMobility):
        self.__Mobility = inMobility
    def GetMobility(self):
        return self.__Mobility
    def SetPEPerVolume(self, inPE):
        self.__PEPerVolume = inPE
    def GetVolumeOrLAMMPSLog(self, lstVolume: list, lstLAMMPS: list,intStart =100):
        arrLogValues = self.GetLogValues()
        arrVolumeSpeed = self.GetVolumeSpeed()
        if len(lstLAMMPS) == 0:
                x = arrVolumeSpeed[lstVolume[0],intStart:self.__MaxRows]
                y = arrVolumeSpeed[lstVolume[1],intStart:self.__MaxRows]
        elif len(lstVolume) == 0:
                x = arrLogValues[intStart:self.__MaxRows,lstLAMMPS[0]]
                y = arrLogValues[intStart:self.__MaxRows,lstLAMMPS[1]]
        else:
                x = arrVolumeSpeed[lstVolume[0],intStart:self.__MaxRows]
                y = arrLogValues[intStart:self.__MaxRows,lstLAMMPS[0]]
        return x,y
    def GetPlanarArea(self):
        return self.__Area



def FindIntersectionsNPointSets(lstAllMeshPoints: list, inPeriodicCellVectors: np.array, fltWidth: float, intMinIntersections: int):
        intPos = 0
        # for k in lstAllMeshPoints:
        #         clustering = DBSCAN(fltWidth).fit(k)
        #         arrLabels = clustering.labels_
        #         arrUniqueLabels,arrCounts = np.unique(arrLabels,return_counts=True)
        #         arrRows1 = np.where(arrCounts > 1)[0]
        #         arrRows2 = np.where(np.isin(arrLabels, arrUniqueLabels[arrRows1]))[0]
        #         lstAllMeshPoints[intPos] = k[arrRows2]
        #         intPos +=1
        intMeshs = len(lstAllMeshPoints)
        arrAllPoints = np.unique(np.vstack(lstAllMeshPoints),axis=0)
        objTreeAll = PeriodicWrapperKDTree(arrAllPoints,inPeriodicCellVectors, FindConstraintsFromBasisVectors(inPeriodicCellVectors),2*fltWidth,['p','p','p']) 
        lstPermutations = list(it.combinations(list(range(intMeshs)),intMinIntersections))
        lstAllOverlap = []
        for k in lstPermutations:
                blnStop = False
                i = 0
                lstOverlap = []
                while not(blnStop) and i < intMinIntersections:
                        arrIndices = objTreeAll.Pquery_radius(lstAllMeshPoints[k[i]],fltWidth)[0]
                        lstIndices = mf.FlattenList(arrIndices)
                        if len(lstOverlap) ==0:
                                lstOverlap = lstIndices
                        else:
                                lstOverlap =list(set(lstOverlap).intersection(set(lstIndices)))
                        if len(lstOverlap) == 0:
                                blnStop = True
                        i += 1
                if not(blnStop):
                        lstAllOverlap.extend(lstOverlap)
        arrIndices = np.unique(lstAllOverlap)
        arrTrueIndices = objTreeAll.GetPeriodicIndices(arrIndices)
        arrPoints = objTreeAll.GetOriginalPoints()[arrTrueIndices]        
        return arrPoints

class EcoOrient(object):
        def __init__(self,fltCutOff: float,fltTolerance: float):
                self.__CutOff = fltCutOff
                self.__Tolerance = fltTolerance
        def GetNormFactor(self,arrPrimitive1):
                arrVectors = GetLinearCombinations(arrPrimitive1,4)
                arrRows = np.where(np.linalg.norm(arrVectors,axis=1) <= self.__CutOff)[0]
                arrVectors = arrVectors[arrRows]
                fltTotalWeight = 0
                zTotal = 0
                arrReciprocal = FindReciprocalVectors(arrPrimitive1)
                for a in arrVectors:
                        fltWeight = self.EcoWeight(a, self.__CutOff)
                        fltTotalWeight += fltWeight
                        for q in arrReciprocal:
                                rtnZ,fltWeight = self.PrimitiveReciprocalProduct(a,q)
                                zTotal += rtnZ*rtnZ.conjugate()
                return 3*fltTotalWeight**2-np.real(zTotal)
        def PrimitiveReciprocalProduct(self, arrPrimitive, arrReciprocal):
                fltWeight = self.EcoWeight(arrPrimitive,self.__CutOff)
                z = np.complex(0,np.dot(arrPrimitive,arrReciprocal))
                rtnZ = fltWeight*np.complex(np.exp(2*np.pi*z))
                return rtnZ, fltWeight
        def EcoOrientPsiFunction(self,arrTestVector, arrPrimitiveBasis1):#fit of basis2 with respect to basis1
                arrQ = FindReciprocalVectors(arrPrimitiveBasis1)
                rtnPsi = np.complex(0)
                for q in arrQ:
                        rtnZ = np.complex(0)
                        fltTotalWeight = 0
                        for a in arrTestVector:
                                Z,fltWeight  = self.PrimitiveReciprocalProduct(a,q)
                                rtnZ += Z
                                fltTotalWeight += fltWeight
                        rtnPsi += rtnZ*rtnZ.conjugate()
                return np.real(np.complex(rtnPsi)), fltTotalWeight
        def EcoWeight(self,inRealVector, fltCutOff):
                if self.__CutOff == 0:
                        fltReturn = 1
                else:
                        fltLength = np.linalg.norm(inRealVector)/fltCutOff
                        if np.round(fltLength,5) < 1:
                                fltReturn =  fltLength**4 -2*fltLength**2 + 1
                        else:
                                fltReturn = 0
                return fltReturn 
        def GetOrderParameter(self,arrTestVectors,arrPrimitive1,arrPrimitive2):
                N = self.GetNormFactor(arrPrimitive1)
                flt1 = self.EcoOrientPsiFunction(arrTestVectors,arrPrimitive1)[0]
                flt2 = self.EcoOrientPsiFunction(arrTestVectors,arrPrimitive2)[0]
                fltValue = (flt1-flt2)/N
                if fltValue > self.__Tolerance:
                        fltReturn = 1
                elif fltValue < -self.__Tolerance:
                        fltReturn = -1
                else:
                        fltReturn = np.sin(fltValue*np.pi/(2*self.__Tolerance))
                return fltReturn
def GetLinearCombinations(arr3Vectors, intNLimit: int):
        lstAllVectors = []
        for i in range(-intNLimit,intNLimit):
                for j in range(-intNLimit,intNLimit):
                        for k in range(-intNLimit,intNLimit):
                                lstAllVectors.append(arr3Vectors[0]*i+arr3Vectors[1]*j+arr3Vectors[2]*k)
        return np.vstack(lstAllVectors)
def GroupClustersPeriodically(lstPoints: np.array, arrPeriodicVectors: np.array, fltMinDistance: float, lstBoundary = ['pp','pp','pp']):
        intLength = len(lstPoints)
        lstAllMatches = []
        lstUsedIndices = []
        for i in range(intLength):
                lstMatches = []
                objPeriodicCell = PeriodicKDTree(lstPoints[i],arrPeriodicVectors, lstBoundary)
                for j in range(i+1, intLength):
                        arrIndices = objPeriodicCell.Pquery_radius(lstPoints[j],fltMinDistance)
                        lstIndices = mf.FlattenList(arrIndices)
                        if len(lstIndices) > 0:
                                lstMatches.append(j)
                                lstMatches.append(i)
                if len(lstMatches) > 0:
                        arrUniqueMatches = np.unique(lstMatches)
                        if not(np.any(np.isin(arrUniqueMatches,lstUsedIndices))):
                                lstAllMatches.append(arrUniqueMatches)
                elif i not in lstUsedIndices:
                        lstAllMatches.append(np.array([i]))
                lstUsedIndices.extend(np.concatenate(lstAllMatches))
                lstUsedIndices = np.unique(lstUsedIndices).tolist()
        return lstAllMatches
def WritePOSCARFile(inCellVectors: np.array, inAtomPositions: np.array, strFilename='POSCAR',strType = None, strConstraints = None):
       arrCellVectors = inCellVectors
       r = inAtomPositions
       intLength = np.shape(inAtomPositions)[0]
       with open(strFilename,'w') as Dfile:
                header = '#VASP coordinate file'
                Dfile.write(header)
                Dfile.write('\n')
        # Lattice parameter already factored into supercell and coordinates
                Dfile.write('1.0\n')
        # Supercell shape
                lstOrder = [0,1,2]
                if np.dot(arrCellVectors[0,:],np.cross(arrCellVectors[1,:],arrCellVectors[2,:])) < 0:
                        lstOrder = [1,0,2]
                for s in range(3):
                        for t in range(3):
                                Dfile.write(str(arrCellVectors[lstOrder[s],t]) + ' ')
                        Dfile.write('\n')
        # Atom positions
                if strType is None:
                        Dfile.write(str(intLength) + '\n')
                else:
                        Dfile.write(strType + '\n')
                Dfile.write('Selective dynamics\n')
                Dfile.write('Cartesian\n')
                for i in range(intLength):
                        Dfile.write(str(r[i,0]) + ' ' + str(r[i,1]) + ' ' + str(r[i,2]))
                        if strConstraints is not None:
                                Dfile.write(' ' + strConstraints[i,0] + ' ' + strConstraints[i,1] + ' ' + strConstraints[i,2])
                        else:
                                Dfile.write(' T T T')
                        Dfile.write('\n')          
                Dfile.flush()
                Dfile.close() 
def FindPrimitiveCellVectors(arrBasis: np.array, arrSymmetries = None):
        lstRows = []
        for i in range(3):
              arrRows = np.zeros(3)
              arrRows[i] = 1
              lstRows.append(arrRows)
              lstRows.append(-arrRows)
        arrSymmetries = np.array(list(map(lambda x: np.array(x),list(it.permutations(lstRows,3)))))
        lstRows = list(map(lambda x: np.linalg.det(x),arrSymmetries))
        arrRows = np.where(abs(np.array(lstRows)) ==1)[0]
        arrSymmetries = arrSymmetries[arrRows]
        arrMatrices = np.matmul(arrBasis, arrSymmetries)
        arrEigenValues1,arrEigenVectors1 = np.linalg.eig(arrMatrices)
        arrRows1, arrCols1 = np.where(np.round(np.imag(arrEigenValues1),10) == 0)
        arrEigenValues2 = np.real(arrEigenValues1[arrRows1,arrCols1])
        arrEigenVectors2 = np.real(arrEigenVectors1[arrRows1,:,arrCols1])
        arrRows2 = np.where(np.round(arrEigenValues2,10) == 1)[0]
        arrRealVectors = arrEigenVectors2[arrRows2]
        #arrLengths = np.abs(arrEigenValues-np.ones(np.shape(arrEigenValues)))
        #arrLengths = np.round(arrLengths,10)
        #arrRows,arrCols = np.where((arrLengths ==0) | (arrLengths ==2))
        #complex number comparisons were unreliable and so this checks
        #whether the eigenvalue is real and it is +/- 1
       # arrRows,arrCols = np.where((arrEigenValues == np.complex(1)) | (arrEigenValues == -np.complex(1)))
        #arrRealVectors = arrEigenVectors[arrRows,:,arrCols]
        #arrRows2 = np.where(np.all(arrRealVectors ==np.real(arrRealVectors),axis=1))[0]
        #arrRealVectors = np.real(arrRealVectors[arrRows2])
        arrRealVectors = np.unique(np.round(arrRealVectors,10), axis=0)
        #arrNonZeroMins = np.array(list(map(lambda x: x/#np.min(np.abs(x[x!=0])), arrRealVectors)))
        arrIntegerVectors =  np.array(list(map(lambda x: recover_integer_vector(x),arrRealVectors)))
        return arrIntegerVectors
def ReducePrimitiveIntegerVectors(inIntegerVectors: np.array, arrPrimitive: np.array):
        intPrimitives = np.matmul(inIntegerVectors,np.linalg.inv(arrPrimitive))
        arrReduced = np.array(list(map(lambda x: x/np.gcd.reduce(x.astype('int')),intPrimitives)))
        return np.matmul(arrReduced, arrPrimitive)
def recover_integer_vector(u, denom=200):
        u /= min(abs(x) for x in u if x)
    # get the denominators of the fractions
        denoms = [Fraction(x).limit_denominator(denom).denominator for x in u]
        if len(denoms) > 1:
               return np.lcm.reduce(denoms)*u
        else:
               return u 
    # multiply the scaled u by LCM(denominators)
        #lcm = lambda a, b: (a * b)/np.gcd.reduce(a, b)
        #return u*reduce(np.lcm.reduce, list(denoms))
               
     

