# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:38:14 2019

@author: twine
"""
import numpy as np
import itertools as it
def RealDistance(inPointOne, inPointTwo)->float:
        fltDistance = 0
        for i in range(len(inPointOne)):
                       fltDistance += (inPointOne[i]-inPointTwo[i])**2
        return np.sqrt(fltDistance)
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
def CheckLinearConstraint(inPoints: np.array, inConstraint: np.array)-> np.array:
        lstDeletedPoints = []
        intDimensions = len(inConstraint)-1
        for j in range(len(inPoints)):
               if ((np.dot(inPoints[j],inConstraint[:-1]) > inConstraint[intDimensions])):
                   lstDeletedPoints.append(j)
        if len(lstDeletedPoints) != 0:                
                arrInsidePoints = np.delete(inPoints, lstDeletedPoints, axis=0)
        else:
                arrInsidePoints = inPoints
        return arrInsidePoints
def CheckLinearEquality(inPoints: np.array, inConstraint: np.array)->np.array:
        lstDeletedPoints = []
        intDimensions = len(inConstraint)-1
        for j in range(len(inPoints)):
               if ((np.dot(inPoints[j],inConstraint[:-1]) == inConstraint[intDimensions])):
                   lstDeletedPoints.append(j)
        if len(lstDeletedPoints) != 0:                
                arrOnPoints = np.delete(inPoints, lstDeletedPoints, axis=0)
        else:
                arrOnPoints = inPoints
        return arrOnPoints

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
def GetQuaternion(inVector: np.array, inAngle)->np.array:
        inAngle = inAngle/180*np.pi
        vctAxis = NormaliseVector(inVector)
        lstQuarternion  = []
        C = np.cos(inAngle/2)
        S = np.sin(inAngle/2)
        lstQuarternion.append(C)
        lstQuarternion.append(vctAxis[0]*S)
        lstQuarternion.append(vctAxis[1]*S)
        if len(inVector) == 3:
                lstQuarternion.append(vctAxis[2]*S)
        return np.array(lstQuarternion)
def QuaternionProduct(inVectorOne: np.array, inVectorTwo:np.array )->np.array:
        if len(inVectorOne) != 4 or len(inVectorTwo) != 4:
                raise "Error quarternions must be 4 dimensional arrays"
        else:
                r1 = inVectorOne[0]
                r2 = inVectorTwo[0]
                v1 = np.delete(inVectorOne , 0)
                v2 = np.delete(inVectorTwo, 0)
                r = r1*r2 - np.dot(v1,v2)
                v  =  r1*v2 + r2*v1 + np.cross(v1,v2)
                return np.array([r, v[0],v[1],v[2]])
def QuaternionConjugate(inVector: np.array)->np.array:
        return np.array([inVector[0],-inVector[1],-inVector[2],-inVector[3]])