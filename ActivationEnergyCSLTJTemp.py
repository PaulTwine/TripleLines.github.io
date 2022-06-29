import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import transforms
from scipy import optimize
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT 
import LatticeDefinitions as ld
import re
import sys 
import matplotlib.lines as mlines
from scipy import stats
import MiscFunctions as mf
from matplotlib import animation

#strRoot = str(sys.argv[1])
#intTemp = int(sys.argv[2])
#intSteps = int(sys.argv[3])
#intLimit = int(sys.argv[4])
strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis221/Sigma9_9_9/Temp750/'
intTemp = 750
intSteps = 100000
intLimit = 400000

fltKeV = 8.617333262e-5
#strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis111/Sigma3_7_21/'
lstTJCluster = []
lstAllTimes = []
lstAllPoints = []
lstProjections = []
strDir = strRoot  + '1Min.lst'
objData = LT.LAMMPSData(strDir,1,4.05,LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
objLT.PartitionGrains(0.99, 25)
objLT.MergePeriodicGrains(25)
arrCellVectors = objLT.GetCellVectors()
ids = objLT.FindMeshAtomIDs([1,2,3])
pts = objLT.GetAtomsByID(ids)[:,1:4]
lstMerged = gf.MergePeriodicClusters(pts,objLT.GetCellVectors(), ['p','p','n'],5)
lstInitialMeans = list(map(lambda x: np.mean(x,axis=0),lstMerged))
arrInitialMeans = np.stack(lstInitialMeans)
objInitialTree = gf.PeriodicWrapperKDTree(arrInitialMeans,objLT.GetCellVectors(),gf.FindConstraintsFromBasisVectors(objLT.GetCellVectors()),50)
strDir = strRoot  + '2Min.lst'
objData = LT.LAMMPSData(strDir,1,4.05,LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
objLT.PartitionGrains(0.99, 25)
objLT.MergePeriodicGrains(25)
ids = objLT.FindMeshAtomIDs([1,2,3])
pts = objLT.GetAtomsByID(ids)[:,1:4]
lstMerged = gf.MergePeriodicClusters(pts,objLT.GetCellVectors(), ['p','p','n'],5)
lstFinalMeans = list(map(lambda x: np.mean(x,axis=0),lstMerged))
arrFinalMeans = np.stack(lstFinalMeans)
arrDistances, arrIndices = objInitialTree.Pquery(arrFinalMeans)
arrRealIndices = objInitialTree.GetPeriodicIndices(np.ravel(arrIndices))
arrFinalMeans = arrFinalMeans[arrRealIndices]
arrExtendedPoints = objInitialTree.GetExtendedPoints()[np.ravel(arrIndices)[arrRealIndices],:]
arrDirections = arrFinalMeans - arrExtendedPoints
arrProjectedDistances = np.linalg.norm(arrDirections,axis=1)
objFinalTree = gf.PeriodicWrapperKDTree(arrFinalMeans,objLT.GetCellVectors(),gf.FindConstraintsFromBasisVectors(objLT.GetCellVectors()),50)
arrDirections = gf.NormaliseMatrixAlongRows(arrDirections)
for k in range(0,intLimit+intSteps,intSteps):
    if objLT.GetGrainLabels() == [0,1,2,3]:   
        strDir = strRoot + '1Sim' + str(k) + '.dmp'
        objData = LT.LAMMPSData(strDir,1,4.05,LT.LAMMPSAnalysis3D)
        objLT = objData.GetTimeStepByIndex(-1)
        objLT.PartitionGrains(0.99,25)
        objLT.MergePeriodicGrains(5)
        ids2 = objLT.FindMeshAtomIDs([1,2,3])
        pts2 = objLT.GetAtomsByID(ids2)[:,1:4]
        lstMerged2 = gf.MergePeriodicClusters(pts2,objLT.GetCellVectors(), ['p','p','n'],20)
        lstMeanPoints = list(map(lambda x: np.mean(x,axis=0),lstMerged2))          
        arrMeanPoints = np.stack(lstMeanPoints)
        arrDistances,arrIndices = objInitialTree.Pquery(arrMeanPoints,1)           
        arrRealIndices = objInitialTree.GetPeriodicIndices(np.ravel(arrIndices))
        arrMeanPoints = arrMeanPoints[arrRealIndices]
        arrDistances,arrIndices = objInitialTree.Pquery(arrMeanPoints,1)
        lstTJCluster.append(lstMerged2[arrRealIndices[0]])
        arrExtendedPoints = objInitialTree.GetExtendedPoints()[np.ravel(arrIndices),:]
        arrTranslation = arrInitialMeans - arrExtendedPoints
        arrAllPoints = (arrMeanPoints+arrTranslation) -arrInitialMeans
        lstAllPoints.append(arrAllPoints)
        arrProjections = (np.matmul(arrAllPoints[:,:2],np.transpose(arrDirections[:,:2]))).diagonal()
        lstProjections.append(arrProjections)
        lstAllTimes.append(k)
lstLogVelocity = []
arrAllProjections = np.vstack(lstProjections)
for p in range(4): 
    intMin = np.max(np.where(arrAllProjections[:,p] < 1)[0])
    intMax = np.min(np.where(arrAllProjections[:,p] > arrProjectedDistances[p]-1)[0])
    lstLogVelocity.append(np.log(stats.linregress(lstAllTimes[intMin:intMax+1],arrAllProjections[intMin:intMax+1,p])[0]))
arrPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[0]+arrCellVectors[2]), 0.5*(arrCellVectors[1]+arrCellVectors[2]), 0.5*(arrCellVectors[0]+arrCellVectors[1]+arrCellVectors[2])])
objPositionsTree = gf.PeriodicWrapperKDTree(arrPositions,objLT.GetCellVectors(),gf.FindConstraintsFromBasisVectors(objLT.GetCellVectors()),50)
arrDistances, arrIndices = objPositionsTree.Pquery(arrFinalMeans)
arrRealIndices = objPositionsTree.GetPeriodicIndices(np.ravel(arrIndices))
lstLogVelocity = list(np.array(lstLogVelocity)[arrRealIndices])
lstLogVelocity.append(np.log(stats.linregress(np.sort(lstAllTimes*4),arrAllProjections.reshape(4*len(arrAllProjections)))[0]))

np.savetxt(strRoot + 'logV.txt', np.array(lstLogVelocity))
np.savetxt(strRoot + 'lstITemp.txt',np.ones(len(lstLogVelocity))/intTemp)



