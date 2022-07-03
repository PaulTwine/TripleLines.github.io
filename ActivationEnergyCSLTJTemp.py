import numpy as np
#import os
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import transforms
#from scipy import optimize
from scipy import stats
#from sklearn.neighbors import NearestNeighbors
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT 
import LatticeDefinitions as ld
#import re
import sys 
#import matplotlib.lines as mlines
from scipy import stats
import MiscFunctions as mf
#from matplotlib import animation

# strRoot = str(sys.argv[1])
# intTemp = int(sys.argv[2])
# intSteps = int(sys.argv[3])
# intLimit = int(sys.argv[4])
intTemp = 600
intSteps = 100000
intLimit = 400000
strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis221/Sigma9_9_9/Temp'
strRoot += str(intTemp) + '/'

fltKeV = 8.617333262e-5
lstAllTimes = []
lstAllPoints = []
lstProjections = []
lstCrossProjections = []
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
fltNearest = 4.05/np.sqrt(2)
ids = objLT.FindMeshAtomIDs([1,2,3])
pts = objLT.GetAtomsByID(ids)[:,1:4]
lstMerged = gf.MergePeriodicClusters(pts,objLT.GetCellVectors(), ['p','p','n'],5)
lstFinalMeans = list(map(lambda x: np.mean(x,axis=0),lstMerged))
arrFinalMeans = np.stack(lstFinalMeans)
arrDistances, arrIndices = objInitialTree.Pquery(arrFinalMeans)
arrRealIndices = objInitialTree.GetPeriodicIndices(np.ravel(arrIndices))
arrFinalMeans = arrFinalMeans[arrRealIndices]
objFinalTree = gf.PeriodicWrapperKDTree(arrFinalMeans,objLT.GetCellVectors(),gf.FindConstraintsFromBasisVectors(objLT.GetCellVectors()),50)
for k in range(0,intLimit+intSteps,intSteps):
    strDir = strRoot + '1Sim' + str(k) + '.dmp'
    objData = LT.LAMMPSData(strDir,1,4.05,LT.LAMMPSAnalysis3D)
    objLT = objData.GetTimeStepByIndex(-1)
    objLT.PartitionGrains(0.99,25)
    objLT.MergePeriodicGrains(5)
    ids2 = objLT.FindMeshAtomIDs([1,2,3])
    pts2 = objLT.GetAtomsByID(ids2)[:,1:4]
    lstMerged2 = gf.MergePeriodicClusters(pts2,objLT.GetCellVectors(), ['p','p','n'],20)
    if objLT.GetGrainLabels() == [0,1,2,3] and len(lstMerged2) ==4:   
        lstMeanPoints = list(map(lambda x: np.mean(x,axis=0),lstMerged2))          
        arrMeanPoints = np.stack(lstMeanPoints)
        arrDistances,arrIndices = objInitialTree.Pquery(arrMeanPoints,1)           
        arrRealIndices = objInitialTree.GetPeriodicIndices(np.ravel(arrIndices))
        arrMeanPoints = gf.WrapVectorIntoSimulationCell(objLT.GetCellVectors(),arrMeanPoints[arrRealIndices])
        arrDistances,arrIndices = objInitialTree.Pquery(arrMeanPoints,1)
        arrInitialPoints = objInitialTree.GetExtendedPoints()[np.ravel(arrIndices),:]
        arrDistances,arrIndices = objFinalTree.Pquery(arrInitialPoints)
        arrFinalPoints = objFinalTree.GetExtendedPoints()[np.ravel(arrIndices),:]
        arrDirections = gf.NormaliseMatrixAlongRows(arrFinalPoints-arrInitialPoints)
        arrCross = np.cross(arrDirections, np.array([0,0,1]))
        arrTranslations = arrMeanPoints -arrInitialPoints
        arrProjections = (np.matmul(arrTranslations[:,:2],np.transpose(arrDirections[:,:2]))).diagonal()
        arrCrossProjections = (np.matmul(arrTranslations[:,:2],np.transpose(arrCross[:,:2]))).diagonal()
        lstProjections.append(arrProjections)
        lstCrossProjections.append(arrCrossProjections)
        lstAllTimes.append(k)
lstLogVelocity = []
arrAllProjections = np.vstack(lstProjections)
arrAllCrossProjections = np.vstack(lstCrossProjections)
arrPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[0]+arrCellVectors[2]), 0.5*(arrCellVectors[1]+arrCellVectors[2]), 0.5*(arrCellVectors[0]+arrCellVectors[1]+arrCellVectors[2])])
objPositionsTree = gf.PeriodicWrapperKDTree(arrPositions,objLT.GetCellVectors(),gf.FindConstraintsFromBasisVectors(objLT.GetCellVectors()),50)
arrDistances, arrIndices = objPositionsTree.Pquery(arrFinalMeans)
arrRealIndices = objPositionsTree.GetPeriodicIndices(np.ravel(arrIndices))
arrAllProjections = arrAllProjections[:,arrRealIndices]
arrCrossProjections  = arrCrossProjections[:, arrRealIndices]

np.savetxt(strRoot + 'Projections.txt', arrAllProjections)
np.savetxt(strRoot + 'CrossProjections.txt', arrCrossProjections)
np.savetxt(strRoot + 'ITemp.txt',np.ones(1)/intTemp)
np.savetxt(strRoot + 'Times.txt', np.array(lstAllTimes))



