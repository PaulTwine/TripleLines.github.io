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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

#strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis221/Sigma9_9_9/Temp'
strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis221/Sigma9_9_9/Temp'
#arrLogV = np.loadtxt(strRoot + 'TemplogV.txt')

#np.save('/home/p17992pt/LAMMPSData/arrLogV.txt',arrLogV)


#strRoot = str(sys.argv[1])
fltKeV = 8.617333262e-5
lstTemp = [700]
lstITemp = list(map(lambda x: 1/x,lstTemp))
#lstTemp =['Temp850Motion']
#plt.scatter(lstITemp,arrLogV)
#plt.show()
#strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis221/Sigma9_9_9/Temp'
#strRoot = '/home/p17992pt/csf4_scratch/CSLTJ/Axis111/Sigma3_7_21/'
dt = 10000
deltaT = 1
#t =1000
dctVelocity = dict()
for i in lstTemp:
    lstTJCluster = []
    lstAllTimes = []
    lstAllPoints = []
    lstProjections = []
    strDir = strRoot + str(i) + '/1Min.lst'
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
    # arrPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[0]+arrCellVectors[2]), 0.5*(arrCellVectors[1]+arrCellVectors[2]), 0.5*(arrCellVectors[0]+arrCellVectors[1]+arrCellVectors[2])])
    # arrInitialMeans = arrInitialMeans[arrRealIndices]
    objInitialTree = gf.PeriodicWrapperKDTree(arrInitialMeans,objLT.GetCellVectors(),gf.FindConstraintsFromBasisVectors(objLT.GetCellVectors()),50)
    strDir = strRoot + str(i) + '/2Min.lst'
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
    objFinalTree = gf.PeriodicWrapperKDTree(arrFinalMeans,objLT.GetCellVectors(),gf.FindConstraintsFromBasisVectors(objLT.GetCellVectors()),50)
    arrDirections = gf.NormaliseMatrixAlongRows(arrDirections)
    for k in range(0,40000+dt,dt):
        if objLT.GetGrainLabels() == [0,1,2,3]:   
            strDir = strRoot + str(i) + '/1Sim' + str(k) + '.dmp'
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
    lstVelocity = []
    arrAllProjections = np.vstack(lstProjections)
    for p in range(4):        
        lstVelocity.append(np.log(stats.linregress(lstAllTimes,arrAllProjections[:,p])[0]))
    lstVelocity.append(np.log(stats.linregress(np.sort(lstAllTimes*4),arrAllProjections.reshape(4*len(arrAllProjections)))[0]))
    dctVelocity[i] = lstVelocity
lstLogVelocity = []
for v in list(lstTemp):
    lstLogVelocity.append(dctVelocity[v])
np.savetxt(strRoot + str(i) + '/logV.txt', np.array(lstLogVelocity))
np.savetxt(strRoot + str(i) + '/lstITemp.txt',np.array(lstITemp))
# plt.scatter(lstITemp, lstLogVelocity)
# plt.show()

# for p in range(len(lstProjections)):
#     plt.scatter(lstAllTimes[p],lstProjections[p][1], c='red')
#     # plt.scatter(lstAllTimes[p],lstProjections[p][1], c='blue')
#     # plt.scatter(lstAllTimes[p],lstProjections[p][2], c='green')
#     # plt.scatter(lstAllTimes[p],lstProjections[p][3], c='orange')
    
    
# # for p in lstAllPoints:
# #     plt.scatter(p[0,0],p[0,1])
# #plt.legend(lstAllTimes)



fig = plt.figure()
ax = plt.axes(projection='3d')

def animate_func(num):
    ax.clear()  # Clears the figure to update the line, point,   
                # title, and axes
    # Updating Trajectory Line (num+1 due to Python indexing)
    #ax.plot3D(*(lstAllPoints[num][0]), c='blue')
    # Updating Point Location 
    ax.scatter(*tuple(zip(*lstTJCluster[num])), c='blue', marker='o')
    # Adding Constant Origin
    #ax.plot3D(*(lstAllPoints[0][0]), c='black', marker='o')
    # Setting Axes Limits
    ax.set_xlim3d([arrInitialMeans[0][0]-50,arrInitialMeans[0][0]+50])
    ax.set_ylim3d([arrInitialMeans[0][1]-50,arrInitialMeans[0][1]+50])
    ax.set_zlim3d([0,objLT.GetCellVectors()[-1,-1]])

    # # Adding Figure Labels
    # ax.set_title('Trajectory \nTime = ' + str(np.round(t[num],    
    #              decimals=2)) + ' sec')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

line_ani = animation.FuncAnimation(fig, animate_func, interval=1000, frames=len(lstAllPoints))
writergif = animation.PillowWriter(fps=len(lstAllPoints)/6)
line_ani.save('/home/p17992pt/LAMMPSData/Animation1.gif', writer=writergif)
plt.show()


