# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%%
from IPython import get_ipython
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
#import pyvista as pv

#strDumpFile1 = str(sys.argv[1]) # pass the file name and relative path if required as an argument
#%%
lstPE = []
lstAtoms = []
lstNames = ['TJ','12BV','13BV','32BH']
for j in lstNames:
    strDumpFile1 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp450/u01/' + j + '/1Min.lst'
    objData = LT.LAMMPSData(strDumpFile1,1, 4.05, LT.LAMMPSGlobal)
    objProcess = objData.GetTimeStepByIndex(-1)
    lstPE.append(np.sum(objProcess.GetColumnByName('c_pe1')))
    lstAtoms.append(objProcess.GetNumberOfAtoms())
#%%
l = 4*objProcess.GetCellVectors()[2,2]
intExcessAtoms = np.sum(lstAtoms[1:])-lstAtoms[0]
fltPEExcess = lstPE[0]-np.sum(lstPE[1:])
print((fltPEExcess+intExcessAtoms*(-3.36))/l)
# objProcess.CategoriseAtoms()    
# objProcess.WriteDumpFile(strDumpFile1)
# objProcess.LabelAtomsByGrain(fltTolerance = 3.14)
# objProcess.RefineGrainLabels()
# objProcess.FindJunctionLines()
# objProcess.FinaliseGrainBoundaries()
# objProcess.AssignVolumes()
# objProcess.AssignPE()
# objProcess.AssignAdjustedMeshPoints()
# objProcess.AppendGrainBoundaries()
# objProcess.AppendJunctionLines()
# objProcess.WriteDefectData(strDumpFile1[:-3] +'dfc')
# objProcess.WriteDumpFile(strDumpFile1)












# def Loop(inlstIDs):
#     lstNewIDs = []
#     for x in inlstIDs:
#         arrCentre1 = arrPoints[arrPoints[:,0] == x]
#         lstNewIDs.extend(HigherPEPoints(arrCentre1[:,1:4]))
#     return list(set(lstNewIDs).difference(inlstIDs))

# def HigherPEPoints(arrPoint):
#     lstReturn = []
#    # lstIndices = objHex.FindSphericalAtoms(arrPoints[:, 0:4], arrPoint[0,1:4],4.05*np.sqrt(3)/2)
#     lstIndices = objNearest.kneighbors(arrPoint,return_distance = False)[0]
#     lstIDs = objHex.GetRows(lstIndices)[:,0].astype('int')
#     arrPE = np.sort(objHex.GetColumnByIDs(lstIDs,7))#[1:-1]
#     fltCurrentPE = np.mean(arrPE)
#     if fltCurrentPE > -3.36:
#         for i in lstIDs:
#             arrCentre = objHex.GetAtomsByID([i])[:,1:4]
#             #lstNewIndices = objHex.FindSphericalAtoms(arrPoints[:, 0:4], arrCentre[0],4.05*np.sqrt(3)/2)
#             lstNewIndices = objNearest.kneighbors(arrCentre, return_distance = False)[0]
#             lstNewIDs = objHex.GetRows(lstNewIndices)[:,0].astype('int')
#             arrPE = np.sort(objHex.GetColumnByIDs(lstNewIDs,7))#[1:-1]
#             fltNewPE = np.mean(arrPE)
#             if fltNewPE >= fltCurrentPE:
#                   lstReturn.append(int(i))
#     return lstReturn



# objdct = dict()
# intSigma = 7
# intGBNumber = 2
# strRoot = '/home/p17992pt/csf3_scratch/CSL/Axis111/Temp700/Sigma7/'
# strRoot = '/home/p17992pt/csf3_scratch/Hex/Axis100/data1/'
# objData = LT.LAMMPSData(strRoot+ '1Sim1500.dmp',1,4.05, LT.LAMMPSGlobal)
# objHex = objData.GetTimeStepByIndex(-1)
# #objHex.ReadInDefectData(strRoot + '1Sim1000.dfc')
# fltMean = np.mean(objHex.GetPTMAtoms()[:,7])
# print(fltMean)
# objHex.LabelAtomsByGrain()
# objHex.RefineGrainLabels()
# objHex.FindJunctionLines() 
# objHex.FinaliseGrainBoundaries()
# objHex.AssignPE()
# objHex.AssignVolumes()
# objHex.AssignAdjustedMeshPoints()
# for k in objHex.GetGrainBoundaryIDs():
#     pts = objHex.GetGrainBoundary(k).GetSurfaceMesh()
# #print(objHex.GetGrainBoundary(intGBNumber).GetEnergyPerSurfaceArea(-3.36),objHex.GetGrainBoundary(intGBNumber).GetGrainBoundaryWidth())
# #pts = gf.GetBoundaryPoints(pts, 8, 7.5)
#     cloud = pv.PolyData(pts)
#     cloud.plot()
# #surf = cloud.delaunay_2d()
# #surf.plot(show_edges=True)



# lstDefects = objHex.GetRemainingDefectAtomIDs()
# #print(gf.FindDelaunayArea(objHex.GetGrainBoundary(2).GetMeshPoints()))
# lstTJIDs = objHex.GetJunctionLine(intGBNumber).GetAtomIDs()


# lstAddition = objHex.SpreadToHigherEnergy(lstTJIDs)

# fig = plt.figure(figsize=plt.figaspect(1)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
# ax = fig.gca(projection='3d')
# ax.scatter(*tuple(zip(*objHex.GetAtomsByID(lstAddition)[:,1:4])), c='b')
# #ax.scatter(*tuple(zip(*objHex.GetAtomsByID(lstTJIDs)[:,1:4])), c='r')
# #ax.scatter(*tuple(zip(*objHex.GetAtomsByID(lstDefects)[:,1:4])), c='b')
# gf.EqualAxis3D(ax)
# #plt.xlim(0, 360)
# #plt.ylim(0,360)
# plt.show()
# print(np.mean(objHex.GetColumnByIDs(lstAddition,7)), np.mean(objHex.GetColumnByIDs(lstTJIDs,7)),print(len(lstDefects)), print(objHex.GetGrainAtomIDs(-1)))

# %%
