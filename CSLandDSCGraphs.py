import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it

def PutAtomPositionsIntoList(inEdgeVectors: np.array, inLatticeBases: np.array, fltLatticeParameter):
    a = fltLatticeParameter
    lstPoints = []
    objSimulationCell = gl.SimulationCell(inEdgeVectors)
    for i in inLatticeBases:
        arrGrain1 = gl.ParallelopiedGrain(inEdgeVectors,a*i,ld.FCCCell,np.ones(3), np.zeros(3))
        objSimulationCell.AddGrain(arrGrain1,str(i))
        arrPoints = arrGrain1.GetAtomPositions()
        lstPoints.append(arrPoints)
    return lstPoints
def MakeCoincidentDictionary(inlstTransformations: list,inEdgeVectors: np.array, fltTolerance =1e-5)->dict():
    objSimulationCell = gl.SimulationCell(inEdgeVectors)
    intL = len(inlstTransformations)
    i =0 
    for g in inlstTransformations:
        arrGrain = gl.ParallelopiedGrain(inEdgeVectors,g,ld.FCCCell,np.ones(3),np.zeros(3))
        arrGrain.FindBoundaryPoints(inEdgeVectors)
        objSimulationCell.AddGrain(arrGrain)
    objSimulationCell.RemoveAtomsOnOpenBoundaries()
    objSimulationCell.WrapAllAtomsIntoSimulationCell()
    arrAllPoints = objSimulationCell.GetAllAtomPositions()
    objKDTreeAll = gf.PeriodicWrapperKDTree(arrAllPoints,inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), 20)
    dctCoincidence = dict()
    lstOverlapIndices = list(range(len(arrAllPoints)))
    for i in range(intL):
        for j in range(i+1,intL):
            ptsI = objSimulationCell.GetGrain(str(i+1)).GetAtomPositions()
            ptsJ = objSimulationCell.GetGrain(str(j+1)).GetAtomPositions()
            objKDTreeI = gf.PeriodicWrapperKDTree(ptsI,inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), 20)
            objKDTreeJ = gf.PeriodicWrapperKDTree(ptsJ,inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), 20)
            arrDistancesI,arrIndicesI = objKDTreeI.Pquery(ptsJ)
            arrRowsI = np.where(arrDistancesI <=fltTolerance)[0]
            arrIndicesI = arrIndicesI[arrRowsI] 
            arrIndicesI = np.unique(mf.FlattenList(arrIndicesI))
            arrIndicesI = objKDTreeI.GetPeriodicIndices(arrIndicesI)
            arrIndicesI = np.unique(arrIndicesI)
            arrDistancesJ,arrIndicesJ = objKDTreeJ.Pquery(ptsI[arrIndicesI])
            arrRowsJ = np.where(arrDistancesJ <=fltTolerance)[0]
            arrIndicesJ = arrIndicesJ[arrRowsJ]
            arrExtendedPointsJ = objKDTreeJ.GetExtendedPoints()
            arrIndicesJ = mf.FlattenList(arrIndicesJ)
            dctCoincidence[(i,j)] = np.unique((ptsI[arrIndicesI]+arrExtendedPointsJ[arrIndicesJ])/2,axis=0)
            arrIndicesAll = objKDTreeAll.Pquery_radius(ptsI,fltTolerance)[0]
            arrIndicesAll = mf.FlattenList(arrIndicesAll)
            arrIndicesAll =  objKDTreeAll.GetPeriodicIndices(arrIndicesAll)
            arrIndicesAll = np.unique(arrIndicesAll).tolist()
            lstOverlapIndices = list(set(lstOverlapIndices).intersection(arrIndicesAll))
    arrTJ = np.unique(arrAllPoints,axis=0)
    return dctCoincidence, arrTJ
def GetBicrystalAtomicLayer(inGrainsList: list,indctCoincidence: dict(),inEdgeVectors: np.array,intLayer: int, intNumberOfLayers:int,arrPlaneNormal, a: float):
    nlength = np.linalg.norm(arrPlaneNormal)
    arrUnitNormal = arrPlaneNormal/nlength
    lstPoints = []
    nMin =  intLayer*nlength/intNumberOfLayers -0.05  
    nMax =  intLayer*nlength/intNumberOfLayers +0.05
    objSimulationCell = gl.SimulationCell(a*inEdgeVectors)
    for i in inGrainsList:
        arrGrain1 = gl.ParallelopiedGrain(a*inEdgeVectors,i,ld.FCCCell,
        a*np.ones(3), np.zeros(3))
        objSimulationCell.AddGrain(arrGrain1,str(i))
        arrPoints = arrGrain1.GetAtomPositions()
        arrRows = np.where((nMin <= np.dot(arrPoints,arrUnitNormal)) & (np.dot(arrPoints,arrUnitNormal) <= nMax))[0]
        if len(arrRows) > 0:
            arrPoints = arrPoints[arrRows]
            lstPoints.append(arrPoints)
    lstCoincidence = []
    for j in indctCoincidence:
        arrCoincidence = indctCoincidence[j]
        arrRows = np.where((nMin <= np.dot(arrCoincidence,arrUnitNormal)) &(np.dot(arrCoincidence,arrUnitNormal) <= nMax))[0]
        if len(arrRows) > 0:
            lstCoincidence.append(arrCoincidence[arrRows])
        else:
            lstCoincidence.append([])
    return lstPoints,lstCoincidence

