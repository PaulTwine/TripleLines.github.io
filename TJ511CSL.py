#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt 
import SmithNormalForm as sn
import scipy as sc
#
#%%
int_sigma = 27
#objMatrix = gf.SigmaRotationMatrix(int_sigma)
#lstMatrix = objMatrix.FindSigmaMatrices()
#arrMatrix = lstMatrix[0]
arrAxis = np.array([5,1,1])
arrSigmas = np.array([9,9,9])
arrCell = gf.CubicCSLGenerator(arrAxis, 100)
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')== np.array([9,9,9]),axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
arrEdgeVectors =objCSL.GetTJBasisVectors(intIndex,False)
arr_TJ_511_cell = np.transpose(2*arrEdgeVectors)
#arr_sigma_3 = np.array([[2,-2,1],[2,1,-2],[1,2,2]])/3
#arr_TJ_511_cell = np.round(np.matmul(arr_sigma_3, arr_TJ_511_cell),5)
arr_fcc_basis = np.transpose(2*ld.FCCPrimitive)
objBasis = sn.HermiteNormalForm(arr_fcc_basis)
arr_fcc_basis = objBasis.FindHermiteNormalForm()
arr_edge_coordinates = np.matmul(np.linalg.inv(arr_fcc_basis),2*arrEdgeVectors)
obj_edge_csl = sn.SmithNormalForm(arr_edge_coordinates)
print(obj_edge_csl.FindSmithNormal(), np.round(np.linalg.inv(obj_edge_csl.GetLeftMatrix(),),0))
objCSLSub = gf.CSLSubLatticeBases(2*np.transpose(arrEdgeVectors), arr_fcc_basis)
lstAllTransforms = objCSLSub.FindTransformationsByReciprocalLattice(True)
#print(np.linalg.det(np.matmul(np.linalg.inv(arrCSL), arrNewCSL)))
len(lstAllTransforms)

#%%
# %%
dctValues = dict()
lstGCDs = []
lstS = []
lstValues = []
for j in lstAllTransforms:
    n = int_sigma/np.gcd.reduce(np.round(np.unique(j*int_sigma)).astype('int'))  
    if n not in lstGCDs:
        dctValues[n] = [j]
        lstGCDs.append(n)
        lstS.append(j)
    else:
        lstValues = dctValues[n]
        lstValues.append(j)
        dctValues[n] = list(np.unique(lstValues,axis=0))
print(np.unique(lstGCDs), len(lstS))
# %%
lstHermite= []
for i in lstAllTransforms:
    n = np.gcd.reduce(np.round(np.unique(i*int_sigma)).astype('int'))
    arrNewBasis = np.linalg.matmul(n*i,arr_fcc_basis)
    obj_smith_normal = sn.HermiteNormalForm(arrNewBasis)
    lstHermite.append(obj_smith_normal.FindHermiteNormalForm())

#lstHermite = list(map(lambda x : sn.HermiteNormalForm(np.linalg.matmul(x,arr_fcc_basis)).GetTransformedMatrix(), lstAllTransforms))
arrHermite, arrIndex = np.unique(np.round(np.array(lstHermite),0),axis=0, return_index=True)


# %%
arrOut = np.array(lstAllTransforms)

# %%
arrCellBasis = arrEdgeVectors/2
arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrEdgeVectors)
arrEdgeVectors = np.abs(np.round(arrEdgeVectors, 10))
objSimulationCell = gl.SimulationCell(arrEdgeVectors)
arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors, arrTransform, ld.FCCCell, np.ones(3), np.zeros(3))
lstPoints = []
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.RemoveAtomsOnOpenBoundaries()
objSimulationCell.WriteLAMMPSDataFile('/home/paul-twine/' + '0.dmp')
objSimulationCell.RemoveAllGrains()
arrPoints = arrGrain1.GetAtomPositions()
lstPoints.append(arrPoints)
objPTree = gf.PeriodicWrapperKDTree(arrGrain1.GetAtomPositions(), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis), 50, ['p', 'p', 'p'])
intTransform = 0
lstTransforms = []
lstTransforms.append(gf.StandardBasisVectors(3))
lstAxes = []
for i in range(len(arrOut)):
    objSimulationCell = gl.SimulationCell(arrEdgeVectors)
    arrBasis = np.matmul(arrOut[i], arrTransform)
    arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors, arrBasis, ld.FCCCell, np.ones(3), np.zeros(3))
    arrPoints = gf.WrapVectorIntoSimulationCell(arrEdgeVectors, arrGrain1.GetAtomPositions())
    arrPoints = objSimulationCell.RemoveRealDuplicates(arrPoints, 1e-5)
    arrDistances, arrIndices = objPTree.Pquery(arrPoints, k=1)
    arrDistances = np.array(mf.FlattenList(arrDistances))
    if not(np.all(arrDistances < 1e-5)):
        objSimulationCell.AddGrain(arrGrain1)
        objSimulationCell.RemoveGrainPeriodicDuplicates()
        lstPoints.append(arrPoints)
        #lstAxes.append(arrAxes[i])
        objSimulationCell.RemoveAtomsOnOpenBoundaries()
        #ax.scatter(*tuple(zip(*lstPoints[-1])))
        objSimulationCell.WriteLAMMPSDataFile('/home/paul-twine/' + str(intTransform+1) + '.dmp')
        objSimulationCell.RemoveAllGrains()
        objPTree = gf.PeriodicWrapperKDTree(np.vstack(lstPoints), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis), 50, ['p', 'p', 'p'])
        lstTransforms.append(arrOut[i])
        intTransform += 1
arrPoints = np.unique(np.vstack(lstPoints), axis=0)
arr_sigma_9 = gf.GetMatrixFromAxisAngle(arrAxis, arrCSL[0,1])
#print(np.matmul(arr_sigma_9,arr_sigma_9)*9)


# %%
def get_rotation_matrix_axis(in_matrix):
    arr_vals, arr_vectors = np.linalg.eig(in_matrix)
    arr_rows = np.where(np.round(arr_vals.real,5) ==float(1))[0]
    arr_axis = np.round(arr_vectors[:,arr_rows].real,5)
    flt_min = np.min(np.abs(arr_axis[np.abs(arr_axis) > 1e-5]))
    return arr_axis/flt_min
    
#%%
lst_new_t = []
arr_sigma_3 = lstTransforms[3]
i = 0
for t in lstTransforms:
    x= np.round(np.matmul(arr_sigma_3,np.transpose(t)),10)
    obj_sn = sn.GenericCSL(t,arr_fcc_basis)
    print(i)
    print(obj_sn.FindSmithNormal(), np.round(np.linalg.inv(obj_sn.GetLeftMatrix()),0))
    if np.linalg.det(x) > 0:
        print(get_rotation_matrix_axis(x))
        print(x*27)
        print(t*9)
    lst_new_t.append(np.round(x,10))
    i +=1
#%%
for t in lstTransforms:
    arr_coords = np.round(np.matmul(np.linalg.inv(np.matmul(np.transpose(t),arr_fcc_basis)),arr_TJ_511_cell),5)
    obj_sn = sn.SmithNormalForm(arr_coords)
    print("diagonal form",obj_sn.FindSmithNormal())
#%%

arr_TJ_cell_new = np.round(np.matmul(arr_sigma_3, arr_TJ_511_cell),5)
obj_csl_new = gf.CSLSubLatticeBases(arr_TJ_cell_new, arr_fcc_basis)
lst_transformations_new = obj_csl_new.FindTransformationsByReciprocalLattice()
#%%
dctValues = dict()
lstGCDs = []
lstS = []
lstValues = []
for j in lst_transformations_new:
    n = int_sigma/np.gcd.reduce(np.round(np.unique(j*int_sigma)).astype('int'))
    if n not in lstGCDs:
        dctValues[n] = [j]
        lstGCDs.append(n)
        lstS.append(j)
    else:
        lstValues = dctValues[n]
        lstValues.append(j)
        dctValues[n] = list(np.unique(lstValues,axis=0))
print(np.unique(lstGCDs), len(lstS))
