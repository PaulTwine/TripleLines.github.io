#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt 
from matplotlib.markers import MarkerStyle
from mpl_toolkits.mplot3d import Axes3D
import SmithNormalForm as sn
import scipy as sc
#
#%%
int_sigma = 147
#objMatrix = gf.SigmaRotationMatrix(int_sigma)
#lstMatrix = objMatrix.FindSigmaMatrices()
#arrMatrix = lstMatrix[0]
arrAxis = np.array([1,1,1])
arrSigmas = np.array([21,21,49])
arrCell = gf.CubicCSLGenerator(arrAxis, 100)
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
arrCell = objCSL.FindTripleLineSigmaValues(75)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')== np.array([21,21,49]),axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
arrEdgeVectors =objCSL.GetTJBasisVectors(intIndex,False)
arr_TJ_111_Cell = np.transpose(2*arrEdgeVectors)
#arr_sigma_3 = np.array([[2,-2,1],[2,1,-2],[1,2,2]])/3
#arr_TJ_111_Cell = np.round(np.matmul(arr_sigma_3, arr_TJ_111_Cell),5)
arr_fcc_basis = np.transpose(2*ld.FCCPrimitive)
objBasis = sn.HermiteNormalForm(arr_fcc_basis)
arr_fcc_basis = objBasis.FindHermiteNormalForm()
arr_edge_coordinates = np.matmul(np.linalg.inv(arr_fcc_basis),2*arrEdgeVectors)
obj_edge_csl = sn.SmithNormalForm(arr_edge_coordinates)
print(obj_edge_csl.FindSmithNormal(), np.round(np.linalg.inv(obj_edge_csl.GetRowOperations(),),0))
objCSLSub = gf.CSLSubLatticeBases(2*np.transpose(arrEdgeVectors), arr_fcc_basis)
lstAllTransforms = objCSLSub.FindTransformationsByReciprocalLattice(True)
#print(np.linalg.det(np.matmul(np.linalg.inv(arrCSL), arrNewCSL)))
len(lstAllTransforms)

#%%
#gf.EqualAxis3D(ax)
#ax.set_xlim([np.min(arr_TJ_111_Cell[0,:]),np.max(arr_TJ_111_Cell[0,:])])
#ax.set_ylim([np.min(arr_TJ_111_Cell[1,:]),np.max(arr_TJ_111_Cell[1,:])])
#ax.set_zlim([np.min(arr_TJ_111_Cell[2,:]),np.max(arr_TJ_111_Cell[2,:])])
plt.show()
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
arrEdgeVectors[[1,2],:] = arrEdgeVectors[[2,1],:] 
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
    arrBasis = np.matmul(np.transpose(arrOut[i]), arrTransform)
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
arr_sigmas = np.array([1,3,49,7,21,147])
lst_markers = [".","1","2","3","4", "o"]

lst_fill = ["top","bottom","left", "right", "none", "none"]
lst_colours = ['black',"blue","green","red","purple","gray"]
arr_rows = np.argsort(arr_sigmas)[::-1]
lst_legend = list(map(lambda x: '$\\Sigma$' + str(x), arr_sigmas[arr_rows]))
#%%
x = 5 
for r in arr_rows:
    i = lstPoints[r]
    i = i[i[:,2]< 0.5]
    if x ==0 :
        plt.scatter(i[:,0],i[:,1],s=100, marker=lst_markers[x], c=lst_colours[x])
    elif x ==5:
        plt.scatter(i[:,0],i[:,1],s=250, linewidths=2,c=lst_colours[x],marker=MarkerStyle(lst_markers[x],fillstyle="none"))
    else:
        plt.scatter(i[:,0],i[:,1],s=250,linewidths=2,c=lst_colours[x], marker =MarkerStyle(lst_markers[x],fillstyle="none"))
    x -= 1
plt.legend(lst_legend)
plt.axis('off') 
plt.show()
#%%
x = 5 
for r in arr_rows:
    i = lstPoints[r]
    i = i[(i[:,2] > 0.5) & (i[:,2] < 1.0)]
    if x ==0 :
        plt.scatter(i[:,0],i[:,1],s=100, marker=lst_markers[x], c=lst_colours[x])
    elif x ==5:
        plt.scatter(i[:,0],i[:,1],s=250, linewidths=2,c=lst_colours[x],marker=MarkerStyle(lst_markers[x],fillstyle="none"))
    else:
        plt.scatter(i[:,0],i[:,1],s=250,linewidths=2,c=lst_colours[x], marker =MarkerStyle(lst_markers[x],fillstyle="none"))
    x -= 1
plt.legend(arr_sigmas[arr_rows].tolist())
plt.axis('off') 
plt.show()
#%%
x = 5 
for r in arr_rows:
    i = lstPoints[r]
    i = i[(i[:,2] > 1.0)]
    if x ==0 :
        plt.scatter(i[:,0],i[:,1],s=100, marker=lst_markers[x], c=lst_colours[x])
    elif x ==5:
        plt.scatter(i[:,0],i[:,1],s=250, linewidths=2,c=lst_colours[x],marker=MarkerStyle(lst_markers[x],fillstyle="none"))
    else:
        plt.scatter(i[:,0],i[:,1],s=250,linewidths=2,c=lst_colours[x], marker =MarkerStyle(lst_markers[x],fillstyle="none"))
    x -= 1
plt.legend(arr_sigmas[arr_rows].tolist())
plt.axis('off') 
plt.show()


