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
intSigma = 27
#objMatrix = gf.SigmaRotationMatrix(intSigma)
#lstMatrix = objMatrix.FindSigmaMatrices()
#arrMatrix = lstMatrix[0]
arrMatrix = np.array([[  2, -14, -23], [  7, -22,  14], [-26,  -7,   2]])/27
#arrAxis = np.array([7,-1,7])
#arrCell = gf.CubicCSLGenerator(arrAxis, 100)
#arrMatrix = gf.GetMatrixFromAxisAngle(arrAxis,arrCell[7,1])
# arrSigmas = np.array([9,9,9])
# arrCell = gf.CubicCSLGenerator(arrAxis, 100)
# objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell) 
# arrCell = objCSL.FindTripleLineSigmaValues(75)
# intIndex = np.where(np.all(arrCell[:,:,0].astype('int')== np.array([9,9,9]),axis=1))[0][0]
# arrCSL = arrCell[intIndex]
# objCSL.GetTJSigmaValue(arrCSL)
# objCSL.GetTJBasisVectors(intIndex,True)
# arrCSL = objCSL.GetSimulationCellBasis()
# arrMatrix = objCSL.GetRotationMatrix()
# arrMatrix = gf.GetMatrixFromAxisAngle(arrAxis,2*np.pi/3)
arrBasis = np.transpose(2*ld.FCCPrimitive)
objBasis = sn.HermiteNormalForm(arrBasis)
arrBasis = objBasis.FindHermiteNormalForm()
# %%for j in range(3):
objCon = sn.GenericCSLandDSC(arrMatrix,arrBasis)
arrCSL = objCon.GetCSLPrimitiveCell()
# %%
objintMatrix = sn.HermiteNormalForm(np.trunc(np.round(arrCSL,0)))
objintMatrix.ReduceCoefficentMagnitude()
arrNewCSL = objintMatrix.FindLLLForm(0.99)
objCSLSub = gf.CSLSubLatticeBases(objintMatrix.GetTransformedMatrix(), arrBasis)
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
    n = np.gcd.reduce(np.round(np.unique(j*intSigma)).astype('int'))
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
    n = np.gcd.reduce(np.round(np.unique(i*intSigma)).astype('int'))
    arrNewBasis = np.linalg.matmul(n*i,arrBasis)
    obj_smith_normal = sn.HermiteNormalForm(arrNewBasis)
    lstHermite.append(obj_smith_normal.FindHermiteNormalForm())

#lstHermite = list(map(lambda x : sn.HermiteNormalForm(np.linalg.matmul(x,arrBasis)).GetTransformedMatrix(), lstAllTransforms))
arrHermite, arrIndex = np.unique(np.round(np.array(lstHermite),0),axis=0, return_index=True)


# %%
arrOut = np.array(lstAllTransforms)
# arrOut = np.array(lstS)
# %%
arrCellBasis = np.transpose(arrNewCSL)/2
arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrCellBasis)
arrEdgeVectors = np.round(arrEdgeVectors, 10)
objSimulationCell = gl.SimulationCell(arrEdgeVectors)
arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors, arrTransform, ld.FCCCell, np.ones(3), np.zeros(3))
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(*tuple(zip(*arrGrain1.GetAtomPositions())))
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
#plt.show()
arrPoints = np.unique(np.vstack(lstPoints), axis=0)
# Matrix R is either the change of basis or you need to multiply
# the arrCellBasis by all the lstUnitMatrices
# objSimulationCell.GetCoincidentLatticePoints(['1','2'])
#lstSigma = list(map(lambda x: np.gcd.reduce(np.unique(x)),lstTransforms))




# %%
print(len(lstTransforms))