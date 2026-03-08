#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt 
import SmithNormalForm as sn
import scipy as sc
#%%
def ConvertRationalVectorToIntegers(inVector: np.array, intIter = 5000):
    fltMin = np.min(np.abs(inVector[np.abs(inVector)>0]))
    lstFactors = []
    if fltMin > 0:
        arrTest = inVector/fltMin
        for i in arrTest:
            blnInt = False
            n = 1
            while not(blnInt) and n < intIter:
                j = i*n
                if j.isinteger():
                    blnInt=True
                    lstFactors.append(int(n))
                n +=1
        arrOut = np.lcm.reduce(lstFactors)*arrTest
    else:
        arrOut = inVector
    return arrOut 

#%%
intSigma = 3*5*7*11*13
objMatrix = gf.SigmaRotationMatrix(intSigma)
lstMatrix = objMatrix.FindSigmaMatrices()
arrMatrix = lstMatrix[1]
arrBasis = np.transpose(2*ld.FCCPrimitive)
objBasis = sn.HermiteNormalForm(arrBasis)
arrBasis = objBasis.FindHermiteNormalForm()
# %%for j in range(3):
objCon = sn.GenericCSLandDSC(arrMatrix,arrBasis)
arrCSL = objCon.GetCSLPrimtiveCell()
# %%
objintMatrix = sn.HermiteNormalForm(np.trunc(np.round(arrCSL,0)))
objintMatrix.ReduceCoefficentMagnitude()
arrNewCSL = objintMatrix.get_gram_schmidt()
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

##%
# lstPairs = []
# lstNonMod = []
# for i in dctValues[3]:
#     for j in dctValues[105]:
#         arrM = np.matmul(i,j)
#       #  print(arrM*intSigma)
#         #if np.abs(np.round(np.linalg.det(arrM),5)) !=1.0:
#       #  if np.any(np.round(arrM*intSigma,5) != np.round(arrM*intSigma,0)) or np.abs(np.round(np.linalg.det(arrM),5)) !=1.0:
#         if np.round(arrM*intSigma,0) in np.round(intSigma*np.array(lstAllTransforms),0):
#         #if np.min(arrD) == 0 and np.max(arrD) == 0:
#             lstPairs.append([i*intSigma,j*intSigma])
#             lstNonMod.append(arrM)
# #%%
# #print(lstPairs)
# print(len(lstNonMod))
# for i in lstNonMod:
#     print(i*intSigma)
#%%
#for i in lstPairs:
##    print(i[0], " and \n", i[1])
#%%
#print(np.unique(lstDet))
# %%
arrOut = np.array(lstS)
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
#objPTree = gf.PeriodicWrapperKDTree(arrGrain1.GetAtomPositions(
#), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis), 50, ['p', 'p', 'p'])
intTransform = 0
lstTransforms = []
#lstTransforms.append(gf.StandardBasisVectors(3))
lstAxes = []
for i in range(len(arrOut)):
    objSimulationCell = gl.SimulationCell(arrEdgeVectors)
    arrBasis = np.matmul(arrOut[i], arrTransform)
    arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors, arrBasis, ld.FCCCell, np.ones(3), np.zeros(3))
    arrPoints = gf.WrapVectorIntoSimulationCell(arrEdgeVectors, arrGrain1.GetAtomPositions())
    arrPoints = objSimulationCell.RemoveRealDuplicates(arrPoints, 1e-5)
    #arrDistances, arrIndices = objPTree.Pquery(arrPoints, k=1)
    #arrDistances = np.array(mf.FlattenList(arrDistances))
    #if not(np.all(arrDistances < 1e-5)):
    objSimulationCell.AddGrain(arrGrain1)
    objSimulationCell.RemoveGrainPeriodicDuplicates()
    lstPoints.append(arrPoints)
        #lstAxes.append(arrAxes[i])
    objSimulationCell.RemoveAtomsOnOpenBoundaries()
        #ax.scatter(*tuple(zip(*lstPoints[-1])))
    objSimulationCell.WriteLAMMPSDataFile('/home/paul-twine/' + str(intTransform+1) + '.dmp')
    objSimulationCell.RemoveAllGrains()
    #objPTree = gf.PeriodicWrapperKDTree(np.vstack(lstPoints), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis), 50, ['p', 'p', 'p'])
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