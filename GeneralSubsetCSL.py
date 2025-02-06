#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
import SmithNormalForm as sn
#%%
objMatrix = gf.SigmaRotationMatrix(1155)
lstMatrix = objMatrix.FindSigmaMatrices()
arrMatrix = lstMatrix[-20]
arrBasis = np.array([[1,1,0],[0,1,1],[1,0,1]])
arrV = np.linalg.eig(arrMatrix)
intPos = np.where(np.abs(arrV[0])==1)[0]
arrAxis = np.round(arrV[1][:,intPos],10)
#arrAxis = arrAxis/arrAxis[np.argmin(np.abs(arrAxis))]
#print(arrAxis)
#print(arrV,arrV/arrV[np.argmin(np.abs(arrV))])
arrOut = gf.CubicCSLGenerator(np.array([0,1,5]),20)
print(arrOut[11])
#arrMatrix = gf.GetMatrixFromAxisAngle(np.array([0,1,5]),arrOut[11,1])
#%%
objSN = sn.SmithNormalForm(arrMatrix*1155)
objSN.FindSmithNormal()
np.matmul(np.linalg.inv(objSN.GetLeftMatrix()).astype('int'),np.matmul(objSN.GetTransformedMatrix().astype('int'),np.linalg.inv(objSN.GetRightMatrix()).astype('int')))
#objSN.GetTransformedMatrix()
# %%
objCon = sn.GenericCSLandDSC(arrMatrix,arrBasis)
objCon.GetCSLPrimtiveCell()
arrCSLL = np.matmul(arrBasis,np.matmul(objCon.GetLeftCoordinates(),objCon.GetLeftScaling()))
arrCSLR = np.matmul(arrMatrix,np.matmul(arrBasis, np.matmul(objCon.GetRightMatrix(),objCon.GetRightScaling())))
print(arrCSLL,arrCSLR)
# %%
objintMatrix = sn.SmithNormalForm(np.round(arrCSLL,1).astype('int'))
objintMatrix.FindLowerTriangular()
objintMatrix.ReduceByFirstCol(2)
objintMatrix.ReduceByFirstCol(1)
arrNewCSL = objintMatrix.GetTransformedMatrix()
objCSLSub = gf.CSLSubLatticeBases(arrNewCSL, arrBasis)
lstTransforms = objCSLSub.FindTransformationsByReciprocalLattice(False)
len(lstTransforms)
# %%
lstGCDs = []
for j in lstTransforms:
    print(np.round(j*1155))
    lstGCDs.append(np.gcd.reduce(np.round(np.unique(j*1155)).astype('int')))
print(np.unique(lstGCDs))
# %%
print(np.round(np.matmul(np.linalg.inv(np.matmul(arrMatrix,arrBasis)), arrCSLR),7))
print(np.matmul(objCon.GetRightCoordinates(),objCon.GetRightScaling()))

# %%
objintMatrix = sn.SmithNormalForm(np.round(arrCSLL,1).astype('int'))
objintMatrix.FindLowerTriangular()
objintMatrix.GetTransformedMatrix()
objintMatrix.ReduceByFirstCol(2)
objintMatrix.GetTransformedMatrix()
objintMatrix.ReduceByFirstCol(1)
np.linalg.norm(objintMatrix.GetTransformedMatrix(),axis=0)
# %%
print((np.matmul(objCon.GetLeftMatrix(), np.matmul(objCon.GetConjugateTransitionMatrix() ,objCon.GetRightMatrix()))))

# %%
np.matmul(arrMatrix, np.matmul(arrBasis,np.matmul(objCon.GetRightCoordinates(),objCon.GetRightScaling())))
# %%
print(objCon.GetLeftCoordinates().astype('int'),objCon.GetRightMatrix().astype('int'))

# %%
np.gcd.reduce(np.unique(arrMatrix*105).astype('int'))
# %%
