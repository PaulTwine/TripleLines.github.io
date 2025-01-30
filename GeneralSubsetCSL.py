#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
import SmithNormalForm as sn
#%%
objMatrix = gf.SigmaRotationMatrix(2025)
lstMatrix = objMatrix.FindSigmaMatrices()
arrMatrix = lstMatrix[-5]
arrBasis = np.array([[1,1,0],[1,0,1],[0,1,1]])
# %%
objCon = sn.GenericCSLandDSC(arrMatrix,arrBasis)
objCon.GetCSLPrimtiveCell()
arrCSL = np.matmul(arrBasis,np.matmul(objCon.GetLeftCoordinates(),objCon.GetLeftScaling()))
# %%
objCSLSub = gf.CSLSubLatticeBases(arrCSL, arrBasis)
lstTransforms = objCSLSub.FindTransformationsByReciprocalLattice(True)
print(len(lstTransforms))
# %%
for j in lstTransforms:
    print(np.round(np.matmul(np.linalg.inv(arrBasis),np.matmul(np.transpose(j), arrCSL)),7))
# %%
