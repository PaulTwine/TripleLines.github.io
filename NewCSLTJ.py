#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

#%%
class CSLConjugateBases(object):
    def __init__(self,inBasis: np.array):
        self.__Basis = inBasis
    def DiagonalUnitEigenMatrices(self,arrCSLBasis: np.array):
        lstMatrices = []
        lstTranslations = []
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    arrMatrix = np.array([np.array([(-1)**i,0,0]),np.array([0,(-1)**j,0]),np.array([0,0,(-1)**k])]).astype('float')
                    lstMatrices.append(arrMatrix)
                    lstTranslations.append(i*arrCSLBasis[0]+j*arrCSLBasis[1] + k*arrCSLBasis[2])
        return lstMatrices, lstTranslations
    def SwapEqualLengthCSLVectors(self, inCSLBasis):
        lstSwapMatrices = []
        arrLengths = np.linalg.norm(inCSLBasis, axis=0)
        arrUniqueLengths = np.unique(arrLengths)
        arrStandardBasis = gf.StandardBasisVectors(3)
        arrBasis = arrStandardBasis
        if len(arrUniqueLengths) == 2:
            for i in arrUniqueLengths:
                arrCols = np.where(arrLengths == i)[0]
                if len(arrCols) == 2:
                    print(str(arrCols) + ' equal length CSL vectors')
                    arrBasis[:,arrCols] = arrStandardBasis[:,arrCols[::-1]]
                    arrLastCol = list(set(range(3)).difference(arrCols))[0]
                    arrVector = np.matmul(arrBasis,inCSLBasis[:, arrLastCol])
                   # if np.all(arrVector == inCSLBasis[:,arrLastCol]) or np.all(arrVector == -inCSLBasis[:,arrLastCol]):
                    arrBasis[:,arrLastCol] = -arrBasis[:,arrLastCol]
                    lstSwapMatrices.append(arrBasis)
                    arrBisector = gf.NormaliseVector(inCSLBasis[:,arrCols[0]] + inCSLBasis[:,arrCols[1]])
                    arrCross = gf.NormaliseVector(np.cross(inCSLBasis[:,arrCols[0]],inCSLBasis[:,arrCols[1]]))
                    arrAxis = np.cross(arrBisector,arrCross)
                    arrMatrix1 = np.transpose(gf.GetMatrixFromAxisAngle(arrAxis, np.pi))
                    fltAngle2, arrAxis2 = gf.FindRotationVectorAndAngle(inCSLBasis[:,arrCols[0]],inCSLBasis[:,arrCols[1]])
                    arrMatrix2 = np.transpose(gf.GetMatrixFromAxisAngle(arrAxis2, fltAngle2))
                    arrMatrix3 = np.transpose(gf.FindReflectionMatrix(arrBisector))
                    lstSwapMatrices.append(arrMatrix3)
                    lstSwapMatrices.append(arrMatrix1)
                    lstSwapMatrices.append(arrBasis)
                    lstSwapMatrices.append(np.matmul(arrMatrix1,arrBasis))
                    lstSwapMatrices.append(np.matmul(arrBasis,arrMatrix2))
        elif len(arrUniqueLengths) == 1:
            lstRows = list(itertools.permutations(list(range(3))))
            for k in lstRows:
                lstSwapMatrices.append(arrStandardBasis[list(k)])
        return lstSwapMatrices
    def SwapCSLRows(self, inCSLBasis):
        lstSwapMatrices = []
        arrLengths = np.linalg.norm(inCSLBasis, axis=0)
        arrUniqueLengths = np.unique(arrLengths)
        arrStandardBasis = gf.StandardBasisVectors(3)
        if len(arrUniqueLengths) == 2:
            for i in arrUniqueLengths:
                arrRows = np.where(arrLengths == i)[0]
                if len(arrRows) == 2:
                    print(str(arrRows) + ' equal length CSL vectors')
                    arrStandardBasis[arrRows] = arrStandardBasis[arrRows[::-1]]
                    lstSwapMatrices.append(arrStandardBasis)
        elif len(arrUniqueLengths) == 1:
            lstRows = list(itertools.permutations(list(range(3))))
            for k in lstRows:
                lstSwapMatrices.append(arrStandardBasis[list(k)])
        return lstSwapMatrices
    def FindSwapTransformations(self,inCSLBasis: np.array):
        lstSwapMatrices = []
        arrLengths = np.linalg.norm(inCSLBasis, axis=1)
        arrUniqueLengths = np.unique(arrLengths)
        arrStandardBasis = gf.StandardBasisVectors(3)
        for i in arrUniqueLengths:
            arrRows = np.where(arrLengths == i)[0]
            arrLastRow = list(range(3))
            if len(arrRows) == 2:
                arrBisector = gf.NormaliseVector(inCSLBasis[arrRows[0]] + inCSLBasis[arrRows[1]])
                arrLastRow = list(set(range(3)).difference(arrRows))[0]
                arrPlaneNormal = gf.NormaliseVector(np.cross(arrBisector, gf.NormaliseVector(inCSLBasis[arrLastRow])))
                arrReflection = gf.FindReflectionMatrix(arrPlaneNormal)
                lstSwapMatrices.append(arrReflection)
                fltAngle,arrAxis = gf.FindRotationVectorAndAngle(inCSLBasis[arrRows[0]],inCSLBasis[arrRows[1]])
                arrRotation = gf.GetMatrixFromAxisAngle(arrAxis,fltAngle)
                lstSwapMatrices.append(arrRotation)
                arrRotation = gf.GetMatrixFromAxisAngle(arrBisector, np.pi)
                lstSwapMatrices.append(arrRotation)
                arrPerpendicular = gf.NormaliseVector(np.cross(arrBisector, inCSLBasis[arrLastRow]))
                arrRotation = gf.GetMatrixFromAxisAngle(arrPerpendicular,np.pi)
                lstSwapMatrices.append(arrRotation)
            elif len(arrRows) == 1:
                lstRows = list(itertools.permutations(list(range(3))))
                for k in lstRows:
                    lstSwapMatrices.append(arrStandardBasis[list(k)])
            else:
                lstSwapMatrices.append(arrStandardBasis)
        return lstSwapMatrices
    def FindConjugates(self, inCSLBasis):
        inPBasis = np.matmul(inCSLBasis,np.linalg.inv(ld.FCCPrimitive))
        arrStandardBasis = gf.StandardBasisVectors(3)
        lstUnitMatrices,lstTranslations = self.DiagonalUnitEigenMatrices(inCSLBasis)
        lstConjugateBases = []
        lstAllMatrices = []
        lstSwapCSLMatrices = self.FindSwapTransformations(inCSLBasis)
        lstSwapCSLMatrices.append(gf.StandardBasisVectors(3))
        inUnitCSL = gf.NormaliseMatrixAlongRows(inCSLBasis)
       # lstSwapCSLMatrices.append(np.linalg.qr(inCSLBasis)[0])
        lstAllMatrices = [] #np.unique(lstAllMatrices,
        # for j in lstSwapCSLMatrices:
        #     k = np.matmul(inCSLBasis,np.transpose(j)) 
        lstConjugateBases.extend(list(map(lambda x: np.transpose(np.matmul(np.matmul(np.transpose(inCSLBasis), x),np.linalg.inv(np.transpose(inCSLBasis)))),lstUnitMatrices)))
        lstAllMatrices.append(lstSwapCSLMatrices)
        lstAllMatrices.append(lstConjugateBases)
      #  lstAllMatrices.append(lstSwapCSLMatrices)
        # for j in lstSwapCSLMatrices:
        #     lstAllMatrices.append(list(map(lambda x: np.matmul(x,j), lstConjugateBases)))
        #     lstAllMatrices.append(list(map(lambda x: np.matmul(j,x), lstConjugateBases)))
        arrAllMatrices = np.concatenate(lstAllMatrices)
        arrAllMatrices = np.unique(np.round(arrAllMatrices,10),axis = 0)
        arrRows = np.where(np.all(np.round(np.linalg.inv(arrAllMatrices),10) == np.round(np.transpose(arrAllMatrices, [0,2,1]),10),axis=1))[0]
        if len(arrRows) > 0:
            arrRows = np.unique(arrRows)
            return arrAllMatrices[arrRows]
        else:
            print('error no conjugates found')
            return []
objCSL = gl.CSLTripleLine(np.array([0,0,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(99)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[5,5,25],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex)
arrCellBasis = objCSL.GetCSLPrimitiveVectors()



#arrCellBasis = objCSL.GetSimulationCellBasis()
objCSLConjugate = CSLConjugateBases(gf.StandardBasisVectors(3))
arrOut = objCSLConjugate.FindConjugates(arrCellBasis)
arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrCellBasis)
arrEdgeVectors = arrEdgeVectors
arrTransform = gf.ConvertToLAMMPSBasis(gf.NormaliseMatrixAlongRows(arrCellBasis))[1]
objSimulationCell = gl.SimulationCell(arrEdgeVectors)
arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors,arrTransform,ld.FCCCell,np.ones(3), np.zeros(3))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
arrPoints = gf.WrapVectorIntoSimulationCell(arrEdgeVectors,arrGrain1.GetAtomPositions())
ax.scatter(*tuple(zip(*arrPoints)))
lstPoints = []
lstPoints.append(arrPoints)
objSimulationCell.AddGrain(arrGrain1)
#objSimulationCell.RemoveAtomsOnOpenBoundaries()
objSimulationCell.WriteLAMMPSDataFile('/home/p17992pt/' + '0.dmp')
objSimulationCell.RemoveAllGrains()
objPTree = gf.PeriodicWrapperKDTree(arrPoints, arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis),50,['p','p','p'])
k = 1
lstTransforms = []
lstTransforms.append(arrTransform)
for i in range(len(arrOut)):
    arrBasis = np.matmul(arrOut[i],arrTransform)
    arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors,arrBasis,ld.FCCCell,np.ones(3),np.zeros(3))
    arrPoints = gf.WrapVectorIntoSimulationCell(arrEdgeVectors,arrGrain1.GetAtomPositions())
    arrDistances, arrIndices = objPTree.Pquery(arrPoints)
    arrDistances = np.array(mf.FlattenList(arrDistances))
    if not(np.all(arrDistances < 1e-5)):
        objSimulationCell.AddGrain(arrGrain1)
        arrPoints = gf.WrapVectorIntoSimulationCell(arrEdgeVectors, arrGrain1.GetAtomPositions())
        lstPoints.append(arrGrain1.GetAtomPositions())
        ax.scatter(*tuple(zip(*arrPoints)))
        objSimulationCell.WrapAllAtomsIntoSimulationCell()
        objSimulationCell.WriteLAMMPSDataFile('/home/p17992pt/' + str(k) + '.dmp')
        objSimulationCell.RemoveAllGrains()
        k +=1
        lstTransforms.append(arrOut[i])
        objPTree = gf.PeriodicWrapperKDTree(np.vstack(lstPoints), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis),50,['p','p','p'])
        
    #print(mf.FlattenList(objPTree.Pquery(lstPoints[-1], k=1)))
plt.show()
arrPoints = np.unique(np.vstack(lstPoints),axis=0)
### Matrix R is either the change of basis or you need to multiply
##the arrCellBasis by all the lstUnitMatrices
#objSimulationCell.GetCoincidentLatticePoints(['1','2'])
print(lstTransforms)
# %%
