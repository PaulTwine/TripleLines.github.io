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
        self.__SigmaValue = 0
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
        if len(arrUniqueLengths) == 2:
            for i in arrUniqueLengths:
                arrRows = np.where(arrLengths == i)[0]
                arrLastRow = list(range(3))
                if len(arrRows) == 2:
                    arrDiagonal = inCSLBasis[arrRows[0]] + inCSLBasis[arrRows[1]]
                    arrBisector = gf.NormaliseVector(arrDiagonal)
                    fltAngle,arrAxis = gf.FindRotationVectorAndAngle(inCSLBasis[arrRows[0]],arrDiagonal)
                    arrMatrix = gf.GetMatrixFromAxisAngle(arrAxis,-fltAngle)
                    lstSwapMatrices.append(arrMatrix)
                    arrLastRow = list(set(range(3)).difference(arrRows))[0]
                    arrPlaneNormal = gf.NormaliseVector(np.cross(arrBisector, gf.NormaliseVector(inCSLBasis[arrLastRow])))
                    arrReflection = gf.FindReflectionMatrix(arrPlaneNormal)
                    lstSwapMatrices.append(arrReflection)
                    fltAngle,arrAxis = gf.FindRotationVectorAndAngle(inCSLBasis[arrRows[0]],inCSLBasis[arrRows[1]])
                    arrRotation = gf.GetMatrixFromAxisAngle(arrAxis,fltAngle)
                    lstSwapMatrices.append(arrRotation)
                    arrPerpendicular = gf.NormaliseVector(np.cross(arrBisector, inCSLBasis[arrLastRow]))
                    arrRotation = gf.GetMatrixFromAxisAngle(arrPerpendicular,np.pi)
                    lstSwapMatrices.append(arrRotation)
       # elif len(arrUniqueLengths) == 1:
      #      lstRows = list(itertools.permutations(list(range(3))))
      #      for k in lstRows:
      #          lstSwapMatrices.append(arrStandardBasis[list(k)])
        else:
            lstSwapMatrices.append(arrStandardBasis)
        lstRows = list(itertools.permutations(list(range(3))))
        for k in lstRows:
            lstSwapMatrices.append(arrStandardBasis[list(k)])
        return lstSwapMatrices
    def FindDiagonalCSL(self, inCSLBasis):
        arrLengths = np.linalg.norm(inCSLBasis,axis=1)
        arrUniqueLengths,arrCounts = np.unique(arrLengths, return_counts=True)
        lstCSLBases = []
        lstCSLBases.append(inCSLBasis)
        lstVectors = []
        if len(arrUniqueLengths) ==2:
            intRow = np.where(arrCounts ==2)[0]
            arrRows = np.where(arrLengths == arrUniqueLengths[intRow])[0]
            lstVectors.append(inCSLBasis[arrRows[0]]+inCSLBasis[arrRows[1]])
            lstVectors.append(inCSLBasis[arrRows[0]]-inCSLBasis[arrRows[1]])
            arrLastRow = list(set(range(3)).difference(arrRows.tolist()))[0]
            lstVectors.append(inCSLBasis[arrLastRow])
            arrReturn = np.vstack(lstVectors)
            lstCSLBases.append(arrReturn)
        return lstCSLBases
    def DerivedCSL(self, inCSLBasis):
        lstAllVectors = []
        arrVectors = inCSLBasis
        for i in range(len(arrVectors)):
            lstAllVectors.append(arrVectors[i])
            for j in range(i,len(arrVectors)):
                lstAllVectors.append(arrVectors[i]+arrVectors[j])
                lstAllVectors.append(arrVectors[i]-arrVectors[j])
                for k in range(j,3):
                    lstAllVectors.append(arrVectors[i]+arrVectors[j]+arrVectors[k])
                    lstAllVectors.append(arrVectors[i]-arrVectors[j]+arrVectors[k])
                    lstAllVectors.append(arrVectors[i]+arrVectors[j]-arrVectors[k])
                    lstAllVectors.append(arrVectors[i]-arrVectors[j]-arrVectors[k])
        arrVectors = np.unique(lstAllVectors,axis=0)
        lstMatrices = []
        for i in range(len(arrVectors)):
            for j in range(i,len(arrVectors)):
                for k in range(k,len(arrVectors)):
                    lstTemp = []
                    lstTemp.append(arrVectors[i])
                    lstTemp.append(arrVectors[j])
                    lstTemp.append(arrVectors[k])
                    arrMatrix = np.vstack(lstTemp)
                    if np.round(np.linalg.det(arrMatrix),10) !=0:
                        lstMatrices.append(np.vstack(lstTemp))
        return lstMatrices
    def FindSpecialSymmetries(self, inCSLBasis):
        lstDerived = []
        lstDerived.append(inCSLBasis)
        lstRows = list(range(3))
        lstMatrix = []
        for j in range(len(inCSLBasis)):
            arrTemp = np.zeros([3,3])
            lstRows.remove(j)
            arrTemp[j] = inCSLBasis[j]
            arrTemp[lstRows[0]] = inCSLBasis[lstRows[0]] + inCSLBasis[lstRows[1]]
            arrTemp[lstRows[1]] = inCSLBasis[lstRows[0]] - inCSLBasis[lstRows[1]]
            arrMatrix =np.copy(arrTemp)
            lstDerived.append(arrMatrix)
            lstMatrix = []
            lstRows = list(range(3))    
        lstSpecial = []           
        return lstDerived
    def FindSigmaFactors(self, inCSLBasis):
        lstSigmaObjects = []
        for j in inCSLBasis:
            objSigma = gl.SigmaCell(2*j,ld.FCCCell)
            lstSigmaObjects.append(objSigma)
        lstFactors = []
        for f in range(2,self.__SigmaValue+1):
            if self.__SigmaValue % f == 0:
                lstFactors.append(f)
        lstTransforms = []
        for obj in lstSigmaObjects:
            arrSigmaValues = obj.GetSigmaValues(50,True)
            for f in lstFactors:
                if f in arrSigmaValues[:,0]:
                    #obj.MakeCSLCell(f, False)
                    lstCSLs,lstBases = obj.GetAllCSLPrimitiveVectors(f)
                    #lstCSLs = [obj.GetCSLPrimitiveVectors()]
                    #lstBases = [obj.GetOriginalBasis()]
                    for c in range(len(lstCSLs)):
                        arrCSL = lstCSLs[c]
                        arrR = np.matmul(inCSLBasis,np.linalg.inv(arrCSL))
                        arrP = np.matmul(arrR,lstBases[c])
                    #arrRP = np.matmul(arrR,np.linalg.inv(ld.FCCPrimitive))
                        if np.all(np.round(arrR,0) == np.round(arrR,10)):
                            lstTransforms.append(lstBases[c])
                       # elif np.all(np.round(arrP,0) == np.round(arrP,10)):
                        #    lstTransforms.append(lstBases[c])
        print(lstTransforms)
        return lstTransforms
    def FindConjugates(self, inCSLBasis):
       # inPBasis = np.matmul(inCSLBasis,np.linalg.inv(ld.FCCPrimitive))
       # arrUnitBasis = gf.ConvertToLAMMPSBasis(inCSLBasis)[1]
       # lstCSLBases = self.DerivedCSL(inCSLBasis)
        self.__SigmaValue = int(np.round(np.abs(np.linalg.det(inCSLBasis)/np.linalg.det(ld.FCCPrimitive))))
        lstSigmaMatrices = self.FindSigmaFactors(inCSLBasis)
        arrStandardBasis = gf.StandardBasisVectors(3)
        #lstCSLBases=self.FindSpecialSymmetries(inCSLBasis)
        lstCSLBases = [arrStandardBasis]
        #for j in lstCSLBases:
        #    lstSigmaMatrices.extend(self.#FindSigmaFactors(inCSLBasis))
        #lstCSLBases = [arrStandardBasis, arrStandardBasis[[1,0,2]], arrStandardBasis[[0,2,1]], arrStandardBasis[[2,0,1]],arrStandardBasis[[2,0,1]], arrStandardBasis[[1,2,0]]]
        #lstSwapMatrices.append(arrStandardBasis)
        lstUnitMatrices,lstTranslations = self.DiagonalUnitEigenMatrices(inCSLBasis)
         
        lstConjugateBases = []
        for j in lstCSLBases:
            lstConjugateBases.extend(list(map(lambda x: np.matmul(np.matmul(np.transpose(j), x),np.transpose(np.linalg.inv(j))),lstUnitMatrices)))
        # for j in lstSwapCSLMatrices:
        #     lstAllMatrices.extend(list(map(lambda x: np.matmul(j,x),lstConjugateBases)))
        arrAllMatrices = np.round(np.unique(lstConjugateBases,axis = 0),10)
        arrPBasis = np.matmul(arrStandardBasis,ld.FCCPrimitive)
        lstAllMatrices = []
        # for j in lstCSLBases:
        #     l = np.matmul(j, ld.FCCPrimitive)
        #     T = np.matmul(arrPBasis,np.linalg.inv(l))
        #     l = np.linalg.norm(j,axis=1)
        #     if np.all(np.round(l,0) ==  l):
        #         #k = np.matmul(gf.NormaliseMatrixAlongRows(j),np.linalg.inv(gf.NormaliseMatrixAlongRows(lstCSLBases[0])))
        #         lstAllMatrices.append(gf.NormaliseMatrixAlongRows(j))
        # # #for i in arrAllMatrices:
        # #      lstAllMatrices.extend(list(map(lambda x: np.matmul(x,i), arrAllMatrices)))
            #  lstAllMatrices.extend(list(map(lambda# x: np.matmul(x,np.linalg.inv(i)), arrAllMatrices)))
        lstAllMatrices.extend(lstConjugateBases)
        lstAllMatrices.extend(lstSigmaMatrices)     
        arrAllMatrices = np.round(np.unique(lstAllMatrices,axis = 0),10)
        
        #### new section
        # arrT = np.array(list(map(lambda x: np.matmul(np.linalg.qr(x)[1],np.linalg.inv(ld.FCCPrimitive)), arrAllMatrices)))
        # arrRows = np.where(np.all(np.round(arrT,10) == np.round(arrT,0), axis=1))[0]

        arrTranspose= np.array(list(map(lambda x: np.transpose(x),arrAllMatrices)))
        arrInverse= np.array(list(map(lambda x: np.linalg.inv(x),arrAllMatrices)))
        arrDifference = np.array(list(map(lambda x: np.max(x), np.abs(arrTranspose-arrInverse))))
        arrRows = np.where(arrDifference  < 1e-10)[0]
        ##arrRows = np.where(np.all(np.round(np.transpose(arrAllMatrices, [0,2,1]),10) == np.round(arrAllMatrices,10),axis=1))[0]


       
       
        # arrScaling = np.array(list(map(lambda x: np.sum(np.abs((np.linalg.qr(np.transpose(x))[1] - np.round(np.linalg.qr(np.transpose(x))[1],0)))), arrAllMatrices)))
        # arrOrthogonal = np.array(list(map(lambda x: np.linalg.qr(np.transpose(x))[0], arrAllMatrices)))
        # arrRows = np.where(arrScaling < 1e-10)[0]
        if len(arrRows) > 0:
            arrRows = np.unique(arrRows)
            return arrAllMatrices[arrRows]
        else:
            print('error no conjugates found')
            return []
objCSL = gl.CSLTripleLine(np.array([1,0,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(100)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[3,9,27],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,False)
arrCellBasis = objCSL.GetCSLPrimitiveVectors()
objSigmaCell = gl.SigmaCell(2*arrCellBasis[0],ld.FCCCell)
objCSLConjugate = CSLConjugateBases(gf.StandardBasisVectors(3))
arrOut = objCSLConjugate.FindConjugates(arrCellBasis)

arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrCellBasis)
objSimulationCell = gl.SimulationCell(arrEdgeVectors)
arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors,arrTransform,ld.FCCCell,np.ones(3), np.zeros(3))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*tuple(zip(*arrGrain1.GetAtomPositions())))
lstPoints = []
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.RemoveAtomsOutsideSimulationCell()
objSimulationCell.WriteLAMMPSDataFile('/home/p17992pt/' + '0.dmp')
objSimulationCell.RemoveAllGrains()
arrPoints = arrGrain1.GetAtomPositions()
lstPoints.append(arrPoints)
objPTree = gf.PeriodicWrapperKDTree(arrGrain1.GetAtomPositions(), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis),50,['p','p','p'])
intTransform = 0
lstTransforms = []
lstTransforms.append(gf.StandardBasisVectors(3))
for i in range(len(arrOut)):
    arrBasis = np.matmul(arrOut[i],arrTransform)
    arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors,arrBasis,ld.FCCCell,np.ones(3),np.zeros(3))
    arrPoints = gf.WrapVectorIntoSimulationCell(arrEdgeVectors,arrGrain1.GetAtomPositions())
    arrDistances,arrIndices = objPTree.Pquery(arrPoints)
    arrDistances = np.array(mf.FlattenList(arrDistances))
    if not(np.all(arrDistances < 1e-5)):
        objSimulationCell.AddGrain(arrGrain1)
        #arrPoints = arrGrain1.GetAtomPositions()
        #lstPoints.append(arrGrain1.GetAtomPositions())
        lstPoints.append(arrPoints)
        objSimulationCell.WrapAllAtomsIntoSimulationCell()
        ax.scatter(*tuple(zip(*lstPoints[-1])))
        objSimulationCell.WriteLAMMPSDataFile('/home/p17992pt/' + str(intTransform+1) + '.dmp')
        objSimulationCell.RemoveAllGrains()
        objPTree = gf.PeriodicWrapperKDTree(np.vstack(lstPoints), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis),50,['p','p','p'])
        lstTransforms.append(arrOut[i])
        intTransform +=1     
        
    #print(mf.FlattenList(objPTree.Pquery(lstPoints[-1], k=1)))
plt.show()
arrPoints = np.unique(np.vstack(lstPoints),axis=0)
### Matrix R is either the change of basis or you need to multiply
##the arrCellBasis by all the lstUnitMatrices
#objSimulationCell.GetCoincidentLatticePoints(['1','2'])
print(np.round(27*np.array(lstTransforms),5))
# %%
