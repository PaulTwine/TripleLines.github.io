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
    def GetCellSigmaValue(self):
        return self.__SigmaValue
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
    def FindSigmaFactors(self, inCSLBasis, blnUniqueSigma = True):
        lstSigmaObjects = []
        lstDerived = self.DerivedCSL(inCSLBasis)
        arrDerived = np.vstack(lstDerived)
        arrVectors = np.unique(arrDerived, axis=0)
        lstReducedVectors = list(map(lambda x: x/np.gcd.reduce(2*x.astype('int')),arrVectors))
        lstReducedVectors = np.unique(lstReducedVectors,axis=0)
        for j in lstReducedVectors:
            objSigma = gl.SigmaCell(2*j,ld.FCCCell)
            lstSigmaObjects.append(objSigma)
        lstFactors = []
        for f in range(2,self.__SigmaValue+1):
            if self.__SigmaValue % f == 0:
                lstFactors.append(f)
        lstTransforms = []
        # for f in lstFactors:
        #     arrAxis = gf.FindAxesFromSigmaValues(f,10)
        #     #arrAxis = inCSLBasis
        #     for a in arrAxis:
        #         objSigma = gl.SigmaCell(2*a,ld.FCCCell)
        #         objSigma.MakeCSLCell(f,False)
        #         arrBasis = objSigma.GetOriginalBasis()
        #         arrCSL = objSigma.GetCSLPrimitiveVectors()
        #         arrR = np.matmul(inCSLBasis, np.linalg.inv(arrCSL))/f
        #         if np.all(np.round(arrR,0) == np.round(arrR,10)):
        #             lstTransforms.append(arrBasis)
        for obj in lstSigmaObjects:
            arrSigmaValues = obj.GetSigmaValues(200,True)
            for f in lstFactors:
                if f in arrSigmaValues[:,0]:
                    lstCSLs,lstBases = obj.GetAllCSLPrimitiveVectors(f)
                    blnFound = False
                    c = 0
                    while c < len(lstCSLs) and not(blnFound):
                    #for c in range(len(lstCSLs)):
                        arrCSL = lstCSLs[c]
                        #lstDerviedCSLs = self.DerivedCSL(lstCSLs[c])
                        #for arrCSL in lstDerviedCSLs:
                        arrR = np.matmul(inCSLBasis,np.linalg.inv(arrCSL))
                        if np.all(np.round(2*arrR,0) == np.round(2*arrR,10)):
                            lstTransforms.append(lstBases[c])
                            if blnUniqueSigma:
                                blnFound = True
                        c += 1
        return lstTransforms
    def FindConjugates(self, inCSLBasis):
        self.__SigmaValue = int(np.round(np.abs(np.linalg.det(inCSLBasis)/np.linalg.det(ld.FCCPrimitive))))
        #lstSigmaMatrices = self.FindSigmaFactors(inCSLBasis)
        arrStandardBasis = gf.StandardBasisVectors(3)
        lstCSLBases = self.DerivedCSL(inCSLBasis)
        #for j in lstCSLBases:
        #    lstSigmaMatrices.extend(self.#FindSigmaFactors(inCSLBasis))
        #lstCSLBases = [arrStandardBasis, arrStandardBasis[[1,0,2]], arrStandardBasis[[0,2,1]], arrStandardBasis[[2,0,1]],arrStandardBasis[[2,0,1]], arrStandardBasis[[1,2,0]]]
        #lstSwapMatrices.append(arrStandardBasis)
        lstUnitMatrices,lstTranslations = self.DiagonalUnitEigenMatrices(inCSLBasis)
        lstConjugateBases = []
        lstSigmaMatrices = self.FindSigmaFactors(inCSLBasis)
        # for i in lstCSLBases:
        #     lstUnitMatrices.extend(self.FindSigmaFactors(i))
        for j in lstCSLBases:
            lstConjugateBases.extend(list(map(lambda x: np.matmul(np.matmul(np.transpose(j), x),np.transpose(np.linalg.inv(j))),lstUnitMatrices)))
        lstAllMatrices = []
        lstAllMatrices.extend(lstConjugateBases)
        lstAllMatrices.extend(lstSigmaMatrices)     
        arrAllMatrices = np.round(np.unique(lstAllMatrices,axis = 0),10)
        arrTranspose= np.array(list(map(lambda x: np.transpose(x),arrAllMatrices)))
        arrInverse= np.array(list(map(lambda x: np.linalg.inv(x),arrAllMatrices)))
        arrDifference = np.array(list(map(lambda x: np.max(x), np.abs(arrTranspose-arrInverse))))
        arrRows = np.where(arrDifference  < 1e-10)[0]
        if len(arrRows) > 0:
            arrRows = np.unique(arrRows)
            return arrAllMatrices[arrRows]
        else:
            print('error no conjugates found')
            return []
objCSL = gl.CSLTripleLine(np.array([1,1,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[21,21,49],axis=1))[0][0]
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
#lstSigma = list(map(lambda x: np.gcd.reduce(np.unique(x)),lstTransforms))
lstSigma = list(map(lambda x: objCSLConjugate.GetCellSigmaValue()/np.gcd.reduce(np.round(np.unique(objCSLConjugate.GetCellSigmaValue()*x,0)).astype('int')),lstTransforms))
print(np.round(objCSLConjugate.GetCellSigmaValue()*np.array(lstTransforms),5),lstSigma)
# %%
