#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it

#%%
class CSLConjugateBases(object):
    def __init__(self,inBasis: np.array):
        self.__Basis = inBasis
        self.__SigmaValue = 0
    def GetCellSigmaValue(self):
        return self.__SigmaValue
    def FindPossibleAxes(self,inCSLBasis):
        lstTransforms = []
        lstAxes = []
        lstFactors = mf.Factorize(self.__SigmaValue)
        lstFactors.remove(1)
        invCSLBasis = np.linalg.inv(inCSLBasis)
        lstUsed = []
        arrBox = gf.FindBoundingBox(inCSLBasis).astype('int')
        intMax = np.max(np.abs(arrBox))
        for s in lstFactors:
            arrAxes = gf.FindAxesFromSigmaValues(s, 5)
            l = list(map(lambda x: np.unique(list(it.permutations(x)),axis=0),arrAxes))
            arrAxes = np.concatenate(l,axis=0)
            t = self.__SigmaValue/s
            arrCAxes = np.matmul(arrAxes, invCSLBasis)*t
            arrRows = np.where(np.all(np.round(arrCAxes,0) == np.round(arrCAxes,10),axis=1))[0]
            if len(arrRows) > 0:
                arrTest = arrAxes[arrRows]
                for i in arrTest:
                    objSigma = gl.SigmaCell(i.astype('int'),ld.FCCCell)
                    arrSigmaValues = objSigma.GetSigmaValues(10)
                    arrRows2 = np.where(arrSigmaValues[:,0] ==s)[0]
                    for a in arrRows2:
                        fltAngle = arrSigmaValues[a,1]
                        arrMatrix = gf.GetMatrixFromAxisAngle(i,fltAngle)
                        arrR = np.matmul(inCSLBasis, np.linalg.inv(arrMatrix))
                        if np.all(np.round(2*arrR,0) == np.round(2*arrR,10)):
                                lstTransforms.append(arrMatrix)
                                lstAxes.append(i)
        return lstTransforms, lstAxes                       
    def FindConjugates(self, inCSLBasis):
        self.__SigmaValue = int(np.round(np.abs(np.linalg.det(inCSLBasis)/np.linalg.det(ld.FCCPrimitive))))
        arrStandardBasis = gf.StandardBasisVectors(3)
       # lstUnitMatrices,lstTranslations = self.DiagonalUnitEigenMatrices(inCSLBasis)
        #lstConjugateBases = []
        lstMatrices, lstAxes = self.FindPossibleAxes(inCSLBasis)
        if len(lstAxes) > 0:
            arrAxes = np.array(lstAxes)
            arrMatrices = np.array(lstMatrices)
            arrLengths = np.linalg.norm(arrAxes, axis=1)
            arrRows = np.argsort(arrLengths)
            return arrMatrices[arrRows], arrAxes[arrRows]
        else:
            return [], []

        # arrTranspose= np.array(list(map(lambda x: np.transpose(x),lstMatrices)))
        # arrInverse= np.array(list(map(lambda x: np.linalg.inv(x),lstMatrices)))
        # arrDifference = np.array(list(map(lambda x: np.max(x), np.abs(arrTranspose-arrInverse))))
        # arrRows = np.where(arrDifference  < 1e-10)[0]
        # if len(arrRows) > 0:
        #     arrRows = np.unique(arrRows)
        #     return np.array(lstMatrices)[arrRows],np.array(lstAxes)[arrRows]
        # else:
        #     print('error no conjugates found')
        #     return [],[]
            
#%%            
objCSL = gl.CSLTripleLine(np.array([5,1,0]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[15,21,35],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,False)
arrCellBasis = objCSL.GetCSLPrimitiveVectors()
#gf.PrimitiveToOrthogonalVectorsGrammSchmdit(arrCellBasis,ld.FCCPrimitive)
arrR = gf.FindReciprocalVectors(arrCellBasis)

objCSLConjugate = CSLConjugateBases(gf.StandardBasisVectors(3))
arrOut,arrAxes = objCSLConjugate.FindConjugates(arrCellBasis)
a = 4.05
#arrCellBasis = gf.PrimitiveToOrthogonalVectorsGrammSchmdit(arrCellBasis, ld.FCCPrimitive)
arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrCellBasis)
objSimulationCell = gl.SimulationCell(arrEdgeVectors)
arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors,arrTransform,ld.FCCCell,np.ones(3), np.zeros(3))

# lstAll = []
# for i in arrOut:
#     lstTemp = list(map(lambda x: np.matmul(x,i), arrOut))
#     lstAll.extend(lstTemp)
# arrOut = np.array(lstAll)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*tuple(zip(*arrGrain1.GetAtomPositions())))
lstPoints = []
objSimulationCell.AddGrain(arrGrain1)
objSimulationCell.RemoveAtomsOnOpenBoundaries()
objSimulationCell.WriteLAMMPSDataFile('/home/p17992pt/' + '0.dmp')
objSimulationCell.RemoveAllGrains()
arrPoints = arrGrain1.GetAtomPositions()
lstPoints.append(arrPoints)
objPTree = gf.PeriodicWrapperKDTree(arrGrain1.GetAtomPositions(), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis),50,['p','p','p'])
intTransform = 0
lstTransforms = []
lstTransforms.append(gf.StandardBasisVectors(3))
lstAxes = []
for i in range(len(arrOut)):
    arrBasis = np.matmul(arrOut[i],arrTransform)
    arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors,arrBasis,ld.FCCCell,np.ones(3),np.zeros(3))
    arrPoints = gf.WrapVectorIntoSimulationCell(arrEdgeVectors,arrGrain1.GetAtomPositions())
    arrPoints = objSimulationCell.RemoveRealDuplicates(arrPoints,1e-5)
    arrDistances,arrIndices = objPTree.Pquery(arrPoints,k=1)
    arrDistances = np.array(mf.FlattenList(arrDistances))
    if not(np.all(arrDistances < 1e-5)):
        objSimulationCell.AddGrain(arrGrain1)
        objSimulationCell.RemoveGrainPeriodicDuplicates()
        lstPoints.append(arrPoints)
        lstAxes.append(arrAxes[i])
        objSimulationCell.RemoveAtomsOnOpenBoundaries()
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
print(np.round(objCSLConjugate.GetCellSigmaValue()*np.array(lstTransforms),5),lstSigma, lstAxes)
# %%
