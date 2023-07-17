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
        self.__SigmaFactors = []
        self.__Axes = []
        self.__CSLBasis = []
        self.__Transforms = []
    def GetCellSigmaValue(self):
        return self.__SigmaValue
    def FindTransforms(self,inCSLBasis):
        self.__SigmaValue = int(np.round(np.abs(np.linalg.det(inCSLBasis)/np.linalg.det(ld.FCCPrimitive))))
        self.__CSLBasis = inCSLBasis
        lstFactors = mf.Factorize(self.GetCellSigmaValue())
        lstFactors.remove(1) 
        self.__SigmaFactors = lstFactors 
        arrStandardBasis = gf.StandardBasisVectors(3)
        self.FindRotationAxesByIteration([2*inCSLBasis],[arrStandardBasis])
        arrTransforms = np.array(self.__Transforms)
        arrAxes = np.array(self.__Axes)
        if len(self.__Transforms) > 0:
            #arrAxes = np.array(lstAxes)
            #arrMatrices = np.array(lstMatrices)
            arrLengths = np.linalg.norm(arrAxes, axis=1)
            arrRows = np.argsort(arrLengths)
            return arrTransforms[arrRows], arrAxes[arrRows]
        else:
            return [], []
    def FindRotationAxesByIteration(self,lstCurrentAxes: np.array,lstPreviousTransform: np.array):
        lstNewTransforms =[]
        lstNewAxes = []
        intL = len(lstPreviousTransform) 
        for n in range(intL):
            arrAxes = lstCurrentAxes[n]
            arrPreviousTransform = lstPreviousTransform[n]
            for i in arrAxes:
                i = i/np.gcd.reduce(i.astype('int'))
                if i[0] < 0:
                    i = -i
                arrCSL = gf.CubicCSLGenerator(i.astype('int'),300)
                for l in self.__SigmaFactors:
                    arrRows = np.where(arrCSL[:,0].astype('int')==l)[0]
                    if len(arrRows) > 0:
                        arrAllAxes = np.unique(list(it.permutations(i,3)),axis=0)
                        for a in arrRows:
                            for j in arrAllAxes:
                                if j[0] < 0:
                                    j = -j
                                fltAngle = arrCSL[a,1]
                                arrMatrix = gf.GetMatrixFromAxisAngle(j,fltAngle)
                                arrMatrix = np.matmul(arrMatrix, arrPreviousTransform)
                                arrR = 2*np.matmul(self.__CSLBasis, np.linalg.inv(arrMatrix))
                                if np.all(np.round(arrR,0) == np.round(arrR,10)):                  
                                    blnAppend = False
                                    self.__Transforms.append(arrMatrix)
                                    self.__Axes.append(j)
                                    lstTempAxes = []
                                    for k in arrR.astype('int'):
                                        if k[0] < 0:
                                            k = - k
                                        k = k/np.gcd.reduce(k.astype('int'))
                                        if not(np.all(np.isin(k,self.__Axes))):
                                            lstTempAxes.append(k)
                                            blnAppend = True
                                    if blnAppend:
                                        arrNewAxes = np.vstack(lstTempAxes)
                                        lstNewAxes.append(arrNewAxes)
                                        lstNewTransforms.append(arrMatrix)
        self.__Axes = list(np.unique(self.__Axes,axis=0))
        self.__Transforms = list(np.unique(self.__Transforms,axis=0))
        if len(lstNewAxes) > 0:
            self.FindRotationAxesByIteration(lstNewAxes, lstNewTransforms)
    def FindRotationAxes(self, arrAxes,intLimt = 20):
        lstTransforms =[]
        lstNewAxes = []
        arrAxes = arrAxes.astype('int')
        intCount = 0
        lstFactors = mf.Factorize(self.GetCellSigmaValue())
        lstFactors.remove(1) 
        for i in arrAxes:
            arrCSL = gf.CubicCSLGenerator(i.astype('int'),5)
            for l in lstFactors:
                arrRows = np.where(arrCSL[:,0].astype('int')==l)[0]
                if len(arrRows) > 0:
                    arrAllAxes = np.unique(list(it.permutations(i,3)),axis=0)
                    for a in arrRows:
                        for j in arrAllAxes:
                            fltAngle = arrCSL[a,1]
                            arrMatrix = gf.GetMatrixFromAxisAngle(i,fltAngle)
                            arrR = 2*np.matmul(self.__CSLBasis, np.linalg.inv(arrMatrix))
                            if np.all(np.round(arrR,0) == np.round(arrR,10)):
                                self.__Transforms.append(arrMatrix)
                                j = j/np.gcd.reduce(j.astype('int'))
                                self.__Axes.append(j)
                                for k in arrR.astype('int'):
                                    k = k/np.gcd.reduce(k.astype('int'))
                                    if not(np.all(np.isin(k,self.__Axes))):
                                        lstNewAxes.append(k)
        if len(lstNewAxes) > 0:
            self.FindRotationAxes(np.unique(np.vstack(lstNewAxes),axis=0))
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
    def FindOrthogonalReciprocalVectors(self,inCSLBasis):
        lstTransforms = []
        arrR = np.transpose(gf.FindReciprocalVectors(np.transpose(inCSLBasis)))
        lstVectors = []
        for a in range(-10,10):
            for b in range(-10,10):
                for c in range(-10,10):
                    lstVectors.append(a*arrR[0]+b*arrR[1]+c*arrR[2])
        arrVectors = np.vstack(lstVectors)
        arrLengths = np.linalg.norm(arrVectors, axis=1)
        arrRows = np.where(np.round(arrLengths,10) ==1)[0]
        arrUnitVectors = arrVectors[arrRows]
        arrRows2,arrCols2 = np.where(np.abs(np.matmul(arrUnitVectors,np.transpose(arrUnitVectors))) < 1e-10)
        for n in arrRows2:
            arrCheck = np.where(arrRows2 == n)[0]
            arrVectors = arrUnitVectors[arrCols2][arrCheck]
            arrRows3,arrCols3 = np.where(np.abs(np.matmul(arrVectors,np.transpose(arrVectors))) < 1e-10)
            for i in range(len(arrRows3)):
                arrMatrix = np.zeros([3,3])
                if arrRows3[i] < arrCols3[i]: #check no double counting
                    arrMatrix[0]= arrUnitVectors[arrRows2][n]
                    arrMatrix[1] = arrVectors[arrRows3][i]
                    arrMatrix[2] = arrVectors[arrCols3][i]
                    lstTransforms.append(arrMatrix)
        return lstTransforms
      
#%%            
objCSL = gl.CSLTripleLine(np.array([1,1,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[7,7,49],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,False)
arrCellBasis = objCSL.GetCSLPrimitiveVectors()
print(arrCellBasis)
# objSigma = gl.SigmaCell(np.array([5,4,2]),ld.FCCCell)
# objSigma.MakeCSLCell(243,False)
# arrCellBasis = objSigma.GetCSLPrimitiveVectors()

gf.PrimitiveToOrthogonalVectorsGrammSchmdit(arrCellBasis,ld.FCCPrimitive)
arrR = gf.FindReciprocalVectors(np.transpose(arrCellBasis))

objCSLConjugate = CSLConjugateBases(gf.StandardBasisVectors(3))
lstNewT = objCSLConjugate.FindOrthogonalReciprocalVectors(2*arrCellBasis)
print(np.array(lstNewT)*49)
#arrOut,arrAxes = objCSLConjugate.FindTransforms(arrCellBasis)
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
#print(np.matmul(arrCellBasis,np.linalg.inv(np.array(lstTransforms)),axis=1))
# %%
