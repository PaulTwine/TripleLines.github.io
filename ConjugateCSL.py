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
                    arrPrimitive = np.matmul(arrMatrix, np.linalg.inv(ld.FCCPrimitive))
                    if np.all(np.round(arrPrimitive,10) == np.round(arrPrimitive,0)):
                        lstMatrices.append(np.vstack(lstTemp))
        return lstMatrices    
    def FindSigmaFactors(self, inCSLBasis: np.array):
        # lstDerived = self.DerivedCSL(inCSLBasis)
        # arrDerived = np.vstack(lstDerived)
        # arrVectors = np.unique(2*arrDerived, axis=0).astype('int')
        # lstReducedVectors = list(map(lambda x: x/np.gcd.reduce(x.astype('int')),arrVectors))
        #lstReducedVectors = np.unique(lstReducedVectors,axis=0)
        arrReducedVectors = self.FindPossibleAxes(inCSLBasis)
        lstReducedVectors = list(map(lambda x: x/np.gcd.reduce(x.astype('int')),2*arrReducedVectors))
        arrReducedVectors = np.unique(np.array(lstReducedVectors), axis=0)
        intSigma = int(np.abs(np.round(np.linalg.det(inCSLBasis)/np.linalg.det(ld.FCCPrimitive))))
        lstFactors = mf.Factorize(intSigma)
        setFactors = set(lstFactors)
        lstTransforms = []
        lstR = []
        arrRows = np.where(np.linalg.norm(arrReducedVectors, axis=1)**2 <= 4*intSigma)[0]
        arrReducedVectors = arrReducedVectors[arrRows]
        for j in arrReducedVectors:
            objSigma = gl.SigmaCell(j,ld.FCCCell)
            arrSigmaValues = objSigma.GetSigmaValues(200,False)
            lstSigmaValues = arrSigmaValues[:,0].astype('int').tolist()
            lstOverlap = list(setFactors.intersection(lstSigmaValues))
            for f in lstOverlap:
                arrRows = np.where(arrSigmaValues[:,0] == f)[0]
                for a in arrRows:
                    fltAngle = arrSigmaValues[a,1]
                    arrMatrix = gf.GetMatrixFromAxisAngle(j,fltAngle)
                    arrR = np.matmul(inCSLBasis, np.linalg.inv(arrMatrix))
                    if np.all(np.round(2*arrR,0) == np.round(2*arrR,10)):
                         lstTransforms.append(arrMatrix)
                         lstR.append(arrR)
            for i in [1,-1]:
                arrMatrix = gf.GetMatrixFromAxisAngle(j, i*np.pi)
                arrR = np.matmul(inCSLBasis, np.linalg.inv(arrMatrix))
                if np.all(np.round(2*arrR,0) == np.round(2*arrR,10)):
                    lstTransforms.append(arrMatrix)
                    lstR.append(arrR)
        return lstTransforms, lstR
    def FindReducedBases(self, inCSLBasis: np.array):
        lstAllAxes = []
        for i in inCSLBasis:
            intGCD = np.gcd.reduce((2*i).astype('int')) ##doubled as primitive cells have 0.5 s for nearest neighbour positions in cubics
            lstNumbers = mf.Factorize(int(intGCD))
            lstScaledAxes = []
            for j in lstNumbers:
                lstScaledAxes.append(i/j)
            lstAllAxes.append(np.vstack(lstScaledAxes))
        lstAllBases = []
        for i in lstAllAxes[0]:
            arrMatrix = np.zeros([3,3])
            arrMatrix[0] = i
            for j in lstAllAxes[1]:
                arrMatrix[1] = j 
                for k in lstAllAxes[2]:
                    arrMatrix[2] = k
                    arrPrimitive = np.matmul(inCSLBasis, np.linalg.inv(ld.FCCPrimitive))
                    if np.all(np.round(arrPrimitive,10) == np.round(arrPrimitive,0)):
                        lstAllBases.append(np.copy(arrMatrix))
        return lstAllBases
    def FindPossibleAxes(self, inCSLBasis):
        lstAllAxes = []
        for i in range(3):
            lstAllAxes.append(inCSLBasis[i])
            for j in range(i+1,3):
                arrVectors = np.vstack(lstAllAxes)
                lstAllAxes.append(arrVectors + inCSLBasis[j])
                lstAllAxes.append(arrVectors - inCSLBasis[j])
                for k in range(i+2,3):
                    arrVectors = np.vstack(lstAllAxes)
                    lstAllAxes.append(arrVectors + inCSLBasis[k])
                    lstAllAxes.append(arrVectors - inCSLBasis[k])
        return np.unique(np.vstack(lstAllAxes),axis=0)
                        
    def FindConjugates(self, inCSLBasis):
        self.__SigmaValue = int(np.round(np.abs(np.linalg.det(inCSLBasis)/np.linalg.det(ld.FCCPrimitive))))
        #lstSigmaMatrices = self.FindSigmaFactors(inCSLBasis)
        #self.FindPossibleAxes(inCSLBasis)
        arrStandardBasis = gf.StandardBasisVectors(3)
      
        lstUnitMatrices,lstTranslations = self.DiagonalUnitEigenMatrices(inCSLBasis)
        lstConjugateBases = []
        #arrReducedVectors = self.FindPossibleAxes(inCSLBasis)
        lstBases = self.FindReducedBases(inCSLBasis)
        # lstBases = []
        # for i in lstReducedBases:
        #     lstBases.extend(self.DerivedCSL(i))
        lstSigmaMatrices = []
        lstIntegerMatrix =[]
        for j in lstBases:
            lstT,lstR = self.FindSigmaFactors(j)
            lstSigmaMatrices.extend(lstT)
            lstIntegerMatrix.extend(lstR)
        lstExtraTransforms = []
        for t in lstSigmaMatrices:
            lstRs = list(map(lambda x: np.matmul(x,np.linalg.inv(t)),lstIntegerMatrix))
            lstRows = list(map(lambda x: np.all(np.equal(np.mod(np.round(2*x,10),1),0)),lstRs))
            #arrRs = np.array(lstRs)
            arrRows = np.where(np.array(lstRows))[0]
            arrRows = np.unique(arrRows)
            if len(arrRows) > 0:
                arrTransforms = np.matmul(t,np.array(lstSigmaMatrices)[arrRows])
                lstExtraTransforms.extend(arrTransforms)
        lstSigmaMatrices.extend(lstExtraTransforms)
        for j in lstBases:
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
#%%
def FindPossibleAxes(inCSLBasis, lstFactors):
    intMax = np.max(lstFactors)
    lstPermutations = it.combinations([0,1,2],3)
    lstUsedFactors = []
    # for i in range(-intMax,intMax,1):
    #     for j in range(-intMax,intMax,1):
    #         for k in range(-intMax,intMax):
    #             arrAxis = i*inCSLBasis[0] + j*inCSLBasis[1]+k*inCSLBasis[2]
    #             if np.all(np.mod(arrAxis, intMax*np.ones(3)) == np.zeros(3)):  
    #                 print(arrAxis/np.gcd.reduce(arrAxis.astype('int')), intMax)
    for l in lstPermutations:
        dctAxes = dict()
        for i in range(-intMax,intMax,1):
            for j in range(-intMax,intMax,1):
                arrAxis = inCSLBasis[l[0]] + i*inCSLBasis[l[1]]+j*inCSLBasis[l[2]]
                for s in lstFactors:
                    if np.all(np.mod(arrAxis, s*np.ones(3)) == np.zeros(3)):
                        arrCurrent = (arrAxis/s)/np.gcd.reduce((arrAxis/s).astype('int'))
                        if s in dctAxes:
                            if np.max(np.abs(arrCurrent)) <  np.max(np.abs(dctAxes[s])):
                                dctAxes[s] = arrCurrent
                        else:
                            dctAxes[s] = arrCurrent
                        lstUsedFactors.append(intMax/s)
    print(np.unique(lstUsedFactors))
    return dctAxes            
#%%            
objCSL = gl.CSLTripleLine(np.array([1,1,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[21,21,49],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,False)
arrCellBasis = objCSL.GetCSLPrimitiveVectors()
dctAxes = FindPossibleAxes(2*arrCellBasis, [3,7,21,49,147])
arrR = gf.FindReciprocalVectors(arrCellBasis)



# arrBasis1 = 4.05*np.matmul(ld.FCCPrimitive,objCSL.GetOriginalBasis(0))
# arrBasis2 = 4.05*np.matmul(ld.FCCPrimitive,objCSL.GetOriginalBasis(2))
# arrBasis3 = 4.05*np.matmul(ld.FCCPrimitive,objCSL.GetOriginalBasis(1))

# objOrient = gf.EcoOrient(4.05,0.25)
# flt1E13 = objOrient.GetOrderParameter(arrBasis1, arrBasis1,arrBasis3)
# flt2E13 = objOrient.GetOrderParameter(arrBasis2, arrBasis1,arrBasis3)
# flt3E13 =  objOrient.GetOrderParameter(arrBasis3, arrBasis1,arrBasis3)
# flt1E12 = objOrient.GetOrderParameter(arrBasis1, arrBasis1,arrBasis2)
# flt2E12 = objOrient.GetOrderParameter(arrBasis2, arrBasis1,arrBasis2)
# flt3E12 =  objOrient.GetOrderParameter(arrBasis3, arrBasis1,arrBasis2)
# print(flt1E13,flt2E13,flt3E13,flt1E12,flt2E12,flt3E12)
# u0 = 0.02
# r = (1+ flt3E12)/(1+flt2E13)
# u1 = u0/(2+r*(1-flt2E13))
# u2 = r*u1
#objS = gl.SigmaCell(np.array([1,1,1]),ld.FCCCell)
#objS.MakeCSLCell(147,False)
#arrCellBasis = objS.GetCSLPrimitiveVectors()
objCSLConjugate = CSLConjugateBases(gf.StandardBasisVectors(3))
arrOut = objCSLConjugate.FindConjugates(arrCellBasis)
a = 4.05
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
print(np.round(objCSLConjugate.GetCellSigmaValue()*np.array(lstTransforms),5),lstSigma)
# %%
