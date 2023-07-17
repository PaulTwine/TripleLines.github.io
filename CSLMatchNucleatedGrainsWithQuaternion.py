# %%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
import CSLandDSCGraphs as CD
import LAMMPSTool as LT
# from scipy.linalg import lu_factor
# from gmpy2 import mpz, xmpz
# from sympy import Matrix, ZZ
# from sympy.matrices.normalforms import smith_normal_form
#from hsnf import column_style_hermite_normal_form, row_style_hermite_normal_form, smith_normal_form


# %%
class CSLSubLatticeBases(object):
    def __init__(self, inCSLBasis: np.array, arrPrimitiveVectors: np.array):
        self.__SigmaValue = int(
            np.round(np.abs(np.linalg.det(inCSLBasis)/np.linalg.det(arrPrimitiveVectors))))
        self.__CSLBasis = inCSLBasis

    def GetCellSigmaValue(self):
        return self.__SigmaValue

    def FindTransformationsByReciprocalLattice(self,blnDirectOnly = False):
        lstRowsPermutations = list(it.permutations((0,1,2),3))
        lstTransforms = []
        intCSLBasis = 2*self.__CSLBasis
        arrR = np.transpose(gf.FindReciprocalVectors(
            np.transpose(intCSLBasis)))
        lstVectors = []
        intL1 = np.ceil(
            1/np.abs(np.dot(arrR[0], gf.NormaliseVector(np.cross(arrR[1], arrR[2]))))).astype('int')+1
        intL2 = np.ceil(
            1/np.abs(np.dot(arrR[1], gf.NormaliseVector(np.cross(arrR[2], arrR[0]))))).astype('int')+1
        intL3 = np.ceil(
            1/np.abs(np.dot(arrR[2], gf.NormaliseVector(np.cross(arrR[0], arrR[1]))))).astype('int')+1
        for a in range(-intL1, intL1):
            for b in range(-intL2, intL2):
                for c in range(-intL3, intL3):
                    lstVectors.append(a*arrR[0]+b*arrR[1]+c*arrR[2])
        arrVectors = np.vstack(lstVectors)
        arrLengths = np.linalg.norm(arrVectors, axis=1)
        arrRows = np.where(np.round(arrLengths, 10) == 1)[0]
        arrUnitVectors = arrVectors[arrRows]
        arrRows2, arrCols2 = np.where(
            np.abs(np.matmul(arrUnitVectors, np.transpose(arrUnitVectors))) < 1e-10)
        for n in arrRows2:
            arrCheck = np.where(arrRows2 == n)[0]
            arrVectors2 = arrUnitVectors[arrCols2][arrCheck]
            arrRows3, arrCols3 = np.where(
                np.abs(np.matmul(arrVectors2, np.transpose(arrVectors2))) < 1e-10)
            for i in range(len(arrRows3)):
                arrMatrix = np.zeros([3, 3])
                if arrRows3[i] < arrCols3[i]:  # check no double counting
                    arrMatrix[0] = arrUnitVectors[n]
                    arrMatrix[1] = arrVectors2[arrRows3][i]
                    arrMatrix[2] = arrVectors2[arrCols3][i]
                    if np.all(np.round(np.matmul(arrMatrix, np.transpose(arrMatrix)), 10) == np.identity(3)):
                        fltDet = np.round(np.linalg.det(arrMatrix),10)
                        if blnDirectOnly and fltDet ==1:
                            lstTransforms.append(arrMatrix)
                        else:
                            lstTransforms.append(arrMatrix)
                        # for i in lstRowsPermutations:
                        #     lstTransforms.append(arrMatrix[i,:])
        return lstTransforms
# %%
arrAxis = np.array([5,1,1])
print(gf.CubicCSLGenerator(arrAxis,15)[:,0])
objCSL = gl.CSLTripleLine(arrAxis, ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:, :, 0].astype(
    'int') == [9, 9, 9], axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex, True)
arrCellBasis = objCSL.GetCSLPrimitiveVectors()
#arrMatrix = Matrix(2*arrCellBasis)
# smith_normal_form(arrMatrix,domain=ZZ)
print(arrCellBasis)
arrAxes = gf.FindAxesFromSigmaValues(81, 200)
objSigma = gl.SigmaCell(np.array([5, 1, 1]), ld.FCCCell)




objCSLConjugate = CSLSubLatticeBases(arrCellBasis, ld.FCCPrimitive)
lstNewT = objCSLConjugate.FindTransformationsByReciprocalLattice(True)
a = 4.05
arrOut = np.array(lstNewT)
#arrOrthogonal = gf.PrimitiveToOrthogonalVectorsGrammSchmdit(arrCellBasis,ld.FCCPrimitive)

arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(objCSL.GetCSLBasisVectors())
objSimulationCell = gl.SimulationCell(arrEdgeVectors)
arrGrain1 = gl.ParallelopiedGrain(
    arrEdgeVectors, arrTransform, ld.FCCCell, np.ones(3), np.zeros(3))
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
objPTree = gf.PeriodicWrapperKDTree(arrGrain1.GetAtomPositions(
), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis), 50, ['p', 'p', 'p'])
intTransform = 0
lstTransforms = []
lstTransforms.append(gf.StandardBasisVectors(3))
lstAxes = []
for i in range(len(arrOut)):
    arrBasis = np.matmul(arrOut[i], arrTransform)
    arrGrain1 = gl.ParallelopiedGrain(
        arrEdgeVectors, arrBasis, ld.FCCCell, np.ones(3), np.zeros(3))
    arrPoints = gf.WrapVectorIntoSimulationCell(
        arrEdgeVectors, arrGrain1.GetAtomPositions())
    arrPoints = objSimulationCell.RemoveRealDuplicates(arrPoints, 1e-5)
    arrDistances, arrIndices = objPTree.Pquery(arrPoints, k=1)
    arrDistances = np.array(mf.FlattenList(arrDistances))
    if not(np.all(arrDistances < 1e-5)):
        objSimulationCell.AddGrain(arrGrain1)
        objSimulationCell.RemoveGrainPeriodicDuplicates()
        lstPoints.append(arrPoints)
        # lstAxes.append(arrAxes[i])
        objSimulationCell.RemoveAtomsOnOpenBoundaries()
        ax.scatter(*tuple(zip(*lstPoints[-1])))
        objSimulationCell.WriteLAMMPSDataFile(
            '/home/p17992pt/' + str(intTransform+1) + '.dmp')
        objSimulationCell.RemoveAllGrains()
        objPTree = gf.PeriodicWrapperKDTree(np.vstack(
            lstPoints), arrCellBasis, gf.FindConstraintsFromBasisVectors(arrCellBasis), 50, ['p', 'p', 'p'])
        lstTransforms.append(arrOut[i])
        intTransform += 1
plt.show()
arrPoints = np.unique(np.vstack(lstPoints), axis=0)
# Matrix R is either the change of basis or you need to multiply
# the arrCellBasis by all the lstUnitMatrices
# objSimulationCell.GetCoincidentLatticePoints(['1','2'])
#lstSigma = list(map(lambda x: np.gcd.reduce(np.unique(x)),lstTransforms))
lstSigma = list(map(lambda x: objCSLConjugate.GetCellSigmaValue()/np.gcd.reduce(np.round(
    np.unique(objCSLConjugate.GetCellSigmaValue()*x, 0)).astype('int')), lstTransforms))
print(np.round(objCSLConjugate.GetCellSigmaValue()
               * np.array(lstTransforms), 5), lstSigma)
# print(np.matmul(arrCellBasis,np.linalg.inv(np.array(lstTransforms)),axis=1))
lstBases = []
for j in lstTransforms:
    lstBases.append(gf.FindPrimitiveCellVectors(j))
print('pause!')
#%%
objData = LT.LAMMPSData('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp600/u03L/TJ/1Sim90000.dmp',1,4.05, LT.LAMMPSGlobal)
objTJ = objData.GetTimeStepByIndex(-1)
#%%
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
ids = objTJ.GetGrainAtomIDs(2)
ids2 = objTJ.GetGrainAtomIDs(4)
ids = np.append(ids,ids2)
pts = objTJ.GetAtomsByID(ids)[:,1:4]
#pts = gf.MergePeriodicClusters(pts,objTJ.GetCellVectors(),['p','p','p'])[0]
#plt.axis('square')
plt.gca().set_aspect('equal')
plt.axis('off')
plt.scatter(pts[:,0],pts[:,1],c='darkgrey')
plt.show()
arrQ = objTJ.GetAtomsByID(ids)[:,10:14]
arrG = gf.GetQuaternionFromBasisMatrix(np.matmul(lstTransforms[-1],arrTransform))
ids2 = objTJ.GetAtomIDsByOrientation(arrG,1,0.025)
pts2 = objTJ.GetAtomsByID(ids2)[:,1:4]
#pts2 = gf.MergePeriodicClusters(pts2,objTJ.GetCellVectors(),['p','p','p'])[0]
plt.gca().set_aspect('equal')
plt.axis('off')
plt.scatter(pts2[:,0],pts2[:,1],c='black')
plt.show()
#%%
arrU = np.loadtxt('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/arrU.txt')
#%%
#load up the float adjust and estimate the artificial driving force per grain.
lstTheGrains = objTJ.GetGrainLabels()
lstTheGrains.remove(0)
for k in lstTheGrains:
    ids = objTJ.GetGrainAtomIDs(k)
    intCol1 = objTJ.GetColumnIndex('f_1[2]')
    intCol2 =objTJ.GetColumnIndex('f_2[2]')
    arrCols = objTJ.GetAtomsByID(ids)[:,[intCol1,intCol2]]
    arrValue = arrU[0]*arrCols[:,0] + arrU[1]*arrCols[:,1]
    print(k, np.mean(0.03*arrValue/2),np.std(0.03*arrValue/2))
#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lstGrains = []
lstGrains.append(np.matmul(lstTransforms[0],arrTransform))
lstGrains.append(np.matmul(lstTransforms[2],arrTransform))
#lstPair = CD.PutAtomPositionsIntoList(a*np.round(arrEdgeVectors,10),lstGrains,a)
dctCoincidence,arrAllCoincidence = CD.MakeCoincidentDictionary(lstGrains,arrEdgeVectors)
objCoincidence = gl.SimulationCell(arrEdgeVectors)
for j in dctCoincidence:
    pts = dctCoincidence[j]
    arrBounds = gf.FindBoundingBox(arrEdgeVectors)
    ax.set_xlim([0,arrBounds[0,1]])
    ax.set_ylim([0,arrBounds[1,1]])
    ax.set_zlim([0,arrBounds[2,1]])
    ax.scatter(*tuple(zip(*pts)))
plt.show()
# %%
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
strRoot = '/home/p17992pt/Dropbox (The University of Manchester)/paul.twine@postgrad.manchester.ac.uk’s files/Home/CSL Reciprocal/Images'
lstG, lstC = CD.GetBicrystalAtomicLayer(lstGrains,dctCoincidence,arrEdgeVectors,0,3,np.linalg.norm(np.array([1,1,1]))*np.array([0,0,1]),1)
lstGrainColours = ['darkblue','purple']
intC = 0
plt.axis('off')
for g in lstG:
    #plt.scatter(*tuple(zip(*g[:,:2])),c=lstGrainColours[intC])
    plt.scatter(*tuple(zip(*g[:,:2])),c='black')
    intC +=1
if len(lstC[0]) > 0:
    arrR = np.unique(np.vstack(lstC),axis=0)
    plt.scatter(*tuple(zip(*np.array(arrR[:,:2]))), c='black')
plt.savefig(strRoot + '/Sigma3Level0')
plt.show()
# %%
lstSigma = list(map(lambda x: objCSLConjugate.GetCellSigmaValue()/np.gcd.reduce(np.round(
    np.unique(objCSLConjugate.GetCellSigmaValue()*x, 0)).astype('int')), lstTransforms))
print(np.round(objCSLConjugate.GetCellSigmaValue()
               * np.array(lstTransforms), 5), lstSigma)
# %%
for i in range(6):
    for j in range(6):
            print(np.round(np.matmul(lstTransforms[i],lstTransforms[j])*147,10))
# %%
