#%%
import numpy as np
import GeometryFunctions as gf
import MiscFunctions as mf
import GeneralLattice as gl
import LatticeDefinitions as ld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools as it
import LAMMPSTool as LT
#%%
#a= 4.05
lstOrder = [1,0,2]
lstGrainColours = ['goldenrod','darkcyan','purple']
arrGrainLabels = np.array([1,2
,3])
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams['figure.dpi'] = 300
#%%
objCSL = gl.CSLTripleLine(np.array([1,1,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[7,7,49],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,True)
arrCellBasis = objCSL.GetCSLBasisVectors()
arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrCellBasis)
#%%
#lstGrainColours = ['goldenrod','blue','green']
dctPoints = dict()
lstPoints = []
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
objSimulationCell = gl.SimulationCell(arrEdgeVectors)
for i in lstOrder:
    arrGrain1 = gl.ParallelopiedGrain(arrEdgeVectors,objCSL.GetLatticeBasis(i),ld.FCCCell,np.ones(3), np.zeros(3))
    objSimulationCell.AddGrain(arrGrain1,str(i))
    objSimulationCell.RemoveAtomsOnOpenBoundaries()
    arrPoints = arrGrain1.GetAtomPositions()
    #arrRows = np.where(arrPoints[:,2] < 2)[0]
    #arrPoints = arrPoints[arrRows]
    arrPoints = np.unique(np.round(arrPoints,5) ,axis=0)
    dctPoints[i] = arrPoints
    lstPoints.append(arrPoints)
    ax.scatter(*tuple(zip(*arrPoints)),s=8,c=lstGrainColours[lstOrder[i]])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.axis('off')
plt.legend([1,2,3])
plt.show()
#%%
def MakeCoincidentDictionary(inlstPoints: list,inEdgeVectors: np.array, fltBiTolerance: float, fltTriTolerance: float)->dict():
    arrAllPoints = np.vstack(inlstPoints)
    fltOverlap = 25#3*np.max([fltBiTolerance,fltTriTolerance])
    dctCoincidence = dict()
    for i in range(len(inlstPoints)):
        objKDTreeI = gf.PeriodicWrapperKDTree(inlstPoints[i],inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), fltOverlap)
        arrExtendedPointsI = objKDTreeI.GetExtendedPoints()
        # arrIndicesAll = mf.FlattenList(arrIndicesAll)
        # arrIndicesAll = objKDTreeAll.GetPeriodicIndices(arrIndicesAll)
        # arrIndicesAll = np.unique(arrIndicesAll).tolist()
        # # if len(lstOverlapIndices) == 0:
        #     lstOverlapIndices.extend(arrIndicesAll)
        # else:
        #     lstOverlapIndices = list(set(lstOverlapIndices).intersection(arrIndicesAll))
        for j in range(i+1,len(inlstPoints)):
            objKDTreeJ = gf.PeriodicWrapperKDTree(inlstPoints[j],inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), fltOverlap)
            arrDistancesI,arrIndicesI = objKDTreeI.Pquery(inlstPoints[j])
            arrRowsI = np.where(arrDistancesI <=fltBiTolerance)[0]
            arrIndicesI = arrIndicesI[arrRowsI] 
            arrIndicesI = np.unique(mf.FlattenList(arrIndicesI))
            #arrIndicesI = objKDTreeI.GetPeriodicIndices(arrIndicesI)
            #arrIndicesI = np.unique(arrIndicesI)
            #arrDistancesJ,arrIndicesJ = objKDTreeJ.Pquery(inlstPoints[i][arrIndicesI])
            arrDistancesJ,arrIndicesJ = objKDTreeJ.Pquery(arrExtendedPointsI[arrIndicesI])
            arrRowsJ = np.where(arrDistancesJ <=fltBiTolerance)[0]
            arrIndicesJ = arrIndicesJ[arrRowsJ]
            arrExtendedPointsJ = objKDTreeJ.GetExtendedPoints()
            arrIndicesJ = mf.FlattenList(arrIndicesJ)
           # dctCoincidence[(i,j)] = (inlstPoints[i][arrIndicesI]+arrExtendedPointsJ[arrIndicesJ])/2
            dctCoincidence[(i,j)] = (arrExtendedPointsI[arrIndicesI]+arrExtendedPointsJ[arrIndicesJ])/2
            # lstLegendCoincidence.append((lstOrder[i]+1,lstOrder[j]+1))
    objKDTreeAll = gf.PeriodicWrapperKDTree(arrAllPoints,inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), fltOverlap)
    arrAllExtendedPoints = objKDTreeAll.GetExtendedPoints()
    arrAllDistances,arrAllIndices = objKDTreeAll.Pquery(np.unique(arrAllPoints,axis=0),3)
    #arrRows = np.where(np.all(arrAllDistances < 2*fltTriTolerance,axis=1))[0]
    lstThreePoints = []
    arrAllThreePoints = np.zeros([len(arrAllIndices),3,3])
   # arrPeriodicIndices = objKDTreeAll.GetPeriodicIndices(arrAllIndices)
    # l = 0
    #for a in arrAllIndices:
    #    arrThreePoints = arrAllExtendedPoints[a]
    #     arrAllThreePoints[l] = arrThreePoints
    #     lstThreePoints.append(arrThreePoints)
    #     lstTJPoints.append(gf.EquidistantPoint(arrThreePoints[0],arrThreePoints[1],arrThreePoints[2]))
    #     l +=1
    arrAllThreePoints = np.array(list(map(lambda x: arrAllExtendedPoints[x],arrAllIndices)))
    arrTJPoints = np.mean(arrAllThreePoints, axis=1)
   # arrDistances = np.linalg.norm(arrAllThreePoints-np.tile(arrTJPoints,3).reshape([len(arrAllIndices),3,3]),axis=2)
    arrDistances = np.linalg.norm(arrAllThreePoints-arrTJPoints[:,np.newaxis],axis=2)
    
    arrRows2 = np.where(np.all(arrDistances < fltTriTolerance,axis=1))[0]
    if len(arrRows2) > 0:
        arrReturn = arrTJPoints[arrRows2]
        arrReturn = gf.WrapVectorIntoSimulationCell(inEdgeVectors,arrReturn)
    else:
        arrReturn = []
    return dctCoincidence, arrReturn,arrDistances[arrRows2]
#%%
for i in range(1,20):
    dctCoincidence,arrTJ,arrDistances = MakeCoincidentDictionary(lstPoints,arrEdgeVectors,0.01/np.sqrt(2),(i/20)/np.sqrt(2))
    plt.hist(arrDistances)
    print(i/20,np.mean(arrDistances),np.std(arrDistances))
    plt.show()
#%%
dctCoincidence,arrTJ,arrDistances = MakeCoincidentDictionary(lstPoints,arrEdgeVectors,0.25/np.sqrt(2),0.25/np.sqrt(2))
print(np.max(np.mean(arrDistances,axis=1)))
#%%
dctExactCoincidence,arrExactTJ,arrExactDistances = MakeCoincidentDictionary(lstPoints,arrEdgeVectors,0,0)
# %%
def GetBicrystalAtomicLayer(inlstPoints: list,arrCoincidence: np.array,inEdgeVectors: np.array,intLayer: int, intNumberOfLayers:int):
    zlength = np.linalg.norm(inEdgeVectors[2])
    lstZPoints = []
    zMin =  intLayer*zlength/intNumberOfLayers -0.05  
    zMax =  intLayer*zlength/intNumberOfLayers +0.05
    #lstGrainOrder = np.array([1,2,3])[inGrainsList]
    for i in inlstPoints:
        #arrGrain1 = gl.ParallelopiedGrain(inEdgeVectors,objCSL.GetLatticeBasis(lstOrder[i]),ld.FCCCell,np.ones(3), np.zeros(3))
        #objSimulationCell.AddGrain(arrGrain1,str(lstOrder[i]))
        #arrPoints = arrGrain1.GetAtomPositions()
        arrPoints = i
        arrRows = np.where((zMin<= arrPoints[:,2]) &(arrPoints[:,2] <= zMax))[0]
        #lstPoints.append(arrPoints[arrRows])
        arrPoints = arrPoints[arrRows]
        lstZPoints.append(arrPoints)
       # plt.scatter(*tuple(zip(*arrPoints[:,:2])),s=18,c=lstColors[i])
    #arrCoincidence = indctCoincidence[tuple(inGrainsList)]
    arrRows = np.where((zMin <= arrCoincidence[:,2]) &(arrCoincidence[:,2] <= zMax))[0]
    if len(arrRows) > 0:
        arrCoincidence = arrCoincidence[arrRows]    
        #plt.scatter(*tuple(zip(*arrCoincidence[:,:2])),c='black')
    else:
        arrCoincidence = []
   # plt.axis('equal')
   # plt.legend(lstGrainOrder)
   # plt.show()
    return lstZPoints,arrCoincidence 



#%%
#plt.rcParams['lines.markersize'] = 8
lstGrainColours = ['goldenrod','darkcyan','purple']
lstCoincidenceColours = ['darkolivegreen','saddlebrown','black']
arrTranslate = arrEdgeVectors[0,:2]
intLayer = 0
lstGrains1 = [0,1]
lstBiPoints1, arrCoincide1 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains1],dctCoincidence[tuple(lstGrains1)],arrEdgeVectors,intLayer,3)

lstExactBiPoints1, arrExactCoincide1 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains1],dctExactCoincidence[tuple(lstGrains1)],arrEdgeVectors,intLayer,3)


#arrCoincide1 = gf.MergeTooCloseAtoms(arrCoincide1,arrEdgeVectors,0.15)

lstGrains2 = [0,2]
lstBiPoints2, arrCoincide2 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains2],dctCoincidence[tuple(lstGrains2)],arrEdgeVectors,intLayer,3)

lstGrains2 = [0,2]
lstExactBiPoints2, arrExactCoincide2 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains2],dctExactCoincidence[tuple(lstGrains2)],arrEdgeVectors,intLayer,3)


#arrCoincide2 = gf.MergeTooCloseAtoms(arrCoincide2,arrEdgeVectors,0.15)

#lstEmpty, arrCoincide3 = GetBicrystalAtomicLayer([],arrTJ,arrEdgeVectors,intLayer,3)

#lstEmpty, arrExactCoincide3 = GetBicrystalAtomicLayer([],arrExactTJ,arrEdgeVectors,intLayer,3)


#arrCoincide3 = gf.MergeTooCloseAtoms(arrCoincide3,arrEdgeVectors,0.15)
arrCoincide3 = []
blnTJ = False
bln12 = True
bln13 = False
intS = 10
#arrTranslate = np.zeros(2)
if bln12:
    for p in range(len(lstBiPoints1)):
        arrWrapped12 = gf.AddPeriodicWrapper(lstBiPoints1[p][:,:2],arrEdgeVectors[:2,:2],0.1,False)
        plt.plot(*tuple(zip(*(arrWrapped12+arrTranslate))),c=lstGrainColours[lstGrains1[p]],linestyle='None',marker='o',markersize=intS)
   #plt.scatter(*tuple(zip(*(lstBiPoints1[p][:,:2]))),s=6,c=lstGrainColours[lstGrains1[p]])
if bln13:
    for p in range(len(lstBiPoints2)):
        arrWrapped13 = gf.AddPeriodicWrapper(lstBiPoints2[p][:,:2],arrEdgeVectors[:2,:2],0.1,False)
        plt.plot(*tuple(zip(*(arrWrapped13))),c=lstGrainColours[lstGrains2[p]],linestyle='None',marker='o',markersize=intS)
if len(arrCoincide1) > 0 and bln12:
    arrWrapped12C = gf.AddPeriodicWrapper(arrCoincide1[:,:2],arrEdgeVectors[:2,:2],0.1,False)
    plt.plot(*tuple(zip(*(arrWrapped12C+arrTranslate))),c=lstCoincidenceColours[0],linestyle='None',marker='s',markersize=1.2*intS)
    #plt.scatter(*tuple(zip(*(arrCoincide1[:,:2]))),s=6,c=lstCoincidenceColours[0])
if len(arrCoincide2) > 0 and bln13:
    arrWrapped13C = gf.AddPeriodicWrapper(arrCoincide2[:,:2],arrEdgeVectors[:2,:2],0.1,False)
    plt.plot(*tuple(zip(*(arrWrapped13C))),c=lstCoincidenceColours[1],linestyle='None',marker='s',markersize=1.2*intS)
 #   plt.scatter(*tuple(zip(*(arrCoincide2[:,:2]+arrTranslate))),s=6,c=lstCoincidenceColours[1])
if blnTJ:
    plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
elif bln12:    
    plt.xlim([-0.5+arrEdgeVectors[0,0],2*arrEdgeVectors[0,0]+0.5])
elif bln13:
    plt.xlim([-0.5,arrEdgeVectors[0,0]+0.5])
if len(arrCoincide3) > 0 and blnTJ:
    arrWrappedTJ = gf.AddPeriodicWrapper(arrCoincide3[:,:2],arrEdgeVectors[:2,:2],0.1,False)
    plt.plot(*tuple(zip(*arrWrappedTJ)),c=lstCoincidenceColours[2],linestyle='None',marker='o',markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrappedTJ+arrTranslate))),c=lstCoincidenceColours[2],linestyle='None',marker='o',markersize=1.2*intS)
plt.plot(*tuple(zip(*(arrWrapped12C[1:3]+arrTranslate))),c=lstCoincidenceColours[0])
plt.plot(*tuple(zip(*(arrWrapped12C[3:5]+arrTranslate))),c=lstCoincidenceColours[0])
plt.plot(*tuple(zip(*(arrWrapped12C[2:5:2]+arrTranslate))),c=lstCoincidenceColours[0])
plt.plot(*tuple(zip(*(arrWrapped12C[1:4:2]+arrTranslate))),c=lstCoincidenceColours[0])
plt.ylim([-0.5,arrEdgeVectors[1,1]+0.5])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
# plt.annotate(text='$r_0$', xy=(0.4+arrTranslate[0],-1),ha='center',fontsize=24,c='black')
# plt.arrow(np.sqrt(2)/2+arrTranslate[0],-0.5, np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
#plt.arrow(np.sqrt(2)/2,-0.5, -np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
plt.axis('off')
#plt.xlim([arrEdgeVectors[0,0],2*a*arrEdgeVectors[0,1]])
#plt.ylim([arrEdgeVectors[1,0],a*arrEdgeVectors[1,1]])
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*tuple(zip(*arrTJ)))
plt.show()
#%%
arrFacetPoint = arrCoincide2[2]

#%%

if len(arrCoincide1) > 0:
    arrCoincide1 = arrCoincide1[np.argsort(np.linalg.norm(arrCoincide1,axis=1))]
    arrWrapped1 = gf.AddPeriodicWrapper(arrCoincide1[:,:2],arrEdgeVectors[:2,:2],0.1,False)
    plt.plot(*tuple(zip(*(arrWrapped1))),linestyle='None',c=lstCoincidenceColours[0],marker='o',markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped1+arrTranslate))),linestyle='None',c=lstCoincidenceColours[0],marker='o',markersize=1.2*intS)
if len(arrCoincide2) > 0:
    arrCoincide1 = arrCoincide1[np.argsort(np.linalg.norm(arrCoincide1,axis=1))]
    arrWrapped2 = gf.AddPeriodicWrapper(arrCoincide2[:,:2],arrEdgeVectors[:2,:2],0.1,False)
    plt.plot(*tuple(zip(*(arrWrapped2))),c=lstCoincidenceColours[1],linestyle='None',marker='o',markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped2+arrTranslate))),linestyle='None',c=lstCoincidenceColours[1],marker='o',markersize=1.2*intS)
### For 7-7-49 use level 1 use arrconincide2[10]
#arrFacetPoint = arrCoincide2[10]
plt.plot([0,arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[1])
plt.plot([arrEdgeVectors[0,0],arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[1])

plt.plot([0,arrEdgeVectors[0,0]-arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[1])

plt.plot([arrEdgeVectors[0,0],arrEdgeVectors[0,0]-arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[1])

plt.plot([2*arrEdgeVectors[0,0],2*arrEdgeVectors[0,0]-arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[0])

plt.plot([arrEdgeVectors[0,0],2*arrEdgeVectors[0,0]-arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[0])

plt.plot([arrEdgeVectors[0,0],arrEdgeVectors[0,0]+arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[0])

plt.plot([2*arrEdgeVectors[0,0],arrEdgeVectors[0,0]+arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[0])

#plt.plot([arrEdgeVectors[0,0],arrEdgeVectors[0,0]-arrCoincide2[10,0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrCoincide2[10,1]],c=lstCoincidenceColours[1])



arrWrapped3 = gf.AddPeriodicWrapper(arrCoincide3[:,:2],arrEdgeVectors[:2,:2],0.1,False)
plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
plt.ylim([-0.5,arrEdgeVectors[1,1]+0.5])
plt.plot(*tuple(zip(*arrWrapped3)),linestyle='None',c=lstCoincidenceColours[2],marker='o',markersize=1.2*intS)
plt.plot(*tuple(zip(*(arrWrapped3+arrTranslate))),linestyle='None',c=lstCoincidenceColours[2],marker='o',markersize=1.2*intS)


#plt.axline(arrEdgeVectors[0,:2],slope=-3*arrEdgeVectors[1,1]/(2*arrEdgeVectors[0,0]),c=lstCoincidenceColours[1],linestyle='dotted')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.axis('off')
plt.show()


# %%
#%matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
k = 0
for m in dctCoincidence:
    if k < 3:
        arrPoints = dctCoincidence[m]
        if len(arrPoints)> 0:
            ax.scatter(*tuple(zip(*arrPoints)),s=36,alpha =0.5)
        else:
            print(m)
    k +=1
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.scatter(*tuple(zip(*arrTJ)))
#lstLegendCoincidence.append((0,1,2))
#plt.axis('square')
#plt.legend(lstLegendCoincidence)
plt.show()
#%%
strDirectory2 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp600/u03L/TJ/'
objData = LT.LAMMPSData(strDirectory2 + '1Sim85000.dmp', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
# %%
lstGrainLabels = np.copy(objAnalysis.GetGrainLabels()).tolist()
lstGrainLabels.remove(0)
for i in lstGrainLabels:
    arrIDs = objAnalysis.GetGrainAtomIDs(i)
    objAnalysis.SetPeriodicGrain(i,arrIDs.tolist(),25)
lstGBPoints = []
pts1 = objAnalysis.FindDefectiveMesh(1,2,25)
pts2 = objAnalysis.FindDefectiveMesh(1,3,25)
pts3 = objAnalysis.FindDefectiveMesh(2,3,25)
lstGBPoints.append(pts1)
lstGBPoints.append(pts2)
lstGBPoints.append(pts3)
arrAllGBPoints = np.vstack(lstGBPoints)
plt.scatter(*tuple(zip(*(pts1[:,:2]))),c='black')
plt.scatter(*tuple(zip(*(pts2[:,:2]))), c='black')
plt.scatter(*tuple(zip(*(pts3[:,:2]))), c='black')
plt.show()
# %%
arrCellVectors = objAnalysis.GetCellVectors()
arrTJPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
lstTJPoints = []
for i in range(4):
    arrPoints = np.loadtxt(strDirectory2 + 'TJ' + str(i) +'Mesh85000.txt')
    arrPoints = gf.PeriodicShiftAllCloser(arrTJPositions[i],arrPoints,arrCellVectors, np.linalg.inv(arrCellVectors),['pp','pp','pp'])
    lstTJPoints.append(np.mean(arrPoints[:,:2],axis=0))
#%%
#plt.axis('equal')
#plt.ylim([0,0.5*arrCellVectors[1,1]])
#plt.xlim([-10,arrCellVectors[0,0]])
plt.scatter(*tuple(zip(*(arrAllGBPoints[:,:2]))), c='grey')
for i in range(4):
    plt.scatter(lstTJPoints[i][0],lstTJPoints[i][1],c='grey')
#Sigma 7-7-49 arrFacetPoint = arrCoincide2[10]
#m1 =1.5*arrEdgeVectors[1,1]/(2.5*arrEdgeVectors[0,0])
#m2 =-3*arrEdgeVectors[1,1]/(2*arrEdgeVectors[0,0])
#Sigma 21-21-49 arrFacetPoint = arrCoincide2[2]
m2 = -arrFacetPoint[1]/(arrEdgeVectors[0,0]-arrFacetPoint[0])
m1 = -arrFacetPoint[1]/(arrEdgeVectors[0,0]-arrFacetPoint[0])

s = 0.2
plt.plot([lstTJPoints[0][0],lstTJPoints[0][0]+s*arrCellVectors[0,0]],[lstTJPoints[0][1],lstTJPoints[0][1]+ s*m1*arrCellVectors[0,0]],c=lstCoincidenceColours[1])

#TJ 1
plt.plot([lstTJPoints[1][0],lstTJPoints[1][0]+s*arrCellVectors[0,0]],[lstTJPoints[1][1],lstTJPoints[1][1] + s*m2*arrCellVectors[0,0]],c=lstCoincidenceColours[1])

#TJ2
plt.plot([lstTJPoints[2][0],lstTJPoints[2][0]+s*arrCellVectors[0,0]],[lstTJPoints[2][1],lstTJPoints[2][1]- s*m2*arrCellVectors[0,0]],c=lstCoincidenceColours[0])
plt.plot([lstTJPoints[2][0],lstTJPoints[2][0]-s*arrCellVectors[0,0]],[lstTJPoints[2][1],lstTJPoints[2][1]- s*m2*arrCellVectors[0,0]],c=lstCoincidenceColours[1])

#TJ3

plt.plot([lstTJPoints[3][0],lstTJPoints[3][0]+s*arrCellVectors[0,0]],[lstTJPoints[3][1],lstTJPoints[3][1]- s*m1*arrCellVectors[0,0]],c=lstCoincidenceColours[0])
plt.plot([lstTJPoints[3][0],lstTJPoints[3][0]-s*arrCellVectors[0,0]],[lstTJPoints[3][1],lstTJPoints[3][1]- s*m1*arrCellVectors[0,0]],c=lstCoincidenceColours[1])

#TJ 0 shifted
plt.plot([arrCellVectors[0,0] +lstTJPoints[0][0],arrCellVectors[0,0]+lstTJPoints[0][0]-s*arrCellVectors[0,0]],[lstTJPoints[0][1],lstTJPoints[0][1]+ s*m1*arrCellVectors[0,0]],c=lstCoincidenceColours[0])

#TJ 1 shifted
plt.plot([arrCellVectors[0,0] + lstTJPoints[1][0],arrCellVectors[0,0] +lstTJPoints[1][0]-s*arrCellVectors[0,0]],[lstTJPoints[1][1],lstTJPoints[1][1] + s*m2*arrCellVectors[0,0]],c=lstCoincidenceColours[0])

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
#plt.axis('off')
plt.show()
# plt.axline(lstTJPoints[0][:2]+arrCellVectors[0,:2],slope=-m1,c=lstCoincidenceColours[1])
# plt.axline(lstTJPoints[3][:2],slope=-m1,c='darkblue')
# plt.axline(lstTJPoints[3][:2],slope=m1,c='darkblue')
# plt.axline(lstTJPoints[1][:2],slope=m2,c='darkgreen')
# plt.axline(lstTJPoints[1][:2]+arrCellVectors[0,:2],slope=-m2,c='darkgreen')

# plt.axline(lstTJPoints[2][:2],slope=-m2,c='darkgreen')
# plt.axline(lstTJPoints[2][:2],slope=m2,c='darkgreen')
# plt.axline(lstTJPoints[3][:2],slope=-arrEdgeVectors[1,1]/(1.5*arrEdgeVectors[0,0]),c='black', linestyle='dotted')
# plt.axline(lstTJPoints[3][:2],slope=-arrEdgeVectors[1,1]/(1.5*arrEdgeVectors[0,0]),c='darkblue', linestyle='dotted')


# %%
