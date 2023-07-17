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
objCSL = gl.CSLTripleLine(np.array([5,1,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[9,9,9],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,True)
arrCellBasis = objCSL.GetCSLBasisVectors()
arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrCellBasis)
#%%
#lstGrainColours = ['goldenrod','blue','green']
n=2 # scale factors
dctPoints = dict()
lstPoints = []
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
objSimulationCell = gl.SimulationCell(n*arrEdgeVectors)
for i in lstOrder:
    arrGrain1 = gl.ParallelopiedGrain(n*arrEdgeVectors,objCSL.GetLatticeBasis(i),ld.FCCCell,np.ones(3), np.zeros(3))
    objSimulationCell.AddGrain(arrGrain1,str(i))
    objSimulationCell.RemoveAtomsOnOpenBoundaries()
    arrPoints = arrGrain1.GetAtomPositions()
    #arrRows = np.where(arrPoints[:,2] < 2)[0]
    #arrPoints = arrPoints[arrRows]
    arrPoints = np.unique(np.round(arrPoints,5) ,axis=0)
    dctPoints[i] = arrPoints
    arrPoints[:,2]= arrPoints[:,2]
    lstPoints.append(arrPoints)
    ax.scatter(*tuple(zip(*arrPoints)),s=8,c=lstGrainColours[lstOrder[i]])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
gf.EqualAxis3D(ax)
plt.axis('off')
plt.legend(['$\Lambda_1$','$\Lambda_2$','$\Lambda_3$'],loc='center right')
plt.tight_layout()
plt.show()
#%%
def MakeCoincidentDictionary(inlstPoints: list,inEdgeVectors: np.array, fltBiTolerance: float, fltTriTolerance: float)->dict():
    arrAllPoints = np.vstack(inlstPoints)
    fltOverlap = 25#3*np.max([fltBiTolerance,fltTriTolerance])
    dctCoincidence = dict()
    for i in range(len(inlstPoints)):
        objKDTreeI = gf.PeriodicWrapperKDTree(inlstPoints[i],inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), fltOverlap)
        arrExtendedPointsI = objKDTreeI.GetExtendedPoints()
        for j in range(i+1,len(inlstPoints)):
            objKDTreeJ = gf.PeriodicWrapperKDTree(inlstPoints[j],inEdgeVectors,gf.FindConstraintsFromBasisVectors(inEdgeVectors), fltOverlap)
            arrDistancesI,arrIndicesI = objKDTreeI.Pquery(inlstPoints[j])
            arrRowsI = np.where(arrDistancesI <=fltBiTolerance)[0]
            arrIndicesI = arrIndicesI[arrRowsI] 
            arrIndicesI = np.unique(mf.FlattenList(arrIndicesI))
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
dctExactCoincidence,arrExactTJ,arrExactDistances = MakeCoincidentDictionary(lstPoints,n*arrEdgeVectors,0.001,0.001)
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
def GetBicrystalAtomicLayerInDirection(inlstPoints: list,arrCoincidence: np.array,arrDirection: np.array,inEdgeVectors: np.array,intLayer: int, intNumberOfLayers:int):
    arrNormal = gf.NormaliseVector(arrDirection)
    nlength = np.linalg.norm(arrDirection)
    lstZPoints = []
    nMin =  intLayer*nlength/intNumberOfLayers -0.05  
    nMax =  intLayer*nlength/intNumberOfLayers +0.05
    #lstGrainOrder = np.array([1,2,3])[inGrainsList]
    for i in inlstPoints:
        #arrGrain1 = gl.ParallelopiedGrain(inEdgeVectors,objCSL.GetLatticeBasis(lstOrder[i]),ld.FCCCell,np.ones(3), np.zeros(3))
        #objSimulationCell.AddGrain(arrGrain1,str(lstOrder[i]))
        #arrPoints = arrGrain1.GetAtomPositions()
        arrPoints = i
        arrRows = np.where((nMin<= np.dot(arrPoints,arrNormal)) &(np.dot(arrPoints,arrNormal) <= nMax))[0]
        #lstPoints.append(arrPoints[arrRows])
        arrPoints = arrPoints[arrRows]
        lstZPoints.append(arrPoints)
       # plt.scatter(*tuple(zip(*arrPoints[:,:2])),s=18,c=lstColors[i])
    #arrCoincidence = indctCoincidence[tuple(inGrainsList)]
    arrRows = np.where((nMin <= arrCoincidence[:,2]) &(arrCoincidence[:,2] <= nMax))[0]
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
strGrainMarker = 'o'
strCSLMarker = 's'
strNearCSLMarker = 'X'
lstGrainColours = ['goldenrod','darkcyan','purple']
lstCoincidenceColours = ['darkolivegreen','saddlebrown','darkblue','black']
arrTranslate = arrEdgeVectors[0,:2]
intLayer = 0
lstGrains1 = [0,1]
lstBiPoints1, arrCoincide12 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains1],dctCoincidence[tuple(lstGrains1)],arrEdgeVectors,intLayer,3)

lstExactBiPoints1, arrExactCoincide12 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains1],dctExactCoincidence[tuple(lstGrains1)],arrEdgeVectors,intLayer,3)

arrExactTJ = np.round(arrExactTJ,5)


#arrCoincide13 = gf.MergeTooCloseAtoms(arrCoincide13,arrEdgeVectors,0.15)

lstGrains2 = [0,2]
lstBiPoints2, arrCoincide13 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains2],dctCoincidence[tuple(lstGrains2)],arrEdgeVectors,intLayer,3)

lstExactBiPoints2, arrExactCoincide13 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains2],dctExactCoincidence[tuple(lstGrains2)],arrEdgeVectors,intLayer,3)

lstGrains3 = [1,2]
lstBiPoints3, arrCoincide23 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains3],dctCoincidence[tuple(lstGrains3)],arrEdgeVectors,intLayer,3)

lstExactBiPoints3, arrExactCoincide23 = GetBicrystalAtomicLayer(np.array(lstPoints)[lstGrains3],dctExactCoincidence[tuple(lstGrains3)],arrEdgeVectors,intLayer,3)



#arrCoincide13 = gf.MergeTooCloseAtoms(arrCoincide13,arrEdgeVectors,0.15)

if len(arrTJ) > 0:
    lstEmpty, arrCoincideTJ = GetBicrystalAtomicLayer([],arrTJ,arrEdgeVectors,intLayer,3)
else:
    arrTJ = []

if len(arrExactTJ) > 0:
    lstEmpty, arrExactCoincideTJ = GetBicrystalAtomicLayer([],arrExactTJ,arrEdgeVectors,intLayer,3)
else:
    arrExactCoincideTJ = []


#arrCoincide23 = gf.MergeTooCloseAtoms(arrCoincide23,arrEdgeVectors,0.15)
blnNearTJ = False # True
blnNear12 = False #True
blnNear13 = False #True
blnNear23 = False #True
blnExactTJ =  True
blnExact12 = True
blnExact13 = True
blnExact23 = True
bln12 = True
bln13 = True # True
bln23 = True
blnHalf = True
if blnHalf:
    intS = 10
    fltWrapper = 0.1
    arrTranslate = np.zeros(2)
else:
    intS = 5
    fltWrapper = 0.1
if bln12:
    for p in range(len(lstBiPoints1)):
        arrWrapped12 = gf.AddPeriodicWrapper(lstBiPoints1[p][:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
        plt.plot(*tuple(zip(*(arrWrapped12+arrTranslate))),c=lstGrainColours[lstGrains1[p]],linestyle='None',marker=strGrainMarker,markersize=intS)
      #  plt.plot(*tuple(zip(*(arrWrapped12))),c=lstGrainColours[lstGrains1[p]],linestyle='None',marker=strGrainMarker,markersize=intS)
   #plt.scatter(*tuple(zip(*(lstBiPoints1[p][:,:2]))),s=6,c=lstGrainColours[lstGrains1[p]])
if bln13:
    for p in range(len(lstBiPoints2)):
        arrWrapped13 = gf.AddPeriodicWrapper(lstBiPoints2[p][:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
        plt.plot(*tuple(zip(*(arrWrapped13))),c=lstGrainColours[lstGrains2[p]],linestyle='None',marker=strGrainMarker,markersize=intS)
      #  plt.plot(*tuple(zip(*(arrWrapped13+arrTranslate))),c=lstGrainColours[lstGrains2[p]],linestyle='None',marker=strGrainMarker,markersize=intS)
if bln23:
    for p in range(len(lstBiPoints3)):
        arrWrapped23 = gf.AddPeriodicWrapper(lstBiPoints3[p][:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
        plt.plot(*tuple(zip(*(arrWrapped23))),c=lstGrainColours[lstGrains3[p]],linestyle='None',marker=strGrainMarker,markersize=intS)
if len(arrCoincide12) > 0 and blnNear12:
    arrWrapped12C = gf.AddPeriodicWrapper(arrCoincide12[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrWrapped12C))),c=lstCoincidenceColours[0],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped12C+arrTranslate))),c=lstCoincidenceColours[0],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
if len(arrExactCoincide12) > 0 and blnExact12:
    arrExactWrapped12C = gf.AddPeriodicWrapper(arrExactCoincide12[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
  #  plt.plot(*tuple(zip(*(arrExactWrapped12C))),c=lstCoincidenceColours[0],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)  
    plt.plot(*tuple(zip(*(arrExactWrapped12C+arrTranslate))),c=lstCoincidenceColours[0],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)   
if len(arrCoincide13) > 0 and blnNear13:
    arrWrapped13C = gf.AddPeriodicWrapper(arrCoincide13[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrWrapped13C))),c=lstCoincidenceColours[1],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped13C+arrTranslate))),c=lstCoincidenceColours[1],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
if len(arrExactCoincide13) > 0 and blnExact13:
    arrExactWrapped13C = gf.AddPeriodicWrapper(arrExactCoincide13[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrExactWrapped13C))),c=lstCoincidenceColours[1],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)
  #  plt.plot(*tuple(zip(*(arrExactWrapped13C+arrTranslate))),c=lstCoincidenceColours[1],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)
if len(arrCoincide23) > 0 and blnNear23:
    arrWrapped23C = gf.AddPeriodicWrapper(arrCoincide23[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrWrapped23C))),c=lstCoincidenceColours[2],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped23C+arrTranslate))),c=lstCoincidenceColours[2],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
if len(arrExactCoincide23) > 0 and blnExact23:
    arrExactWrapped23C = gf.AddPeriodicWrapper(arrExactCoincide23[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrExactWrapped23C))),c=lstCoincidenceColours[2],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrExactWrapped23C+arrTranslate))),c=lstCoincidenceColours[2],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)
# if blnExactTJ:
#     plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
# elif bln12:    
#     plt.xlim([-0.5+arrEdgeVectors[0,0],2*arrEdgeVectors[0,0]+0.5])
# elif bln13:
if blnHalf:
    plt.xlim([-0.5,arrEdgeVectors[0,0]+0.5])
    #plt.xrange([-0.5,2*arrEdgeVectors[0,0]-0.5])
else:
    plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
if len(arrCoincideTJ) > 0 and blnNearTJ:
    arrWrappedTJ = gf.AddPeriodicWrapper(arrCoincideTJ[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*arrWrappedTJ)),c=lstCoincidenceColours[3],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrappedTJ+arrTranslate))),c=lstCoincidenceColours[3],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
if len(arrExactCoincideTJ) > 0 and blnExactTJ:
    arrExactWrappedTJ = gf.AddPeriodicWrapper(arrExactCoincideTJ[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*arrExactWrappedTJ)),c=lstCoincidenceColours[3],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrExactWrappedTJ+arrTranslate))),c=lstCoincidenceColours[3],linestyle='None',marker=strCSLMarker,markersize=1.2*intS)
plt.ylim([-1,arrEdgeVectors[1,1]+0.5])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
#plt.axis('equal')
plt.axis('off')
#plt.xlim([arrEdgeVectors[0,0],2*a*arrEdgeVectors[0,1]])
#plt.ylim([arrEdgeVectors[1,0],a*arrEdgeVectors[1,1]])
#plt.annotate(text='', xy=(0,-0.5), xytext=(np.sqrt(2)/2,-0.5),c='black',fontsize=5, arrowprops={'arrowstyle':'<->'},color='black')
plt.annotate(text='$r_0$', xy=(0.4,-1),ha='center',fontsize=24,c='black')
plt.arrow(0,-0.5, np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
plt.arrow(np.sqrt(2)/2,-0.5, -np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
#plt.arrow(1,-0.5, -1, 0, head_width=0.05, length_includes_head=True,color='black')
plt.show()


#%%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*tuple(zip(*arrTJ)))
plt.show()
#%%
#arrFacetPoint = arrCoincide13[10]
arrFacetPoint = arrCoincide13[1]
dctExactTJByLevel = dict()
dctTJByLevel = dict()
#%%
if len(arrCoincide12) > 0 and blnNear12:
    arrCoincide12 = arrCoincide12[np.argsort(np.linalg.norm(arrCoincide13,axis=1))]
    arrWrapped12 = gf.AddPeriodicWrapper(arrCoincide12[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrWrapped12))),linestyle='None',c=lstCoincidenceColours[0],marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped12+arrTranslate))),linestyle='None',c=lstCoincidenceColours[0],marker=strNearCSLMarker,markersize=1.2*intS)
if len(arrCoincide13) > 0 and blnNear13:
    arrCoincide13 = arrCoincide13[np.argsort(np.linalg.norm(arrCoincide13,axis=1))]
    arrWrapped13 = gf.AddPeriodicWrapper(arrCoincide13[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrWrapped13))),c=lstCoincidenceColours[1],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped13+arrTranslate))),linestyle='None',c=lstCoincidenceColours[1],marker=strNearCSLMarker,markersize=1.2*intS)
if len(arrCoincide23) > 0 and blnNear23:
    #arrCoincide23 = arrCoincide23[np.argsort(np.linalg.norm(arrCoincide13,axis=1))]
    arrWrapped23 = gf.AddPeriodicWrapper(arrCoincide23[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrWrapped23))),c=lstCoincidenceColours[2],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrapped23+arrTranslate))),linestyle='None',c=lstCoincidenceColours[2],marker=strNearCSLMarker,markersize=1.2*intS)
# arrWrapped3 = gf.AddPeriodicWrapper(arrCoincide23[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
# plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
# plt.ylim([-0.5,arrEdgeVectors[1,1]+0.5])
# plt.plot(*tuple(zip(*arrWrapped3)),linestyle='None',c=lstCoincidenceColours[2],marker=strNearCSLMarker,markersize=1.2*intS)
# plt.plot(*tuple(zip(*(arrWrapped3+arrTranslate))),linestyle='None',c=lstCoincidenceColours[2],marker=strNearCSLMarker,markersize=1.2*intS)

if len(arrCoincideTJ) > 0 and blnNearTJ:
    #arrCoincideTJ = arrCoincideTJ[np.argsort(np.linalg.norm(arrCoincide13,axis=1))]
    arrWrappedTJ = gf.AddPeriodicWrapper(arrCoincideTJ[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.plot(*tuple(zip(*(arrWrappedTJ))),c=lstCoincidenceColours[3],linestyle='None',marker=strNearCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrWrappedTJ+arrTranslate))),linestyle='None',c=lstCoincidenceColours[3],marker=strNearCSLMarker,markersize=1.2*intS)
    lstTJs = []
    lstTJs.append(arrWrappedTJ)
    lstTJs.append(arrWrappedTJ+arrTranslate)
    dctTJByLevel[intLayer] = np.concatenate(lstTJs,axis=0)
### For 7-7-49 use level 1 use arrconincide2[10]
#arrFacetPoint = arrCoincide13[10]
blnIncludeGBs = True
if blnIncludeGBs:
    #brown bottom long
    plt.plot([0,arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[1])
    #brown bottom short
    plt.plot([arrEdgeVectors[0,0],arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[1])
    #brown top short
    plt.plot([0,arrEdgeVectors[0,0]-arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[1])
    #brown top long
    plt.plot([arrEdgeVectors[0,0],arrEdgeVectors[0,0]-arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[1])
    #green bottom long
    plt.plot([2*arrEdgeVectors[0,0],2*arrEdgeVectors[0,0]-arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[0])
    #green bottom short
    plt.plot([arrEdgeVectors[0,0],2*arrEdgeVectors[0,0]-arrFacetPoint[0]],[0,arrFacetPoint[1]],c=lstCoincidenceColours[0])
    #green top long
    plt.plot([arrEdgeVectors[0,0],arrEdgeVectors[0,0]+arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[0])
    #green top short
    plt.plot([2*arrEdgeVectors[0,0],arrEdgeVectors[0,0]+arrFacetPoint[0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[0])
    ###additional points for TJ movement
    ## For Sigma 21-21-49 also used arrCoincide13[9]
   # plt.plot([arrFacetPoint[0],arrEdgeVectors[0,0]],[arrFacetPoint[1],arrEdgeVectors[1,1]/2],c=lstCoincidenceColours[1])
 #   plt.plot([2*arrEdgeVectors[0,0] - arrFacetPoint[0],arrEdgeVectors[0,0]],[arrFacetPoint[1],arrEdgeVectors[1,1]/2],c=lstCoincidenceColours[0])
    #arrFacetPoint2 = arrExactCoincide13[9]
    
    # plt.plot([arrFacetPoint2[0],arrEdgeVectors[0,0]],[arrFacetPoint2[1],arrEdgeVectors[1,1]/2],c=lstCoincidenceColours[1])
    # plt.plot([2*arrEdgeVectors[0,0]-arrFacetPoint2[0],arrEdgeVectors[0,0]],[arrFacetPoint2[1],arrEdgeVectors[1,1]/2],c=lstCoincidenceColours[0])
    # plt.plot([arrFacetPoint2[0],arrEdgeVectors[0,0]],[arrFacetPoint2[1],arrEdgeVectors[1,1]],c=lstCoincidenceColours[1],linestyle='dashed')
    # plt.plot([2*arrEdgeVectors[0,0]-arrFacetPoint2[0],arrEdgeVectors[0,0]],[arrFacetPoint2[1],arrEdgeVectors[1,1]],c=lstCoincidenceColours[0],linestyle='dashed')
    # #green top long
    # plt.plot([2*arrEdgeVectors[0,0]-arrFacetPoint2[0],arrEdgeVectors[0,0]+arrFacetPoint[0]],[arrFacetPoint2[1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[0])
    # #brown top long
    # plt.plot([arrFacetPoint2[0],arrEdgeVectors[0,0]-arrFacetPoint[0]],[arrFacetPoint2[1],arrEdgeVectors[1,1] -arrFacetPoint[1]],c=lstCoincidenceColours[1])
    
#pl
#plt.plot([arrEdgeVectors[0,0],arrEdgeVectors[0,0]-arrCoincide13[10,0]],[arrEdgeVectors[1,1],arrEdgeVectors[1,1] -arrCoincide13[10,1]],c=lstCoincidenceColours[1])




if len(arrExactCoincide12) > 0 and blnExact12:
    arrExactWrapped12 = gf.AddPeriodicWrapper(arrExactCoincide12[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
    plt.ylim([-1,arrEdgeVectors[1,1]+0.5])
    plt.plot(*tuple(zip(*arrExactWrapped12)),linestyle='None',c=lstCoincidenceColours[0],marker=strCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrExactWrapped12+arrTranslate))),linestyle='None',c=lstCoincidenceColours[0],marker=strCSLMarker,markersize=1.2*intS)
if len(arrExactCoincide13) > 0 and blnExact13:
    arrExactWrapped13 = gf.AddPeriodicWrapper(arrExactCoincide13[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
    plt.ylim([-1,arrEdgeVectors[1,1]+0.5])
    plt.plot(*tuple(zip(*arrExactWrapped13)),linestyle='None',c=lstCoincidenceColours[1],marker=strCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrExactWrapped13+arrTranslate))),linestyle='None',c=lstCoincidenceColours[1],marker=strCSLMarker,markersize=1.2*intS)
if len(arrExactCoincide23) > 0 and blnExact23:
    arrExactWrapped23 = gf.AddPeriodicWrapper(arrExactCoincide23[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
    plt.ylim([-1,arrEdgeVectors[1,1]+0.5])
    plt.plot(*tuple(zip(*arrExactWrapped23)),linestyle='None',c=lstCoincidenceColours[2],marker=strCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrExactWrapped23+arrTranslate))),linestyle='None',c=lstCoincidenceColours[2],marker=strCSLMarker,markersize=1.2*intS)
if len(arrExactCoincideTJ) > 0 and blnExactTJ:
    #arrExactWrappedTJ = gf.AddPeriodicWrapper(arrExactCoincideTJ[:,:2],arrEdgeVectors[:2,:2],fltWrapper,False)
    plt.xlim([-0.5,2*arrEdgeVectors[0,0]+0.5])
    plt.ylim([-1,arrEdgeVectors[1,1]+0.5])
    plt.plot(*tuple(zip(*arrExactWrappedTJ)),linestyle='None',c=lstCoincidenceColours[3],marker=strCSLMarker,markersize=1.2*intS)
    plt.plot(*tuple(zip(*(arrExactWrappedTJ+arrTranslate))),linestyle='None',c=lstCoincidenceColours[3],marker=strCSLMarker,markersize=1.2*intS)
    lstExactTJs = []
    lstExactTJs.append(arrExactWrappedTJ)
    lstExactTJs.append(arrExactWrappedTJ+arrTranslate)
    dctExactTJByLevel[intLayer] = np.concatenate(lstExactTJs,axis=0)

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
#plt.annotate(text='', xy=(0,-0.5), xytext=(1,-0.5),c='black',fontsize=3, arrowprops=dict(arrowstyle='|-|',color='black'))
#plt.annotate(text='$a_0$', xy=(0.25,-1),fontsize=12,c='black')
plt.annotate(text='$r_0$', xy=(0.4,-1),ha='center',fontsize=12,c='black')
plt.arrow(0,-0.5, np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
plt.arrow(np.sqrt(2)/2,-0.5, -np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
plt.axis('off')
plt.show()
#%%
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
lstColours = ['black','dimgrey','silver']
plt.arrow(0,-1, np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
plt.arrow(np.sqrt(2)/2,-1, -np.sqrt(2)/2, 0, head_width=0.05,length_includes_head=True,color='black')
plt.arrow(arrEdgeVectors[0,0],0,0,arrEdgeVectors[1,1],color='black')
plt.arrow(arrEdgeVectors[0,0]-1.435,0,0,arrEdgeVectors[1,1],color='black')
plt.arrow(arrEdgeVectors[0,0]+1.435,0,0,arrEdgeVectors[1,1],color='black')
##For sigma21-21-49
#plt.arrow(arrEdgeVectors[0,0]-1.5,0,0,arrEdgeVectors[1,1],color='black', linestyle='dotted')
#plt.arrow(arrEdgeVectors[0,0]+1.5,0,0,arrEdgeVectors[1,1],color='black', linestyle='dotted')
#plt.scatter(*tuple(zip(*dctTJByLevel[0])))
#plt.scatter(*tuple(zip(*dctTJByLevel[1])))
for i in list(dctTJByLevel.keys()):
    plt.scatter(*tuple(zip(*dctTJByLevel[i])),marker='X',c=lstColours[i])
for j in list(dctExactTJByLevel.keys()):
    plt.scatter(*tuple(zip(*dctExactTJByLevel[j])),marker='s',c=lstColours[j])
#plt.scatter(*tuple(zip(*dctTJByLevel[1])),marker='X',c='dimgrey')
#plt.scatter(*tuple(zip(*dctTJByLevel[2])),marker='X',c='silver')

#plt.scatter(*tuple(zip(*dctExactTJByLevel[0])),marker='s',c='black')
#plt.scatter(*tuple(zip(*dctExactTJByLevel[1])),marker='s',c='dimgrey')
#plt.scatter(*tuple(zip(*dctExactTJByLevel[2])),marker='s',c='silver')
plt.axis('off')
plt.annotate(text='$r_0$', xy=(0.4,-0.75),ha='center',fontsize=12,c='black')
plt.show()

# %%
%matplotlib inline
lstCoincidenceColours = ['darkolivegreen','saddlebrown','darkblue','black']
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
k = 0
arrStart = gf.NormaliseVector(-arrEdgeVectors[0]/3+2*arrEdgeVectors[2]/3)
fltAngle,arrAxis = gf.FindRotationVectorAndAngle(arrStart,np.array([0,0,1]))
for m in dctExactCoincidence:
    if k < 3:
        arrPoints = dctExactCoincidence[m]
        if len(arrPoints)> 0:
           #arrPoints = gf.RotateVectors(fltAngle,    arrAxis,arrPoints)
           # arrPoints = gf.RotateVectors(-np.pi/2,np.array([0,1,0]),arrPoints)
            arrRows = np.where((arrPoints[:,2]>=-0.1) & (arrPoints[:,2] < 4))
            arrPoints = arrPoints[arrRows]
           # ax.scatter(*tuple(zip(*arrPoints)),s=36,marker='s',c=lstCoincidenceColours[k])
        else:
            print(m)
    k +=1
arrNewTJ = arrExactTJ
#arrRows = np.where((arrNewTJ[:,2]>=arrEdgeVectors[2,2]/2-0.1)) #& (arrNewTJ[:,2] <= arrEdgeVectors[2,2]/4+0.1))
arrRows3 = np.where((arrNewTJ[:,2] > 3) & (arrNewTJ[:,2] < 4))
arrRows2 = np.where((arrNewTJ[:,2] > 1) & (arrNewTJ[:,2] < 3))
arrRows1  = np.where(np.round(arrNewTJ[:,2],1)==0) 
#arrNewTJ = gf.RotateVectors(fltAngle,arrAxis,arrNewTJ)
#arrPoints = gf.RotateVectors(np.pi/2,np.array([0,1,0]),arrNewTJ)
ax.scatter(*tuple(zip(*arrNewTJ[arrRows3])),s=36,marker='s',c='silver')
ax.scatter(*tuple(zip(*arrNewTJ[arrRows2])),s=36,marker='s',c='dimgrey')
ax.scatter(*tuple(zip(*arrNewTJ[arrRows1])),s=36,marker='s',c='black')

#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.set_xlim([0,1])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
#ax.get_zaxis().line.set_linewidth(0)
#ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) 
#ax.axis('off')
#ax.plot([0,1],[0,0],[0,0],c='black')
#ax.plot([0,0],[0,1],[0,0],c='black')
#ax.plot([0,0],[0,0],[0,1],c='black')
#ax.yaxis.grid(True)
#ax.zaxis.grid(False)
#plt.scatter(*tuple(zip(*arrTJ)))
#lstLegendCoincidence.append((0,1,2))
#plt.axis('square')
#plt.legend(lstLegendCoincidence)
gf.EqualAxis3D(ax)
ax.view_init(elev=90, azim=90)
ax.set_ylim([0,4])
ax.set_zlim([0,4])
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
#Sigma 7-7-49 arrFacetPoint = arrCoincide13[10]
#m1 =1.5*arrEdgeVectors[1,1]/(2.5*arrEdgeVectors[0,0])
#m2 =-3*arrEdgeVectors[1,1]/(2*arrEdgeVectors[0,0])
#Sigma 21-21-49 arrFacetPoint = arrCoincide13[2]
blnGBs = False

if blnGBs:
    m1 = -arrFacetPoint[0]/(arrEdgeVectors[0,0]-arrFacetPoint[0])
    m1 = -arrFacetPoint[0]/(-arrFacetPoint[0])
    m2 = -arrFacetPoint[1]/(arrEdgeVectors[0,0]-arrFacetPoint[0])

    s = 0.15
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
