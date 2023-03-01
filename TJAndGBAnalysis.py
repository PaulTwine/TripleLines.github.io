#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D 
import copy as cp
from scipy import spatial
from scipy import optimize
import AxisSigmaDeltaStores as AS
#%%
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams['figure.dpi'] = 300
#%%
strBase001 = '/home/p17992pt/csf4_scratch/TJ/Axis001/TJSigma'
lstSigma001 = [5,13,17,29,37]
arrAxis = np.array([0,0,1])
objTJ001 = AS.AxisStore(arrAxis, 'TJ')
objGB001 = AS.AxisStore(arrAxis, 'GB')
for s in lstSigma001:
    strDir = strBase001 + str(s) + '/'
    objStoreTJ = AS.PopulateSigmaStoreByFile(s,arrAxis,strDir,'TJ',['GE','PE','V','S','TE'])
    objStoreGB = AS.PopulateSigmaStoreByFile(s,arrAxis,strDir,'GB',['GE','PE','V','S','TE'])
    objTJ001.SetSigmaStore(objStoreTJ, s)
    objGB001.SetSigmaStore(objStoreGB,s)
#%%
strBase101 = '/home/p17992pt/csf4_scratch/TJ/Axis101/TJSigma'
lstSigma101 = [3,9,11,19,27]
arrAxis = np.array([1,0,1])
objTJ101 = AS.AxisStore(arrAxis, 'TJ')
objGB101 = AS.AxisStore(arrAxis, 'GB')
for s in lstSigma101:
    strDir = strBase101 + str(s) + '/'
    objStoreTJ = AS.PopulateSigmaStoreByFile(s,arrAxis,strDir,'TJ',['GE','PE','V','S','TE'])
    objStoreGB = AS.PopulateSigmaStoreByFile(s,arrAxis,strDir,'GB',['GE','PE','V','S','TE'])
    objTJ101.SetSigmaStore(objStoreTJ, s)
    objGB101.SetSigmaStore(objStoreGB,s)
#%%
strBase111 = '/home/p17992pt/csf4_scratch/TJ/Axis111/TJSigma'
lstSigma111 = [3,7,13,21,31]
arrAxis = np.array([1,1,1])
objTJ111 = AS.AxisStore(arrAxis, 'TJ')
objGB111 = AS.AxisStore(arrAxis, 'GB')
for s in lstSigma111:
    strDir = strBase111 + str(s) + '/'
    objStoreTJ = AS.PopulateSigmaStoreByFile(s,arrAxis,strDir,'TJ',['GE','PE','V','S','TE'])
    objStoreGB = AS.PopulateSigmaStoreByFile(s,arrAxis,strDir,'GB',['GE','PE','V','S','TE'])
    objTJ111.SetSigmaStore(objStoreTJ, s)
    objGB111.SetSigmaStore(objStoreGB,s)    

#%%
objStoreTJ = AS.PopulateSigmaStoreByFile(31, np.array([1,1,1]),'/home/p17992pt/csf4_scratch/TJ/Axis111/TJSigma31/','TJ',['GE','PE','V','S'])
#objDirStore = objStoreTJ.GetDirStore(0)
# objDeltaStore = objDirStoreTJ.GetDeltaStore(0)
# print(objDeltaStore.GetValues('PE')[1])
#%%
objStoreGB = AS.PopulateSigmaStoreByFile(21, np.array([1,1,1]),'/home/p17992pt/csf4_scratch/TJ/Axis111/TJSigma21/','GB',['GE','PE','V','S'])
#objDirStoreGB = objStore.GetDirStore(0)
#objDeltaStoreGB = objDirStore.GetDeltaStore(0)
#print(objDeltaStore.GetValues('PE')[1])
#%%
fig, ax = plt.subplots()
#%%
##Per atom graphs of excess PE, excess volume and hydrostatc stress
strDeltaAxis = r'$\bm{\delta_{i}}$'
strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
#fltDatum = 4.05**3/4 
#fltDatum = -3.36
fltDatum = 0
#strType = 'V' 
#strType = 'PE'
strType = 'S'
#lstColours = ['b', 'c', 'r', 'g', 'm','y','k','w']
lstColours = ['black','darkolivegreen','darkgoldenrod']
intDeltaMax = 10
intDeltaMin = 0
lstAllTJs = []
for j in range(10): #directories
    objDirGB = objStoreGB.GetDirStore(j)
    objDirTJ = objStoreTJ.GetDirStore(j) 
    for i in range(intDeltaMin,intDeltaMax): #delta values
        objDeltaTJ = objDirTJ.GetDeltaStore(i)
        lstValuesTJ = objDeltaTJ.GetValues(strType)
        intCol = 0
        lstAllTJs.extend(np.concatenate(lstValuesTJ,axis=0))
        for l in lstValuesTJ[1:]:
            plt.scatter(i,(np.mean(l)-fltDatum),c=lstColours[0],label='Tripleline')
            intCol +=1
        objDeltaGB = objDirGB.GetDeltaStore(i)
        lstValuesGB = objDeltaGB.GetValues(strType)
        intCol = 0
        arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGB)))
        # if len(arrLengths) != 3:
        #     print(len(arrLengths),i,j)
        for k in arrLengths:
            arrValues = lstValuesGB[k]
            # if np.mean(arrValues)-fltDatum < 0.01:
            #      print(i,j, len(arrValues))
            if intCol ==0:
                plt.scatter(i-0.1,(np.mean(arrValues)-fltDatum),c=lstColours[1],label = 'Cylinder')
            elif intCol ==1:
                plt.scatter(i+0.1,(np.mean(arrValues)-fltDatum),c=lstColours[1],label = 'Cylinder')
                
            elif intCol ==2:
                plt.scatter(i,(np.mean(arrValues)-fltDatum),c=lstColours[2],label = 'CSL')
            #else:
            #plt.scatter(i,np.mean(k)-(4.05**3/4),c= 'b') #c =lstColours[intCol])
                #print(i,j)
               # plt.scatter(i/10,(np.mean(arrValues)-fltDatum),c='b',label = 'Grain')#c =lstColours[intCol])
            
            intCol +=1 
   # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
   #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
plt.legend(*d)
if strType == 'V':
    strYlabel = 'Mean excess volume per atom in $\AA^{3}$'   
elif strType == 'PE':
    strYlabel = 'Mean excess potential energy per atom in eV'
elif strType =='S':
    strYlabel = 'Mean hydrostatic stress per atom in eV $\AA^{-3}$'
plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
#plt.ylim([0.0, 0.06])
plt.ylabel(strYlabel)
plt.xlabel(strDMinAxis)
plt.tight_layout()
plt.show()

# %%
strType = 'PE'
fltDatum = -3.36 #4.05**3/4#
for j in range(10): #directories
    objDirGB = objStoreGB.GetDirStore(j)
    objDirTJ = objStoreTJ.GetDirStore(j) 
    for i in range(intDeltaMin,intDeltaMax): #delta values
        objDeltaTJ = objDirTJ.GetDeltaStore(i)
        lstValuesTJ = objDeltaTJ.GetValues(strType)
        intCol = 0
        for l in lstValuesTJ[1:]:
            plt.scatter(i,(np.sum(l)-len(l)*fltDatum),c=lstColours[0],label='Tripleline')
            intCol +=1
        objDeltaGB = objDirGB.GetDeltaStore(i)
        lstValuesGB = objDeltaGB.GetValues(strType)
        intCol = 0
        arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGB)))
        # if len(arrLengths) != 3:
        #     print(len(arrLengths),i,j)
        # for k in arrLengths:
        #     arrValues = lstValuesGB[k]
        #     intL = len(arrValues)
        #     # if np.mean(arrValues)-fltDatum < 0.01:
            #     print(i,j, len(arrValues))
            # if intCol ==0:
            #     plt.scatter(i-0.1,(np.sum(arrValues)-intL*fltDatum),c=lstColours[1],label = 'Cylinder')
            # elif intCol ==1:
            #     plt.scatter(i+0.1,(np.sum(arrValues)-intL*fltDatum),c=lstColours[1],label = 'Cylinder')
                
            # elif intCol ==2:
            #     plt.scatter(i,(np.sum(arrValues)-intL*fltDatum),c=lstColours[2],label = 'CSL')
            # #else:
            # #plt.scatter(i,np.mean(k)-(4.05**3/4),c= 'b') #c =lstColours[intCol])
                #print(i,j)
               # plt.scatter(i/10,(np.mean(arrValues)-fltDatum),c='b',label = 'Grain')#c =lstColours[intCol])
         #   intCol +=1
        arrGValues = objDeltaTJ.GetValues('GE')[0]
        arrExcess = arrGValues[1]-arrGValues[0]*fltDatum
        plt.scatter(i-0.1,arrExcess, c='navy',label='TJ Grain')
        # arrGValues = objDeltaGB.GetValues('GE')[0]
        # arrExcess = arrGValues[1]-arrGValues[0]*fltDatum
        # plt.scatter(i+0.1,arrExcess, c='darkgoldenrod',label='GB Grain')
         
   # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
   #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
plt.legend(*d)
if strType == 'V':
    strYlabel = 'Excess volume per atom in $\AA^{3}$ per atom'   
elif strType == 'PE':
    strYlabel = 'Excess potential energy in eV'
elif strType =='S':
    strYlabel = 'Hydrostatic stress per atom in eV $\AA^{-3}$ per atom'
plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
#plt.ylim([0.0, 0.06])
plt.ylabel(strYlabel)
plt.xlabel(strDMinAxis)
plt.tight_layout()
plt.show()
#%%
def PlotEnergyProportions(inStoreGBorTJ: AS.DirStore, strType: str):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36
    lstColours = ['black','darkolivegreen','darkgoldenrod','darkred']
    intDeltaMax = 8
    intDeltaMin = 0
    lstAllValues = []
    lstGrainValues = []
    for j in range(10): #directories
        objDir = inStoreGBorTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            if strType == 'TJ':
                objDelta = objDir.GetDeltaStore(i)
                lstValuesTJPE = objDelta.GetValues('PE')
                lstTJTotalExcess = objDelta.GetValues('TE')[0]
                fltTotalExcess = lstTJTotalExcess[0]-fltDatum*lstTJTotalExcess[1]
                intCol = 0
                intL = len(lstValuesTJPE)
                lstTripleLines = np.concatenate(lstValuesTJPE,axis=0)
                arrTJ = (np.sum(lstTripleLines)-fltDatum*len(lstTripleLines))/fltTotalExcess
                plt.scatter(i,arrTJ,c=lstColours[0],label='Tripleline')
                lstAllValues.append(arrTJ)
            elif strType == 'GB':
                objDelta = objDir.GetDeltaStore(i)
                lstValuesGBPE = objDelta.GetValues('PE')
                lstGBTotalExcess = objDelta.GetValues('TE')[0]
                fltTotalExcess = lstGBTotalExcess[0]-fltDatum*lstGBTotalExcess[1]
                intCol = 0
                arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBPE)))
                lstCylinders = []
                for k in arrLengths[:2]:
                     lstCylinders.extend(lstValuesGBPE[k])
                #     lstAllValues.extend(lstCylinders)
                arrCylinder = (np.sum(lstCylinders)-fltDatum*len(lstCylinders))/fltTotalExcess
                plt.scatter(i,arrCylinder, c=lstColours[1],label = 'Cylinder')
                arrCSL = (np.sum(lstValuesGBPE[arrLengths[2]])-fltDatum*len(lstValuesGBPE[arrLengths[2]]))/fltTotalExcess
                plt.scatter(i,arrCSL,c=lstColours[2],label = 'CSL')
                #else:
                #plt.scatter(i,np.mean(k)-(4.05**3/4),c= 'b') #c =lstColours[intCol])
                    #print(i,j)
                    # plt.scatter(i/10,(np.mean(arrValues)-fltDatum),c='b',label = 'Grain')#c =lstColours[intCol])
            lstValues = objDelta.GetValues('GE')[0]
            arrValues = (lstValues[1]-fltDatum*lstValues[0])/fltTotalExcess
            lstGrainValues.append(arrValues)
            plt.scatter(i,arrValues,c=lstColours[3],label = 'Grain')    
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Mean excess potential energy density in eV $\AA^{3}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstAllValues,lstGrainValues 

# %%
def PlotEnergyDensities(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36/(4.05**3/4)
    lstColours = ['black','darkolivegreen','darkgoldenrod']
    intDeltaMax = 8
    intDeltaMin = 0
    lstAllTJs = []
    lstAllGBs = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j)
        objDirTJ = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirTJ.GetDeltaStore(i)
            lstValuesTJPE = objDeltaTJ.GetValues('PE')
            lstValuesTJV = objDeltaTJ.GetValues('V')
            intCol = 0
            intL = len(lstValuesTJPE)
            for l in range(intL):
                arrValues = np.sum(lstValuesTJPE[l])/np.sum(lstValuesTJV[l])
                plt.scatter(i,(arrValues-fltDatum),c=lstColours[0],label='Tripleline')
                lstAllTJs.append(arrValues-fltDatum)
                intCol +=1
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBPE = objDeltaGB.GetValues('PE')
            lstValuesGBV = objDeltaGB.GetValues('V')
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBPE)))
            for k in arrLengths:
                arrValues = np.sum(lstValuesGBPE[k])/np.sum(lstValuesGBV[k])
                lstAllGBs.append(arrValues-fltDatum)
                # if np.mean(arrValues)-fltDatum < 0.01:
                #      print(i,j, len(arrValues))
                if intCol ==0:
                    plt.scatter(i-0.1,(arrValues-fltDatum),c=lstColours[1],label = 'Cylinder')
                elif intCol ==1:
                    plt.scatter(i+0.1,(arrValues-fltDatum),c=lstColours[1],label = 'Cylinder')
                    
                elif intCol ==2:
                    plt.scatter(i,(arrValues-fltDatum),c=lstColours[2],label = 'CSL')
                #else:
                #plt.scatter(i,np.mean(k)-(4.05**3/4),c= 'b') #c =lstColours[intCol])
                    #print(i,j)
                    # plt.scatter(i/10,(np.mean(arrValues)-fltDatum),c='b',label = 'Grain')#c =lstColours[intCol])
                
                intCol +=1 
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Mean excess potential energy density in eV $\AA^{3}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstAllTJs, lstAllGBs
#%%
dctTJAxes = dict()
dctGBAxes = dict()
dctTJSigma = dict()
dctTJAxes['001'] = objTJ001
dctTJAxes['101'] = objTJ101
dctTJAxes['111'] = objTJ111
dctTJSigma['001'] = lstSigma001
dctTJSigma['101'] = lstSigma101
dctTJSigma['111'] = lstSigma111
#%%
lstAllTJs = []
lst101GBs = []
lstAllGrains = []
lstAxes = ['001','101','111']
for a in lstAxes:
    objTJ = dctTJAxes[a]
    lstSigma = dctTJSigma[a]
    j= 0
    while j  < len(lstSigma):
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
    #objStoreGB = objGB101.GetSigmaStore(j)
        lstTJs,lstGVs = PlotEnergyProportions(objStoreTJ,'TJ')
        lstAllTJs.extend(lstTJs)
        lstAllGrains.extend(lstGVs)
        j +=1
    # lst101TJs.extend(lstTJs)
    # lst101GBs.extend(lstGBs)
#%%
plt.hist(lstAllTJs,bins=15,alpha=0.5)
plt.hist(lstAllGrains,bins=15)
plt.show()
arrRows = np.where(np.array(lstAllGrains) < 0.05)[0]
print(len(arrRows)/len(lstAllGrains))
# %%
lst101TJs = []
lst101GBs = []
for j in lstSigma111:
    objStoreTJ = objTJ111.GetSigmaStore(j)
    objStoreGB = objGB111.GetSigmaStore(j)
    lstTJs,lstGBs = PlotEnergyDensities(objStoreGB,objStoreTJ)
    lst101TJs.extend(lstTJs)
    lst101GBs.extend(lstGBs)
#%%
plt.hist(lst101GBs,bins=25)
plt.hist(lst101TJs,bins=25,alpha =0.8)
plt.legend(['Grain boundary','Triple line'])
plt.show()
# %%
def PlotGBEnergyPerArea(inStoreGB, fltWidth: float):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36
    lstColours = ['black','darkolivegreen','darkgoldenrod']
    intDeltaMax = 10
    intDeltaMin = 0
    lstAllTJs = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBPE = objDeltaGB.GetValues('PE')
            lstValuesGBV = objDeltaGB.GetValues('V')
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBPE)))
            # if len(arrLengths) != 3:
            #     print(len(arrLengths),i,j)
            for k in arrLengths:
                arrValues = (np.sum(lstValuesGBPE[k])-fltDatum*(len(lstValuesGBPE[k])))/np.sum(lstValuesGBV[k])
                # if np.mean(arrValues)-fltDatum < 0.01:
                #      print(i,j, len(arrValues))
                if intCol ==0:
                    plt.scatter(i-0.1,arrValues*fltWidth,c=lstColours[1],label = 'Cylinder')
                elif intCol ==1:
                    plt.scatter(i+0.1,arrValues*fltWidth,c=lstColours[1],label = 'Cylinder')
                    
                elif intCol ==2:
                    if arrValues*fltWidth < 0.025:
                        print(i,j)
                    plt.scatter(i,arrValues*fltWidth,c=lstColours[2],label = 'CSL')
                #else:
                #plt.scatter(i,np.mean(k)-(4.05**3/4),c= 'b') #c =lstColours[intCol])
                    #print(i,j)
                    # plt.scatter(i/10,(np.mean(arrValues)-fltDatum),c='b',label = 'Grain')#c =lstColours[intCol])
                
                intCol +=1 
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Excess potential energy per unit area in eV $\AA^{2}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()

# %%
PlotGBEnergyPerArea(objStoreGB,6*4.05)
# %%
