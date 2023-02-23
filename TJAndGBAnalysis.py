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
objStoreTJ = AS.PopulateSigmaStoreByFile(31, np.array([1,1,1]),'/home/p17992pt/csf4_scratch/TJ/Axis111/TJSigma31/','TJ',['GE','PE','V','S'])
#objDirStore = objStoreTJ.GetDirStore(0)
# objDeltaStore = objDirStoreTJ.GetDeltaStore(0)
# print(objDeltaStore.GetValues('PE')[1])
#%%
objStoreGB = AS.PopulateSigmaStoreByFile(31, np.array([1,1,1]),'/home/p17992pt/csf4_scratch/TJ/Axis111/TJSigma31/','GB',['GE','PE','V','S'])
#objDirStoreGB = objStore.GetDirStore(0)
#objDeltaStoreGB = objDirStore.GetDeltaStore(0)
#print(objDeltaStore.GetValues('PE')[1])
#%%
fig, ax = plt.subplots()
#%%
##Per atom graphs of excess PE, excess volume and hydrostatc stress
strDeltaAxis = r'$\bm{\delta_{i}}$'
strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
fltDatum = 4.05**3/4 
fltDatum = -3.36
#fltDatum = 0
strType = 'V' 
#strType = 'PE'
#strType = 'S'
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
    strYlabel = 'Mean excess volume per atom in $\AA^{3}$ per atom'   
elif strType == 'PE':
    strYlabel = 'Mean excess potential energy per atom in eV per atom'
elif strType =='S':
    strYlabel = 'Mean hydrostatic stress per atom in eV $\AA^{-3}$ per atom'
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
# %%
def PlotEnergyDensities(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36/(4.05**3/4)
    lstColours = ['black','darkolivegreen','darkgoldenrod']
    intDeltaMax = 10
    intDeltaMin = 0
    lstAllTJs = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j)
        objDirTJ = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirTJ.GetDeltaStore(i)
            lstValuesTJPE = objDeltaTJ.GetValues('PE')
            lstValuesTJV = objDeltaTJ.GetValues('V')
            intCol = 0
            for l in range(1,len(lstValuesTJ)):
                plt.scatter(i,(np.sum(lstValuesTJPE[l])/np.sum(lstValuesTJV[l])-fltDatum),c=lstColours[0],label='Tripleline')
                intCol +=1
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBPE = objDeltaGB.GetValues('PE')
            lstValuesGBV = objDeltaGB.GetValues('V')
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBPE)))
            # if len(arrLengths) != 3:
            #     print(len(arrLengths),i,j)
            for k in arrLengths:
                arrValues = np.sum(lstValuesGBPE[k])/np.sum(lstValuesGBV[k])
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

# %%
PlotEnergyDensities(objStoreGB,objStoreTJ)
# %%
