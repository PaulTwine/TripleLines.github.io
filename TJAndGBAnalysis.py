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
dctColours = dict()
dctColours['Triple lines'] = 'gray'
dctColours['Cylinders'] = 'darkolivegreen'
dctColours['CSLs'] = 'darkgoldenrod'
dctColours['Grain boundaries'] = 'darkblue'
dctColours['Grains'] = 'saddlebrown'
dctColours['Bicrystal'] = 'purple'
strMeanPotentialEnergyPerVolume ='Mean excess potential energy density in eV \AA$^{-3}$'
strMeanExcessPotentialEnergyPerUnitArea = 'Mean excess potential energy per unit area in eV \AA$^{-2}$' 
strMeanExcessVolumePerAtom = 'Mean excess volume per atom in \AA$^{3}$'
strMeanPotentialEnergyPerAtom ='Mean excess potential energy per atom in eV \AA$^{-3}$'
strEnergyProportion = 'Proportion of the total excess energy'
#%%
dctLabels = dict()
dctLabels['Grains'] = 'Grains'
dctLabels['Triple lines'] = 'Triple lines'
dctLabels['Grain boundaries'] = 'Grain boundaries'
dctLabels['CSLs'] = 'CSL grain boundaries'
dctLabels['Cylinders'] = 'Cylindrical grain boundaries'
dctLabels['Bicrystal'] = 'Bicrystal grain boundaries'
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
dctCSL001 = dict()
strBase001CSL = '/home/p17992pt/csf4_scratch/BiCrystal/Axis001/Sigma'
for i in lstSigma001:
    arrValues = np.loadtxt(strBase001CSL + str(i) + '/Values.txt')
    dctCSL001[i] = (arrValues[:,1] +3.36*arrValues[:,3])/(2*arrValues[:,-1])
#%%
dctCSL101 = dict()
strBase101CSL = '/home/p17992pt/csf4_scratch/BiCrystal/Axis101/Sigma'
for i in lstSigma101:
    arrValues = np.loadtxt(strBase101CSL + str(i) + '/Values.txt')
    dctCSL101[i] = (arrValues[:,1] +3.36*arrValues[:,3])/(2*arrValues[:,-1])
#%%
dctCSL111 = dict()
strBase111CSL = '/home/p17992pt/csf4_scratch/BiCrystal/Axis111/Sigma'
for i in lstSigma111:
    arrValues = np.loadtxt(strBase111CSL + str(i) + '/Values.txt')
    dctCSL111[i] = (arrValues[:,1] +3.36*arrValues[:,3])/(2*arrValues[:,-1])
#%%
dctCSLs = dict()
dctCSLs['001'] = dctCSL001
dctCSLs['101'] = dctCSL101
dctCSLs['111'] = dctCSL111
#%%
dctTJAxes = dict()
dctGBAxes = dict()
dctSigma = dict()
dctTJAxes['001'] = objTJ001
dctTJAxes['101'] = objTJ101
dctTJAxes['111'] = objTJ111
dctSigma['001'] = lstSigma001
dctSigma['101'] = lstSigma101
dctSigma['111'] = lstSigma111
dctGBAxes['001'] = objGB001
dctGBAxes['101'] = objGB101
dctGBAxes['111'] = objGB111
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
def PlotEnergyProportions(inStoreGBorTJ: AS.DirStore, strType: str, indctColours: dict(),indctLabels: dict(),inDeltaMax=10):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    lstGBValues = []
    lstTJValues = []
    lstGrainValues = []
    for j in range(10): #directories
        objDir = inStoreGBorTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDelta = objDir.GetDeltaStore(i)
            lstTotalExcess = objDelta.GetValues('TE')[0]
            fltTotalExcess = lstTotalExcess[0]-fltDatum*lstTotalExcess[1]
            lstValues = objDelta.GetValues('GE')[0]
            arrValues = (lstValues[1]-fltDatum*lstValues[0])/fltTotalExcess
            lstGrainValues.append(arrValues)
            plt.scatter(i,arrValues,c=dctColours['Grains'],label = indctLabels['Grains'])
            if strType == 'TJ':
                lstValuesTJPE = objDelta.GetValues('PE')
                lstTripleLines = np.concatenate(lstValuesTJPE,axis=0)
                arrTJ = (np.sum(lstTripleLines)-fltDatum*len(lstTripleLines))/fltTotalExcess
                plt.scatter(i,arrTJ,c=indctColours['Triple lines'],label=indctLabels['Triple lines'])
                lstTJValues.append(arrTJ)
                lstGBValues.append(1-lstGrainValues[-1]-arrTJ)
            elif strType == 'GB':
                lstValuesGBPE = objDelta.GetValues('PE')
                lstGBTotalExcess = objDelta.GetValues('TE')[0]
                fltTotalExcess = lstGBTotalExcess[0]-fltDatum*lstGBTotalExcess[1]
                arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBPE)))
                lstCylinders = []
                for k in arrLengths[:2]:
                     lstCylinders.extend(lstValuesGBPE[k])
                #     lstAllValues.extend(lstCylinders)
                arrCylinder = (np.sum(lstCylinders)-fltDatum*len(lstCylinders))/fltTotalExcess
                plt.scatter(i,arrCylinder, c=indctColours['Cylinders'],label = indctLabels['Cylinders'])
                arrCSL = (np.sum(lstValuesGBPE[arrLengths[2]])-fltDatum*len(lstValuesGBPE[arrLengths[2]]))/fltTotalExcess
                plt.scatter(i,arrCSL,c=indctColours['CSLs'],label = indctLabels['CSLs'])
                lstGBValues.append(arrCSL+arrCylinder)    
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Mean excess potential energy proportion'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstTJValues,lstGBValues,lstGrainValues 
#%%
def PlotExcessEnergyPerAtom(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(), indctLabels: dict(), inDeltaMax = 10):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    fltDatum = -3.36
    lstAllTJs = []
    lstAllCSLs = []
    lstAllCylinders = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j)
        objDirTJ = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirTJ.GetDeltaStore(i)
            lstValuesTJV = objDeltaTJ.GetValues('PE')
            intCol = 0
            intL = len(lstValuesTJV)
            for l in range(intL):
                arrValues = np.mean(lstValuesTJV[l])-fltDatum
                plt.scatter(i,arrValues,c=indctColours['Triple lines'],label=indctLabels['Triple lines'])
                lstAllTJs.append(arrValues)
                intCol +=1
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBV = objDeltaGB.GetValues('PE')
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBV)))
            for k in arrLengths:
                arrValues = np.mean(lstValuesGBV[k])-fltDatum
                # if np.mean(arrValues)-fltDatum < 0.01:
                #      print(i,j, len(arrValues))
                if intCol ==0:
                    plt.scatter(i-0.1,arrValues,c=indctColours['Cylinders'],label = indctLabels['Cylinders'])
                    lstAllCylinders.append(arrValues)
                elif intCol ==1:
                    plt.scatter(i+0.1,arrValues,c=indctColours['Cylinders'],label = indctLabels['Cylinders'])
                    lstAllCylinders.append(arrValues)
                elif intCol ==2:
                    plt.scatter(i,arrValues,c=indctColours['CSLs'],label = indctLabels['CSLs'])     
                    lstAllCSLs.append(arrValues)
                intCol +=1 
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Excess potential energy per atom in eV'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstAllTJs, lstAllCylinders,lstAllCSLs

#%%
def PlotHydrostaticStress(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(), inDeltaMax = 10):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    lstAllTJs = []
    lstAllCSLs = []
    lstAllCylinders = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j)
        objDirTJ = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirTJ.GetDeltaStore(i)
            lstValuesTJV = objDeltaTJ.GetValues('S')
            intCol = 0
            intL = len(lstValuesTJV)
            for l in range(intL):
                arrValues = np.mean(lstValuesTJV[l])
                plt.scatter(i,arrValues,c=indctColours['Triple line'],label='Tripleline')
                lstAllTJs.append(arrValues)
                intCol +=1
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBV = objDeltaGB.GetValues('S')
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBV)))
            for k in arrLengths:
                arrValues = np.mean(lstValuesGBV[k])
                # if np.mean(arrValues)-fltDatum < 0.01:
                #      print(i,j, len(arrValues))
                if intCol ==0:
                    plt.scatter(i-0.1,arrValues,c=indctColours['Cylinder'],label = 'Cylinder')
                    lstAllCylinders.append(arrValues)
                elif intCol ==1:
                    plt.scatter(i+0.1,arrValues,c=indctColours['Cylinder'],label = 'Cylinder')
                    lstAllCylinders.append(arrValues)
                elif intCol ==2:
                    plt.scatter(i,arrValues,c=indctColours['CSL'],label = 'CSL')     
                    lstAllCSLs.append(arrValues)
                intCol +=1 
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Excess volume per atom in $\AA^{3}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstAllTJs, lstAllCylinders,lstAllCSLs

#%%
def PlotExcessVolumes(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(), inDeltaMax = 10):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = 4.05**3/4
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    lstAllTJs = []
    lstAllGBs = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j)
        objDirTJ = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirTJ.GetDeltaStore(i)
            lstValuesTJV = objDeltaTJ.GetValues('V')
            intCol = 0
            intL = len(lstValuesTJV)
            for l in range(intL):
                arrValues = np.mean(lstValuesTJV[l])-fltDatum
                plt.scatter(i,arrValues,c=indctColours['Triple line'],label='Tripleline')
                lstAllTJs.append(arrValues)
                intCol +=1
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBV = objDeltaGB.GetValues('V')
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBV)))
            for k in arrLengths:
                arrValues = np.mean(lstValuesGBV[k])-fltDatum
                lstAllGBs.append(arrValues)
                # if np.mean(arrValues)-fltDatum < 0.01:
                #      print(i,j, len(arrValues))
                if intCol ==0:
                    plt.scatter(i-0.1,arrValues,c=indctColours['Cylinder'],label = 'Cylinder')
                elif intCol ==1:
                    plt.scatter(i+0.1,arrValues,c=indctColours['Cylinder'],label = 'Cylinder')
                    
                elif intCol ==2:
                    plt.scatter(i,arrValues,c=indctColours['CSL'],label = 'CSL')     
                intCol +=1 
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Excess volume per atom in $\AA^{3}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstAllTJs, lstAllGBs
# %%
def PlotEnergyDensities(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(),indctLabels: dict(), inDeltaMax = 10):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36/(4.05**3/4)
    lstColours = ['gray','darkolivegreen','darkgoldenrod']
    intDeltaMax = inDeltaMax
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
                plt.scatter(i,(arrValues-fltDatum),c=indctColours['Triple lines'],label=indctLabels['Triple lines'])
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
                    plt.scatter(i-0.1,(arrValues-fltDatum),c=indctColours['Cylinders'],label =indctLabels['Cylinders'])
                elif intCol ==1:
                    plt.scatter(i+0.1,(arrValues-fltDatum),c=indctColours['Cylinders'],label =indctLabels['Cylinders'])
                    
                elif intCol ==2:
                    plt.scatter(i,(arrValues-fltDatum),c=indctColours['CSLs'],label = indctLabels['CSLs'])
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
def PlotGBEnergyPerArea(inStoreGB, fltWidth: float, inDeltaMax: int,indctColours: dict(),indctLabels: dict(),arrCSLValues = None):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    lstBiCrystal = []
    lstCSL = []
    lstCylinder = []
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
                    plt.scatter(i-0.1,arrValues*fltWidth,c=indctColours['Cylinders'],label = indctLabels['Cylinders'])
                    lstCylinder.append(arrValues*fltWidth)
                elif intCol ==1:
                    plt.scatter(i+0.1,arrValues*fltWidth,c=indctColours['Cylinders'],label = indctLabels['Cylinders'])
                    lstCylinder.append(arrValues*fltWidth)
                elif intCol ==2:
                    #if arrValues*fltWidth < 0.025:
                    #    print(i,j)
                    plt.scatter(i,arrValues*fltWidth,c=indctColours['CSLs'],label = indctLabels['CSLs'])
                    arrCSL = arrValues*fltWidth
                    lstCSL.append(arrCSL)
                #else:
                #plt.scatter(i,np.mean(k)-(4.05**3/4),c= 'b') #c =lstColours[intCol])
                    #print(i,j)
                    # plt.scatter(i/10,(np.mean(arrValues)-fltDatum),c='b',label = 'Grain')#c =lstColours[intCol])
                intCol +=1
            if arrCSLValues is not None: 
                plt.scatter(i,arrCSLValues[i],c=indctColours['Bicrystal'], label=indctLabels['Bicrystal'])
                lstBiCrystal.append(arrCSLValues[i])    
                
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
    return lstCylinder,lstCSL,lstBiCrystal
#%%
def PlotTJLineTension(inStoreTJ, fltRadius: float, inDeltaMax: int,indctColours: dict(),indctLabels: dict()):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = -3.36
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    lstBiCrystal = []
    lstCSL = []
    lstCylinder = []
    for j in range(10): #directories
        objDirGB = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirGB.GetDeltaStore(i)
            lstValuesTJPE = objDeltaTJ.GetValues('PE')
            lstValuesTJV = objDeltaTJ.GetValues('V')
            arrValues = np.sum(lstValuesTJPE)/np.sum(lstValuesTJV)*np.pi*fltRadius**2
            plt.scatter(i,arrValues,c=indctColours['Triple lines'], label=indctLabels['Triple lines'])    
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Triple line tension in eV $\AA^{-1}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstCylinder,lstCSL,lstBiCrystal

#%%

#%%
##Energy proportions
lstAllTJ1 = []
lstAllGB1 = []
lstAllG1 = []
lstAllGB2 = []
lstAllG2 = []
lstAxes = ['001','101','111']
for a in lstAxes:
    objTJ = dctTJAxes[a]
    objGB = dctGBAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
        objStoreGB = objGB.GetSigmaStore(lstSigma[j])
        lstTJ1,lstGB1,lstG1 = PlotEnergyProportions(objStoreTJ,'TJ',dctColours,dctLabels,8)
        lstTJ2,lstGB2, lstG2 = PlotEnergyProportions(objStoreGB, 'GB',dctColours,dctLabels,8)
        lstAllTJ1.extend(lstTJ1)
        lstAllGB1.extend(lstGB1)
        lstAllG1.extend(lstG1)
        lstAllGB2.extend(lstGB2)
        lstAllG2.extend(lstG2)
        j +=1
    # lst101TJs.extend(lstTJs)
    # lst101GBs.extend(lstGBs)
#%%

#plt.hist(lstAllTJs,bins=15,color='darkgrey')
#plt.hist(np.array(lstAllGrainsTJ)-np.array(lstAllGrainsGB),color='grey',bins=20,alpha=0.75)
#plt.hist(lstAllGrainsTJ,bins=15,color='darkred')
plt.hist(lstAllG1,bins=15,color=dctColours['Grains'])
plt.hist(lstAllGB1,bins=15,color=dctColours['Grain boundaries'])

#plt.legend(['Triple line', 'Grains'])
plt.legend([dctLabels['Grains'], dctLabels['Grain boundaries']])
plt.xticks(np.array(list(range(11)))/10)
plt.xlabel('Proportion of the total excess energy')
plt.show()
print(np.mean(lstAllTJ1), np.std(lstAllTJ1))
print(np.mean(lstAllGB2), np.std(lstAllGB2))
#%%
plt.hist(lstAllG1,bins=15,density=True,color=dctColours['Grains'])
plt.hist(lstAllTJ1,bins=15,density=True,color=dctColours['Triple lines'])
plt.hist(lstAllGB1,bins=15,density=True,color=dctColours['Grain boundaries'])
#plt.legend(['Triple line', 'Grains'])
plt.legend([dctLabels['Grains'], dctLabels['Triple lines'], dctLabels['Grain boundaries']])
plt.xticks(np.array(list(range(11)))/10)
plt.xlabel('Proportion of the total excess energy')
plt.show()
print(np.mean(lstAllG1), np.std(lstAllG1))    
print(np.mean(lstAllTJ1),np.std(lstAllTJ1))
print(np.mean(lstAllGB1), np.std(lstAllGB1))
# arrRows = np.where(np.array(lstAllGrainsTJ) < 0.05)[0]
# print(len(arrRows)/len(lstAllGrainsTJ))
# %%
##Energy densities
lstAllTJs = []
lstAllGBs = []
lstAxes = ['001','101','111']
for a in lstAxes:
    objTJ = dctTJAxes[a]
    objGB = dctGBAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
        objStoreGB = objGB.GetSigmaStore(lstSigma[j])
        lstTJs,lstGBs = PlotEnergyDensities(objStoreGB,objStoreTJ,dctColours,8)
        lstAllTJs.extend(lstTJs)
        lstAllGBs.extend(lstGBs)
        j +=1
#%%
plt.hist(lstAllGBs,bins=25,color=dctColours['Both GBs'],density=True)
plt.hist(lstAllTJs,bins=25,color = dctColours['Triple line'],alpha =0.8,density=True)
plt.xlabel(strMeanPotentialEnergyPerVolume)
plt.legend(['Grain boundary','Triple line'])
plt.show()
print(np.mean(lstAllTJs),np.std(lstAllTJs))
print(np.mean(lstAllGBs),np.std(lstAllGBs))
#%%
##Hydrostatic stress
lstAllTJs = []
lstAllCylinders = []
lstAllCSLs = []
lstAxes = ['001','101','111']
for a in lstAxes:
    objTJ = dctTJAxes[a]
    objGB = dctGBAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
        objStoreGB = objGB.GetSigmaStore(lstSigma[j])
        lstTJs,lstCylinders,lstCSLs = PlotHydrostaticStress(objStoreGB,objStoreTJ,dctColours,8)
        lstAllTJs.extend(lstTJs)
        lstAllCylinders.extend(lstCylinders)
        lstAllCSLs.extend(lstCSLs)
        j +=1
#%%
lstAllGBs= []
lstAllGBs.extend(lstAllCylinders)
lstAllGBs.extend(lstAllCSLs)
plt.hist(lstAllGBs,bins=25,density=True,color=dctColours['Both GBs'])
plt.hist(lstAllTJs,bins=25,color = dctColours['Triple line'],alpha =0.8,density=True)
plt.xlabel('Mean hydrostatic stress per atom in eV \AA$^{-3}$')
plt.legend(['Grain boundaries','Triple lines'])
plt.show()

#%%
plt.hist(lstAllCylinders,bins=25,density=True,color=dctColours['Cylinder'])
plt.hist(lstAllCSLs,bins=25,density=True,color=dctColours['CSL'])
plt.hist(lstAllTJs,bins=25,color = dctColours['Triple line'],alpha =0.8,density=True)
plt.xlabel('Mean hydrostatic stress per atom in eV \AA$^{-3}$')
plt.legend(['CSL','Cylindrical','Triple line'])
plt.show()
print(np.mean(lstAllTJs)*10e-6,np.mean(lstAllGBs)*10e-6,np.mean(lstAllCylinders))
print(np.std(lstAllTJs)*10e-6,np.std(lstAllGBs)*10e-6)
#%%
arrCSLs = np.array(lstAllCSLs)
arrCylinders = np.array(lstAllCylinders)
lstDifferences = []
for i in range(len(arrCSLs)):
    lstDifferences.append(arrCSLs[i]-arrCylinders[2*i])
    lstDifferences.append(arrCSLs[i]-arrCylinders[2*i+1])
#%%
plt.hist(np.array(lstDifferences)*10**(-6))
plt.show()
# %%
##Excess volumes
lstAllTJs = []
lstAllGBs = []
lstAxes = ['001','101','111']
for a in lstAxes:
    objTJ = dctTJAxes[a]
    objGB = dctGBAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
        objStoreGB = objGB.GetSigmaStore(lstSigma[j])
        lstTJs,lstGBs = PlotExcessVolumes(objStoreGB,objStoreTJ,dctColours,8)
        lstAllTJs.extend(lstTJs)
        lstAllGBs.extend(lstGBs)
        j +=1
#%%
plt.hist(lstAllGBs,bins=25,density=True,color=dctColours['Both GBs'])
plt.hist(lstAllTJs,bins=25,color = dctColours['Triple line'],alpha =0.8,density=True)
plt.xlabel('Excess volume per atom in \AA$^{3}$')
plt.legend(['Grain boundary','Triple line'])
plt.show()
#%%
#%%
###Excess energy per atom 
lstAllTJ = []
lstAllCylinder = []
lstAllCSL = []
lstAxes = ['001','101','111']
for a in lstAxes:
    dctCSL = dctCSLs[a]
    objGB = dctGBAxes[a]
    objTJ = dctTJAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        print(dctCSL[lstSigma[j]])
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
        objStoreGB = objGB.GetSigmaStore(lstSigma[j])
        lstTJ,lstCylinder,lstCSL = PlotExcessEnergyPerAtom(objStoreGB,objStoreTJ,dctColours,dctLabels,8)
        lstAllCSL.extend(lstCSL)
        lstAllCylinder.extend(lstCylinder)
        lstAllTJ.extend(lstTJ)
        j +=1
#inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(), inDeltaMax = 10)
#%%
lstAllGBs = []
lstAllGBs.extend(lstAllCSL)
lstAllGBs.extend(lstAllCylinder)
plt.hist(lstAllGBs,density=True,color=dctColours['Grain boundaries'], bins=20)
plt.hist(lstAllTJ,density=True,color=dctColours['Triple lines'], bins=20,alpha=0.8)
plt.xlabel(strMeanPotentialEnergyPerAtom)
plt.legend([dctLabels['Triple lines'], dctLabels['Grain boundaries']])
plt.show()
print(np.mean(np.array(lstAllGBs)), np.mean(np.array(lstAllTJ)))
# %%
###Energy per unit area and compared with CSL bicrystal data
lstAllCSL = []
lstAllCylinder = []
lstAllBiCrystal = []
lstAxes = ['001','101','111']
for a in lstAxes:
    dctCSL = dctCSLs[a]
    objGB = dctGBAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        print(dctCSL[lstSigma[j]])
        objStoreGB = objGB.GetSigmaStore(lstSigma[j])
        lstCylinder,lstCSL,lstBiCrystal = PlotGBEnergyPerArea(objStoreGB,6*4.05,8,dctColours,dctLabels,dctCSL[lstSigma[j]])
        lstAllCSL.extend(lstCSL)
        lstAllCylinder.extend(lstCylinder)
        lstAllBiCrystal.extend(lstBiCrystal)
        j +=1

# %%
plt.hist(lstAllBiCrystal,density=True,color=dctColours['Bicrystal'], bins=20)
plt.hist(lstAllCSL,density=True,color=dctColours['Grain boundaries'], bins=20,alpha=0.8)
plt.xlabel(strMeanExcessPotentialEnergyPerUnitArea)
plt.legend(['Constant width approximation', 'Gibbsian approach'])
plt.show()
print(np.mean(np.array(lstAllCSL)), np.mean(np.array(lstAllBiCrystal)))
#%%
lstAllGBs= []
lstAllGBs.extend(lstAllCSL)
lstAllGBs.extend(lstAllCylinder)
plt.hist(lstAllGBs, density = True, color=dctColours['Grain boundaries'], bins =25)
plt.legend([dctLabels['Grain boundaries']])
plt.xlabel(strMeanExcessPotentialEnergyPerUnitArea)
# %%
arrErrors = (np.array(lstAllCSL))/np.array(lstAllBiCrystal)
plt.hist(arrErrors,bins=40,density =True)
plt.xlim([0.85,1.1])
plt.show()
print(np.mean(arrErrors), np.std(arrErrors))
print(np.corrcoef(np.array(lstAllBiCrystal),np.array(lstAllCSL)))
# %%
print(len(np.where(arrErrors < 0.85)[0])+len(np.where(arrErrors > 1.05)[0]))
# %%
##Line tensions
for a in lstAxes:
    objTJ = dctTJAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
        PlotTJLineTension(objStoreTJ,3*4.05,8,dctColours,dctLabels)
        j +=1

# %%
