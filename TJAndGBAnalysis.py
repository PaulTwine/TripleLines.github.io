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
import MiscFunctions as mf
#%%
strRoot = '/home/p17992pt/csf4_scratch/TJ/Axis001/TJSigma5/0/'
objData = LT.LAMMPSData(strRoot + 'TJ0P.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objLT = objData.GetTimeStepByIndex(-1)
print(objLT.GetColumnNames())
#%%
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 20})
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
strMeanPotentialEnergyPerAtom ='Mean excess potential energy per atom in eV '
strEnergyProportion = 'Proportion of the total excess energy'
strMeanExcessPotentialEnergyPerUnitAreaGibbsian = 'Gibbsian calculation of $\gamma$ in eV \AA$^{-2}$' 
strMeanExcessPotentialEnergyPerUnitAreaConstantWidth = 'Constant width approximation of $\gamma$ in eV \AA$^{-2}$' 

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
lstAxes = ['001','101','111']
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
def PlotHydrostaticStress(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(),indctLabels: dict(), inDeltaMax = 10, blnAdjustDatum = False):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    fltDatum = 0
    lstAllTJs = []
    lstAllCSLs = []
    lstAllCylinders = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j)
        objDirTJ = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirTJ.GetDeltaStore(i)
            lstValuesTJS = objDeltaTJ.GetValues('S')
            lstValuesTJV = objDeltaTJ.GetValues('V')
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBS = objDeltaGB.GetValues('S')
            lstValuesGBV = objDeltaGB.GetValues('V')
            if lstValuesGBS[-1] == lstValuesGBS[-2]:
                lstValuesGBS.pop()
            if lstValuesTJS[-1] == lstValuesTJS[-2]:
                lstValuesTJS.pop()
            if blnAdjustDatum:
                lstGBGrainValues = objDeltaGB.GetValues('GE')[0]
                fltDatum = lstGBGrainValues[-1]/(3*lstGBGrainValues[0])
                #Stress is given in pressure*volume units in LAMMPS
                # lstStress = []
                # lstStress.extend(mf.FlattenList(lstValuesGBS))
                # intAtoms = lstGBGrainValues[0] + len(lstStress)
                # lstStress.append(lstGBGrainValues[-1])
                # fltDatum = np.sum(lstStress)/intAtoms
            intCol = 0
            intL = len(lstValuesTJV)
            for l in range(intL):
                #arrTrueStress = np.divide(lstValuesTJS[l],lstValuesTJV[l])/3
                #arrTrueStress = np.sum(lstValuesTJS[l])/(3*np.sum(lstValuesTJV[l]))
                #arrValues = np.mean(lstValuesTJS[l])/3 -fltDatum
                arrValues = np.mean(np.divide(lstValuesTJS[l],lstValuesTJV[l])/3)
                #arrValues = np.mean(arrTrueStress)-fltDatum
                #arrValues = np.mean(lstValuesTJS[l])-fltDatum
                plt.scatter(i,arrValues,c=indctColours['Triple lines'],label=indctLabels['Triple lines'])
                lstAllTJs.append(arrValues)
                intCol +=1
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBS)))
            for k in arrLengths:
                arrTrueStress = np.divide(lstValuesGBS[k],lstValuesGBV[k])/3
                arrValues = np.mean(arrTrueStress)
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
    strYlabel = 'Hydrostatic stress per atom in eV $\AA^{-3}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstAllTJs, lstAllCylinders,lstAllCSLs

#%%
def PlotExcessVolumes(inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(),indctLabels: dict(), inDeltaMax = 10,blnAdjustDatum = False):
    strDeltaAxis = r'$\bm{\delta_{i}}$'
    strDMinAxis = r'$10d_{\mathrm{min}}/r_0$'
    fltDatum = 4.05**3/4
    intDeltaMax = inDeltaMax
    intDeltaMin = 0
    lstAllTJs = []
    lstGBCSLs = []
    lstGBCylinders = []
    for j in range(10): #directories
        objDirGB = inStoreGB.GetDirStore(j)
        objDirTJ = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirTJ.GetDeltaStore(i)
            lstValuesTJV = objDeltaTJ.GetValues('V')
            objDeltaGB = objDirGB.GetDeltaStore(i)
            lstValuesGBV = objDeltaGB.GetValues('V')
            if blnAdjustDatum:
                lstGBGrainValues = objDeltaGB.GetValues('GE')[0]
                lstVolume = []
                lstVolume.extend(mf.FlattenList(lstValuesGBV))
                intAtoms = lstGBGrainValues[0] + len(lstVolume)
                lstVolume.append(lstGBGrainValues[2])
                fltDatum = np.sum(lstVolume)/intAtoms
            intCol = 0
            intL = len(lstValuesTJV)
            for l in range(intL):
                arrValues = np.mean(lstValuesTJV[l])-fltDatum
                if arrValues < 0:
                    print(i,j)
                plt.scatter(i,arrValues,c=indctColours['Triple lines'],label=indctLabels['Triple lines'])
                lstAllTJs.append(arrValues)
                intCol +=1
            intCol = 0
            arrLengths  = np.argsort(list(map(lambda x: len(x),lstValuesGBV)))
            for k in arrLengths:
                arrValues = np.mean(lstValuesGBV[k])-fltDatum
                #lstAllGBs.append(arrValues)
                # if np.mean(arrValues)-fltDatum < 0.01:
                #      print(i,j, len(arrValues))
                if intCol ==0:
                    plt.scatter(i-0.1,arrValues,c=indctColours['Cylinders'],label = indctLabels['Cylinders'])
                    lstGBCylinders.append(arrValues)
                elif intCol ==1:
                    plt.scatter(i+0.1,arrValues,c=indctColours['Cylinders'],label = indctLabels['Cylinders'])
                    lstGBCylinders.append(arrValues)
                elif intCol ==2:
                    plt.scatter(i,arrValues,c=indctColours['CSLs'],label = indctLabels['CSLs'])
                    lstGBCSLs.append(arrValues)     
                intCol +=1 
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Mean excess volume per atom in $\AA^{3}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstAllTJs, lstGBCylinders,lstGBCSLs
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
    lstAllCylinders = []
    lstAllCSLs = []
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
                    lstAllCylinders.append(arrValues-fltDatum)
                elif intCol ==1:
                    plt.scatter(i+0.1,(arrValues-fltDatum),c=indctColours['Cylinders'],label =indctLabels['Cylinders'])
                    lstAllCylinders.append(arrValues-fltDatum)
                elif intCol ==2:
                    plt.scatter(i,(arrValues-fltDatum),c=indctColours['CSLs'],label = indctLabels['CSLs'])
                    lstAllCSLs.append(arrValues-fltDatum)
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
    return lstAllCylinders,lstAllCSLs, lstAllTJs
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
    plt.legend(*d,fontsize=12,loc='upper right')
    strYlabel = '$\gamma$ in eV $\AA^{-2}$'   
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
    lstTJsinFours = []
    for j in range(10): #directories
        objDirGB = inStoreTJ.GetDirStore(j) 
        for i in range(intDeltaMin,intDeltaMax): #delta values
            objDeltaTJ = objDirGB.GetDeltaStore(i)
            lstValuesTJPE = objDeltaTJ.GetValues('PE')
            lstValuesTJV = objDeltaTJ.GetValues('V')
            lstGroup = []
            if len(lstValuesTJPE) != 4:
                print(j,i,objDeltaTJ)
            for t in range(len(lstValuesTJPE)):
                arrValues = np.sum(lstValuesTJPE[t]-np.ones(len(lstValuesTJPE[t]))*fltDatum)/(np.sum(lstValuesTJV[t]))*np.pi*fltRadius**2
                plt.scatter(i,arrValues,c=indctColours['Triple lines'], label=indctLabels['Triple lines'])
                lstGroup.append(arrValues)
            lstTJsinFours.append(lstGroup)
    # arrDeltaMeanTJ = np.mean(lstAllTJs)-fltDatum
    #plt.scatter(arrDeltaMeanTJ, i/10, c='black', marker='x')
    a = plt.gca().get_legend_handles_labels()  # a = [(h1 ... h2) (l1 ... l2)]  non unique
    b = {l:h for h,l in zip(*a)}        # b = {l1:h1, l2:h2}             unique
    c = [*zip(*b.items())]              # c = [(l1 l2) (h1 h2)]
    d = c[::-1]                         # d = [(h1 h2) (l1 l2)]
    plt.legend(*d)
    strYlabel = 'Line tension in eV $\AA^{-1}$'   
    plt.xticks(np.array(list(range(intDeltaMin,intDeltaMax))))
    plt.ylabel(strYlabel)
    plt.xlabel(strDMinAxis)
    plt.tight_layout()
    plt.show()
    return lstTJsinFours

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
lstBinsEP = (np.linspace(0,1,100)).tolist()
#plt.hist(lstAllTJs,bins=15,color='darkgrey')
#plt.hist(np.array(lstAllGrainsTJ)-np.array(lstAllGrainsGB),color='grey',bins=20,alpha=0.75)
#plt.hist(lstAllGrainsTJ,bins=15,color='darkred')
plt.hist(lstAllG2,bins=lstBinsEP,density=True,color=dctColours['Grains'])
plt.hist(lstAllGB2,bins=lstBinsEP,density=True,color=dctColours['Grain boundaries'])
#plt.hist(lstAllTJ1,bins=15,color='darkred')
#plt.legend(['Triple line', 'Grains'])
plt.legend([dctLabels['Grains'], dctLabels['Grain boundaries']])
plt.ylim([0,60])
plt.xticks(np.array(list(range(11)))/10)
plt.xlabel('Proportion of the total excess energy')
plt.show()
print(np.mean(lstAllG2), np.std(lstAllG2))
print(np.mean(lstAllGB2), np.std(lstAllGB2))
#%%
plt.hist(lstAllG1,bins=lstBinsEP,density=True,color=dctColours['Grains'])
plt.hist(lstAllGB1,bins=lstBinsEP,density=True,color=dctColours['Grain boundaries'])
plt.hist(lstAllTJ1,bins=lstBinsEP,density=True,color=dctColours['Triple lines'])
#plt.legend(['Triple line', 'Grains'])
plt.legend([dctLabels['Grains'], dctLabels['Grain boundaries'],dctLabels['Triple lines']])
plt.xticks(np.array(list(range(11)))/10)
plt.xlabel('Proportion of the total excess energy')
plt.ylim([0,60])
plt.show()
print(np.mean(lstAllG1), np.std(lstAllG1))    
print(np.mean(lstAllTJ1),np.std(lstAllTJ1))
print(np.mean(lstAllGB1), np.std(lstAllGB1))
# arrRows = np.where(np.array(lstAllGrainsTJ) < 0.05)[0]
# print(len(arrRows)/len(lstAllGrainsTJ))
# %%
##Energy densities
lstAllTJEDs = []
lstAllGBEDs = []
lstAllCylinderEDs = []
lstAllCSLEDs = []
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
        lstCylinders,lstCSLs,lstTJs= PlotEnergyDensities(objStoreGB,objStoreTJ,dctColours,dctLabels,8)
        lstAllTJEDs.extend(lstTJs)
        lstAllCylinderEDs.extend(lstCylinders)
        lstAllCSLEDs.extend(lstCSLs)
        j +=1
#%%
lstBinsED = (np.linspace(0,12,25)/1000).tolist()
plt.xlim([0,0.012])
plt.ylim([0,650])
lstAllGBEDs = []
lstAllGBEDs.extend(lstAllCylinderEDs)
lstAllGBEDs.extend(lstAllCSLEDs)
lstAllGBEDs.extend(lstAllCSLEDs)
plt.hist(lstAllGBEDs,bins=lstBinsED,color=dctColours['Grain boundaries'],density=True)
plt.hist(lstAllTJEDs,bins=lstBinsED,color = dctColours['Triple lines'],alpha =0.8, density=True)
plt.xlabel(strMeanPotentialEnergyPerVolume)
plt.legend([dctLabels['Grain boundaries'],dctLabels['Triple lines']])
plt.show()
print(np.mean(lstAllTJEDs),np.std(lstAllTJEDs))
print(np.mean(lstAllGBEDs),np.std(lstAllGBEDs))
#%%
lstBinsED = (np.linspace(0,12,25)/1000).tolist()
plt.xlim([0,0.012])
plt.ylim([0,650])
plt.xlabel(strMeanPotentialEnergyPerVolume)
plt.hist(lstAllCSLEDs,bins=lstBinsED,color = dctColours['CSLs'],alpha =1,density=True)
plt.legend([dctLabels['CSLs']])
plt.show()
#%%
plt.xlim([0,0.012])
plt.ylim([0,650])
plt.xlabel(strMeanPotentialEnergyPerVolume)
plt.hist(lstAllCylinderEDs,bins=lstBinsED,color=dctColours['Cylinders'],alpha = 1,density=True)
plt.legend([dctLabels['Cylinders']])
plt.show()
#%%
plt.xlim([0,0.012])
plt.ylim([0,650])
plt.xlabel(strMeanPotentialEnergyPerVolume)
plt.hist(lstAllTJEDs,bins=lstBinsED,color = dctColours['Triple lines'],alpha =1,density=True)
plt.legend([dctLabels['Triple lines']])
plt.show()
#%%
print(np.mean(lstAllTJEDs),np.std(lstAllTJEDs))
print(np.mean(lstAllCSLEDs),np.std(lstAllCSLEDs))
print(np.mean(lstAllCylinderEDs),np.std(lstAllCylinderEDs))
print(np.mean(lstAllGBEDs),np.std(lstAllGBEDs))
#%%
l
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
        lstTJs,lstCylinders,lstCSLs = PlotHydrostaticStress(objStoreGB,objStoreTJ,dctColours,dctLabels,8,False)
        lstAllTJs.extend(lstTJs)
        lstAllCylinders.extend(lstCylinders)
        lstAllCSLs.extend(lstCSLs)
        j +=1
#%%
lstAllGBH= []
lstAllGBH.extend(lstAllCylinders)
lstAllGBH.extend(lstAllCSLs)
plt.hist(lstAllGBH,bins=20,density=True,color=dctColours['Grain boundaries'])
plt.hist(lstAllTJs,bins=20,color = dctColours['Triple lines'],alpha =0.8,density=True)
plt.xlabel('Mean hydrostatic stress per atom in eV \AA$^{-3}$')
plt.legend([dctLabels['Grain boundaries'],dctLabels['Triple lines']])
plt.show()

#%%
lstTJBins = (np.linspace(0,8,25)/10).tolist()
lstGBBins = (np.linspace(0,4,13)/10).tolist()
plt.hist(lstAllCylinders,bins=25,density=True,color=dctColours['Cylinders'])
plt.hist(lstAllCSLs,bins=25,density=True,color=dctColours['CSLs'])
plt.hist(lstAllTJs,bins=25,color = dctColours['Triple lines'],alpha =0.8,density=True)
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
lstAllTJVs = []
lstAllCSLVs = []
lstAllCylinderVs = []
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
        lstTJs,lstCylinders,lstCSLs = PlotExcessVolumes(objStoreGB,objStoreTJ,dctColours,dctLabels,8,True)
        lstAllTJVs.append(lstTJs)
        lstAllCSLVs.append(lstCSLs)
        lstAllCylinderVs.append(lstCylinders)
        j +=1
#%%
lstTJBins = (np.linspace(0,8,25)/10).tolist()
lstGBBins = (np.linspace(0,4,13)/10).tolist()
lstAllGBVs = []
lstAllGBVs.extend(mf.FlattenList(lstAllCSLVs))
lstAllGBVs.extend(mf.FlattenList(lstAllCSLVs))
lstAllGBVs.extend(mf.FlattenList(lstAllCylinderVs))
plt.hist(lstAllGBVs,bins=lstGBBins,density=True,color=dctColours['Grain boundaries'])
plt.hist(mf.FlattenList(lstAllTJVs),color = dctColours['Triple lines'],bins=lstTJBins,alpha =0.8,density=True)
plt.xlabel('Mean excess volume per atom in \AA$^{3}$')
plt.legend([dctLabels['Grain boundaries'],dctLabels['Triple lines']])
plt.tight_layout()
plt.show()
#%%
fltMeanTJV = np.mean(np.concatenate(lstAllTJVs))
fltStdTJV = np.std(np.concatenate(lstAllTJVs))
fltMeanGBV = np.mean(lstAllGBVs)
fltStdGBV = np.std(lstAllGBVs)
fltMeanCSLV = np.mean(lstAllCSLVs)
fltStdCSLV = np.std(lstAllCSLVs)
fltMeanCylinderV = np.mean(lstAllCylinderVs)
fltStdCylinderV = np.std(lstAllCylinderVs)
print(fltMeanTJV,fltStdTJV,fltMeanGBV,fltStdGBV)
print(fltMeanCSLV,fltStdCSLV,fltMeanCylinderV,fltStdCylinderV)
#%%
lstBinsVPerAtom = (np.linspace(0,7,25)/10).tolist()
lstAllGBEs = []
lstAllGBEs.extend(mf.FlattenList(lstAllCSLVs))
lstAllGBEs.extend(mf.FlattenList(lstAllCSLVs))
lstAllGBEs.extend(mf.FlattenList(lstAllCylinderVs))
plt.hist(lstAllGBVs,density=True,color=dctColours['Grain boundaries'], bins=lstBinsVPerAtom)
plt.hist(mf.FlattenList(lstAllTJVs),density=True,color=dctColours['Triple lines'], bins=lstBinsVPerAtom,alpha=0.8)
plt.xlabel(strMeanExcessVolumePerAtom)
plt.xlim([0,0.7])
plt.legend([dctLabels['Grain boundaries'], dctLabels['Triple lines']])
plt.show()
#%%
plt.hist(mf.FlattenList(lstAllCSLVs),density=True,color=dctColours['CSLs'], bins=lstBinsVPerAtom)
plt.xlabel(strMeanExcessVolumePerAtom)
plt.xlim([0,0.7])
plt.ylim([0,13])
plt.legend([dctLabels['CSLs']])
plt.show()
#%%
plt.hist(mf.FlattenList(lstAllCylinderVs),density=True,color=dctColours['Cylinders'], bins=lstBinsVPerAtom)
plt.xlabel(strMeanExcessVolumePerAtom)
plt.xlim([0,0.7])
plt.ylim([0,13])
plt.legend([dctLabels['Cylinders']])
plt.show()
#%%
plt.hist(mf.FlattenList(lstAllTJVs),density=True,color=dctColours['Triple lines'], bins=lstBinsVPerAtom)
plt.xlabel(strMeanExcessVolumePerAtom)
plt.xlim([0,0.7])
plt.ylim([0,13])
plt.legend([dctLabels['Triple lines']])
plt.show()

#%%
###Excess energy per atom 
lstAllTJE = []
lstAllCylinderE = []
lstAllCSLE = []
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
        lstAllCSLE.append(lstCSL)
        lstAllCylinderE.append(lstCylinder)
        lstAllTJE.append(lstTJ)
        j +=1
#inStoreGB: AS.DirStore, inStoreTJ: AS.DirStore,indctColours: dict(), inDeltaMax = 10)
#%%
lstBinsPEPerAtom = (np.linspace(0,6,25)/100).tolist()
lstAllGBEs = []
lstAllGBEs.extend(mf.FlattenList(lstAllCSLE))
lstAllGBEs.extend(mf.FlattenList(lstAllCSLE))
lstAllGBEs.extend(mf.FlattenList(lstAllCylinderE))
plt.hist(lstAllGBEs,density=True,color=dctColours['Grain boundaries'], bins=lstBinsPEPerAtom)
plt.hist(mf.FlattenList(lstAllTJE),density=True,color=dctColours['Triple lines'], bins=lstBinsPEPerAtom,alpha=0.8)
plt.xlabel(strMeanPotentialEnergyPerAtom)
plt.legend([dctLabels['Grain boundaries'], dctLabels['Triple lines']])
plt.show()
print(np.mean(np.array(lstAllGBEs)), np.mean(np.array(lstAllTJE)))
#%%
plt.xlim([0,0.06])
plt.ylim([0,165])
plt.xlabel(strMeanPotentialEnergyPerAtom)
plt.hist(mf.FlattenList(lstAllTJE),bins=lstBinsPEPerAtom,color = dctColours['Triple lines'],alpha =1,density=True)
plt.legend([dctLabels['Triple lines']])
plt.show()
#%%
plt.xlim([0,0.06])
plt.ylim([0,165])
plt.xlabel(strMeanPotentialEnergyPerAtom)
plt.hist(mf.FlattenList(lstAllCSLE),bins=lstBinsPEPerAtom,color = dctColours['CSLs'],alpha =1,density=True)
plt.legend([dctLabels['CSLs']])
plt.show()
#%%
plt.xlim([0,0.06])
plt.ylim([0,165])
plt.xlabel(strMeanPotentialEnergyPerAtom)
plt.hist(mf.FlattenList(lstAllCylinderE),bins=lstBinsPEPerAtom,color = dctColours['Cylinders'],alpha =1,density=True)
plt.legend([dctLabels['Cylinders']])
plt.show()
#%%
plt.xlim([0,0.06])
plt.ylim([0,150])
plt.xlabel(strMeanPotentialEnergyPerVolume)
plt.hist(lstAllCylinderE,bins=lstBinsPEPerAtom,color=dctColours['Cylinders'],alpha = 1,density=True)
plt.legend([dctLabels['Cylinders']])
plt.show()
#%%
plt.xlim([0,0.012])
plt.ylim([0,650])
plt.xlabel(strMeanPotentialEnergyPerVolume)
plt.hist(lstAllTJEDs,bins=lstBinsED,color = dctColours['Triple lines'],alpha =1,density=True)
plt.legend([dctLabels['Triple lines']])
plt.show()



#%%
##Correlations between excess volume and energy
# for i in range(len(lstAllCylinderVs)):
#     plt.scatter(lstAllCylinderVs[i],lstAllCylinderE[i])
#     print(np.corrcoef(lstAllCylinderVs[i],lstAllCylinderE[i]))
#     plt.show()
plt.scatter(np.concatenate(lstAllCSLVs),np.concatenate(lstAllCSLE),c=dctColours['CSLs'])
plt.xlabel(strMeanExcessVolumePerAtom,fontsize=16)
plt.ylabel(strMeanPotentialEnergyPerAtom,fontsize=16)
plt.legend([dctLabels['CSLs']])
plt.show()
#%%
fltMeanTJE = np.mean(np.concatenate(lstAllTJE))
fltStdTJE = np.std(np.concatenate(lstAllTJE))
fltMeanGBE = np.mean(lstAllGBEs)
fltStdGBE = np.std(lstAllGBEs)
fltMeanCSLE = np.mean(lstAllCSLE)
fltStdCSLE = np.std(lstAllCSLE)
fltMeanCylinderE = np.mean(lstAllCylinderE)
fltStdCylinderE = np.std(lstAllCylinderE)
print(fltMeanTJE, fltStdTJE, fltMeanGBE, fltStdGBE)
print(fltMeanCSLE,fltStdCSLE,fltMeanCylinderE,fltStdCylinderE)

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
#%%
lstAllGBs= []
lstAllGBs.extend(lstAllCSL)
lstAllGBs.extend(lstAllCSL)
lstAllGBs.extend(lstAllCylinder)
#plt.gca().set_aspect('equal')
plt.axline((0,0),slope=1,c='black',linestyle ='dashed')
plt.scatter(lstAllCSL,lstAllBiCrystal,s=12,c='black')
plt.xlabel(strMeanExcessPotentialEnergyPerUnitAreaConstantWidth)
plt.ylabel(strMeanExcessPotentialEnergyPerUnitAreaGibbsian)
plt.show()
print(np.where(np.array(lstAllCSL) > 0.0438))
# %%
lstBinsGandCSL = (np.linspace(0,5,25)/100).tolist()
plt.hist(lstAllBiCrystal,density=True,color=dctColours['Bicrystal'], bins=lstBinsGandCSL,alpha=1)
plt.hist(lstAllCSL,density=True,color=dctColours['Grain boundaries'], bins=lstBinsGandCSL,alpha=0.8)
plt.xlim([0,0.05])
plt.ylim([0,120])
plt.xlabel('$\gamma$ in eV \AA$^{-2}$')
plt.legend(['Gibbsian approach', 'Constant width'],loc='upper left',fontsize=17)
#plt.legend(['Gibbsian approach'])
#plt.legend(['Constant width approximation'])
plt.show()
print(np.mean(np.array(lstAllCSL)), np.mean(np.array(lstAllBiCrystal)))
#%%
lstAllGBs= []
lstAllGBs.extend(lstAllCSL)
lstAllGBs.extend(lstAllCylinder)
plt.hist(lstAllGBs, density = True, color=dctColours['Grain boundaries'], bins =25)
plt.legend([dctLabels['Grain boundaries']])
plt.xlabel('$\gamma$ in eV \AA$^{-2}$')
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
lstTJsinFours = []
for a in lstAxes:
    objTJ = dctTJAxes[a]
    lstSigma = dctSigma[a]
    j= 0
    while j  < len(lstSigma):
        print(a, lstSigma[j])
        objStoreTJ = objTJ.GetSigmaStore(lstSigma[j])
        lstTJs = PlotTJLineTension(objStoreTJ,3*4.05,8,dctColours,dctLabels)
        lstTJsinFours.extend(lstTJs)
        j +=1

# %%
plt.hist(np.concatenate(lstTJsinFours),bins=20,density=True,color=dctColours['Triple lines'],label=dctLabels['Triple lines'])
plt.legend([dctLabels['Triple lines']])
plt.xlabel('Line tension in eV $\AA^{-1}$')
plt.show()
#%%
lstDeviations = list(map(lambda x: np.std(x),lstTJsinFours))
plt.hist(lstDeviations,bins=20,density= True,color=dctColours['Triple lines'])
plt.xlabel(r'Standard deviation of the four lines tensions for each $S_{\text{TJ}}$ in eV \AA$^{-1}$')
plt.legend([dctLabels['Triple lines']])
# %%
np.median(lstDeviations)
# %%
np.mean(np.concatenate(lstTJsinFours))
np.std(np.concatenate(lstTJsinFours))

# %%
