# %%
import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D
import copy as cp
from scipy import spatial
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from matplotlib import animation
import MiscFunctions as mf
from sklearn.cluster import DBSCAN
import pickle
# %%
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams['figure.dpi'] = 300
strTime = 'Time in fs'
strPotentialEnergy = 'Potential energy in eV'
strVolume =r'Volume in \AA$^{3}$'
plt.rcParams.update({'font.size': 16})
#%%
def DataMobilityConversion(inMob):
    fltC = 1.602176634*10**(-19)
    AtoM = 10**(-40)
    fstoS =  10**(-15)   
    return inMob*AtoM/(fltC*fstoS)
print(DataMobilityConversion(0.5))
#%%
arrValues = np.loadtxt('/home/p17992pt/147PsPtBV12.txt')
print(np.shape(arrValues))
fltArea = 2*np.unique(arrValues[2,:]/arrValues[0,:])[0]
intEnd = 200
intStart = 0
plt.scatter(arrValues[0,intStart:intEnd],arrValues[3,intStart:intEnd])
popt,pop =optimize.curve_fit(FitLine,(arrValues[1,intStart:intEnd])**2/4, arrValues[3,intStart:intEnd])
#plt.plot((arrValues[1,intStart:intEnd])**2/4,FitLine((arrValues[1,intStart:intEnd])**2/4,*popt),c='black')
# lstValues = BootstrapEstimate((arrValues[1,:-600]/2)**2,arrValues[3,:-600],400)
print(1/popt[0])
plt.show()
#plt.scatter(arrValues[0,:],arrValues[3,:])
#plt.show()
#print(1/np.mean(lstValues),np.std(lstValues))
# %%
def FitCurve(x, a, b, c):
    return a*x + b*np.sqrt(x)+c
# %%
def DiffFitCurve(x, a, b):
    return a - 1/2*b*x**(-1/2)
# %%
def FitProportional(x, a):
    return a*x
# %%
def FitLine(x, a, b):
    return a*x + b
# %%
def BlockBootstrapEstimate(lstX, lstY, fitFunction=None):
    lstValues = []
    lstAllX = []
    lstAllY = []
    intN = min(list(map(lambda x: len(x), lstX)))
    for i in range(len(lstX)):
        inX = lstX[i]
        inY = lstY[i]
        arrPositions = mf.BootStrapRows(intN, 1)[0]
        arrX = np.array(inX)[arrPositions]
        arrY = np.array(inY)[arrPositions]
        lstAllX.append(arrX)
        lstAllY.append(arrY)
    arrAllX = np.vstack(lstAllX)
    arrAllY = np.vstack(lstAllY)
    if fitFunction is None:
        fitFunction = FitLine
    lstValues.append(list(map(lambda k: optimize.curve_fit(
        fitFunction, arrAllX[:, k], arrAllY[:, k])[0][0], list(range(intN)))))
    return lstValues
# %%
def BootstrapEstimate(inX, inY, intN, fitFunction=None):
    lstValues = []
    if fitFunction is None:
        fitFunction = FitLine
    arrPositions = mf.BootStrapRows(len(inX), intN)
    lstValues = list(map(lambda k: optimize.curve_fit(
        fitFunction, np.array(inX)[k], np.array(inY)[k])[0][0], arrPositions))
    # for k in arrPositions:
    # popt,pop = optimize.curve_fit(FitLine,np.array(inX)[k],np.array(inY)[k])
    # lstValues.append(popt[0])
    return lstValues
# %%
def DoubleBootstrapEstimate(inX1, inY1, inX2, inY2, intN, fitFunction = None):
    if fitFunction is None:
        fitFunction = FitLine
    arrPositions = mf.BootStrapRows(len(inX1), intN)
    lstValues1 = list(map(lambda k: optimize.curve_fit(
        fitFunction, np.array(inX1)[k], np.array(inY1)[k])[0][0], arrPositions))
    lstValues2 = list(map(lambda k: optimize.curve_fit(
        fitFunction, np.array(inX2)[k], np.array(inY2)[k])[0][0], arrPositions))
    return lstValues1, lstValues2

# %%
def PlotMobilities(lstTemp, lstTJ, lst12BV, lst13BV, lstTJE, lst12BVE, lst13BVE, lstYlim=None):
    lstColours = ['darkolivegreen', 'saddlebrown', 'black']
    intCapsize = 5
    lstLegend = []
    if len(lstTJ) > 0:
        plt.scatter(lstTemp, lstTJ, c=lstColours[-1])
        plt.errorbar(lstTemp, lstTJ, lstTJE,
                     capsize=intCapsize, c=lstColours[-1])
        lstLegend.append('TJ')
# plt.scatter(lstNewTemp,lstMobGB)
# plt.scatter(lstNewTemp,lstMobTJ21)
    if len(lst12BV) > 0:
        plt.scatter(lstTemp, lst12BV, c=lstColours[0])
        plt.errorbar(lstTemp, lst12BV, lst12BVE,
                     capsize=intCapsize, c=lstColours[0])
        lstLegend.append('B$_{1,2}$')
    if len(lst13BV) > 0:
        plt.scatter(lstTemp, lst13BV, c=lstColours[1])
        plt.errorbar(lstTemp, lst13BV, lst13BVE,
                     capsize=intCapsize, c=lstColours[1])
        lstLegend.append('B$_{1,3}$')
# plt.scatter(lstNewTemp,arrMins)
    plt.legend(lstLegend)
    plt.xlabel('Temperature in K')
    plt.ylabel('$m_{t}$ in \AA$^4$ eV$^{-1}$ fs$^{-1}$')
#plt.legend(['TJ 7-7-49', 'TJ 21-21-49'])
#plt.legend(['TJ','Min of 12BV 13BV'])
# plt.ylim([0.1,0.5])
    if lstYlim is not None:
        plt.ylim(lstYlim)
    plt.xticks(lstTemp)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
#%%
def PlotAllArhenius(arrTemp, arrTJ, arr12BV, lst13BV, lstTJE, lst12BVE, lst13BVE, lstYlim,strMobLatex):
    lstColours = ['darkolivegreen', 'saddlebrown', 'black']
    intCapsize = 5
    lstLegend = []
    if len(arrTJ) > 0:
        plt.plot(1/arrTemp, np.log(arrTJ), c=lstColours[-1])
        plt.errorbar(1/arrTemp, np.log(arrTJ), lstTJE,
                     capsize=intCapsize, c=lstColours[-1])
        lstLegend.append('TJ')
# plt.scatter(lstNewTemp,lstMobGB)
# plt.scatter(lstNewTemp,lstMobTJ21)
    if len(arr12BV) > 0:
        plt.scatter(1/arrTemp, np.log(arr12BV), c=lstColours[0])
        plt.errorbar(1/arrTemp, np.log(arr12BV), lst12BVE,
                     capsize=intCapsize, c=lstColours[0])
        lstLegend.append('1,2 BC')
    if len(lst13BV) > 0:
        plt.scatter(1/arrTemp, lst13BV, c=lstColours[1])
        plt.errorbar(1/arrTemp, lst13BV, lst13BVE,
                     capsize=intCapsize, c=lstColours[1])
        lstLegend.append('1,3 BC')
# plt.scatter(lstNewTemp,arrMins)
    plt.legend(lstLegend)
    plt.xlabel('Inverse Temperature in K$^{-1}$')
    plt.ylabel('$' + strMobLatex + '$ in $\AA^4$ eV$^{-1}$ fs$^{-1}$')
#plt.legend(['TJ 7-7-49', 'TJ 21-21-49'])
#plt.legend(['TJ','Min of 12BV 13BV'])
# plt.ylim([0.1,0.5])
    if lstYlim is not None:
        plt.ylim(lstYlim)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
#%%
PlotAllArhenius(np.array(lstTemp),arr21[0],arr21[1],arr21[2],arr21[3],arr21[4],arr21[5],[-1,-0.3],'m_t')
# %%
def PlotArrhenius(inlstTemp, inlstMob, blnLine, ylim=None):
    arrITemp = 1/np.array(inlstTemp)
    arrLogMob = np.array(np.log(inlstMob))
    plt.scatter(arrITemp, arrLogMob)
    plt.xlabel('Inverse temperature K$^{-1}$')
    plt.ylabel('$\ln(M)$')
    if blnLine:
        popt, pop = optimize.curve_fit(FitLine, arrITemp, arrLogMob)
        plt.plot(arrITemp, FitLine(arrITemp, *popt), c='black')
        print(popt)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([np.min(arrITemp)-0.0001, np.max(arrITemp)+0.0001])
    plt.tight_layout()
    plt.show()
# %%
strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp600/u03/' 
strType = 'TJ'
strDir = strRoot + strType + '/'
intStart = 20
intFinish = 750
arrVolume = np.loadtxt(
    strDir + 'Volume' +strType + '.txt')
objLog = LT.LAMMPSLog(strDir +strType +  '.log')
objDat = LT.LAMMPSDat(strDir + strType + '.dat')
arrCellVectors = objDat.GetCellVectors()
fltArea = 2*np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
print(fltArea)
lstVPerTime = BootstrapEstimate(arrVolume[0,intStart:intFinish],arrVolume[1,intStart:intFinish],intFinish - intStart,FitLine)
popt,pop = optimize.curve_fit(FitLine,arrVolume[0,intStart:intFinish],arrVolume[2,intStart:intFinish])
plt.scatter(arrVolume[0,intStart:intFinish],arrVolume[1,intStart:intFinish],c='grey')
#plt.plot(arrVolume[0,intStart:intFinish],FitLine(arrVolume[0,intStart:intFinish],*popt),c='black')
plt.xlabel(strTime)
plt.ylabel(strVolume)
plt.show()
print(np.mean(lstVPerTime)/fltArea,np.std(lstVPerTime)/fltArea)
plt.scatter(arrVolume[1,intStart:intFinish],-objLog.GetValues(1)[intStart:intFinish,2])
lstM = BootstrapEstimate(arrVolume[1,intStart:intFinish],objLog.GetValues(1)[intStart:intFinish,2],intFinish-intStart)
print(np.mean(lstM),np.std(lstM))
plt.show()
#%%
popt2,pop2 = optimize.curve_fit(FitLine,arrVolume[1,intStart:intFinish],objLog.GetValues(1)[intStart:intFinish,2])
#plt.scatter( arrVolume[1,50:],objLog.GetValues(1)[50:,2],c='grey')
plt.plot(arrVolume[1,intStart:intFinish],FitLine(arrVolume[1,intStart:intFinish],*popt2),c='black')
lstPEPerV = BootstrapEstimate(arrVolume[1,intStart:intFinish],objLog.GetValues(1)[intStart:intFinish,2],intFinish-intStart,FitLine)
plt.xlabel(strVolume)
plt.ylabel(strPotentialEnergy)
plt.show()
print(popt2)
#%%
print(-popt[0]/(2*fltArea),popt2[0])
print(np.mean(lstVPerTime)/(2*fltArea),np.std(np.array(lstVPerTime)/(2*fltArea)))
print(np.mean(lstPEPerV),np.std(lstPEPerV))
print(np.mean(lstVPerTime)/(2*fltArea*np.mean(lstPEPerV)))
#%%
lstNewLegend = ['TJ', 'B$_{1,2}$', 'B$_{1,3}$' ,'Combined BCs']
#%%
##Potential energy against time
lstTemp = [450,500,550,600,650,700,750]
lstU = ['u01','u02','u03','u04']
strDirBase = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp'
intStart =20
intEnd = 700
lstColours = [ 'black','darkolivegreen', 'saddlebrown']
for T in lstTemp:
    for u in lstU:
        strTemp = strDirBase + str(T) + '/' + u + '/'
        objTJLog = LT.LAMMPSLog(strTemp + 'TJ/TJ.log')
        obj12BVLog = LT.LAMMPSLog(strTemp + '12BV/12BV.log')
        obj13BVLog = LT.LAMMPSLog(strTemp + '13BV/13BV.log')
        arrLogTJ = objTJLog.GetValues(1)
        arrLog12BV = obj12BVLog.GetValues(1)
        arrLog13BV = obj13BVLog.GetValues(1)
        intEnd = len(arrLogTJ[:,0])
        print(T,u)
        plt.scatter(arrLogTJ[intStart:intEnd,0], arrLogTJ[intStart:intEnd,2],c=lstColours[0],s=2)
        intEnd = np.min([len(arrLog12BV[:,0]),len(arrLog13BV[:,0])])
        plt.scatter(arrLog12BV[intStart:intEnd,0], 2*arrLog12BV[intStart:intEnd,2],c=lstColours[1],s=2)
        plt.scatter(arrLog13BV[intStart:intEnd,0], 2*arrLog13BV[intStart:intEnd,2],c=lstColours[2],s=2)
       # plt.plot(arrLogTJ[intStart:intEnd,0], (arrLog12BV[intStart:intEnd,2] +arrLog13BV[ intStart:intEnd,2]),c='purple',linestyle='dashed')
        #plt.ylim([0,350])
        plt.xlim([0,100000])
        plt.xlabel(strTime)
        plt.ylabel(strPotentialEnergy)
        plt.legend(['TJ', 'B$_{1,2}$', 'B$_{1,3}$' ,'Combined BCs'],markerscale=4)
        plt.show()
#%% PE against Volume
lstTemp = [450,500,550,600,650,700,750]
lstU = ['u01','u02','u03','u04']
#strDirBase = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp'
intStart =20
intEnd = 700
lstColours = [ 'black','darkolivegreen', 'saddlebrown']
for T in lstTemp:
    for u in lstU:
        strTemp = strDirBase + str(T) + '/' + u + '/'
        arrVolumeTJ = np.loadtxt(strTemp + 'TJ/VolumeTJ.txt')
        arrVolume12BV = np.loadtxt(strTemp + '12BV/Volume12BV.txt')
        arrVolume13BV = np.loadtxt(strTemp + '13BV/Volume13BV.txt')
        objTJLog = LT.LAMMPSLog(strTemp + 'TJ/TJ.log')
        obj12BVLog = LT.LAMMPSLog(strTemp + '12BV/12BV.log')
        obj13BVLog = LT.LAMMPSLog(strTemp + '13BV/13BV.log')
        arrLogTJ = objTJLog.GetValues(1)
        arrLog12BV = obj12BVLog.GetValues(1)
        arrLog13BV = obj13BVLog.GetValues(1)
        intEnd = len(arrVolumeTJ[1,:])
        print(T, u)
        plt.scatter(arrVolumeTJ[1, intStart:intEnd], arrLogTJ[intStart:intEnd,2],c=lstColours[0],s=2)
        intEnd = len(arrVolume12BV[1,:])
        plt.scatter(2*arrVolume12BV[1, intStart:intEnd], 2*arrLog12BV[intStart:intEnd,2],c=lstColours[1],s=2)
        intEnd = len(arrVolume13BV[1,:])
        plt.scatter(2*arrVolume13BV[1, intStart:intEnd], 2*arrLog13BV[intStart:intEnd,2],c=lstColours[2],s=2)
        # plt.plot(arrVolumeTJ[0, intStart:intEnd], (arrVolume12BV[1, intStart:intEnd] +arrVolume13BV[1, intStart:intEnd]),c='purple',linestyle='dashed')
        # #plt.ylim([0,350])
        #plt.xlim([0,100000])
        plt.legend(lstNewLegend,markerscale=4)
        plt.ylabel(strPotentialEnergy)
        plt.xlabel(strVolume)
        plt.show()
#%%
## Volume against time 
lstTemp = [450,500,550,600,650,700,750]
lstU = ['u01','u02','u03','u04']
#lstU = ['u005','u0075','u01','u0125']
#strDirBase = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp'
intStart =20
intEnd = 700
lstColours = [ 'black','darkolivegreen', 'saddlebrown']
for T in lstTemp:
    for u in lstU:
        strTemp = strDirBase + str(T) + '/' + u + '/'
        arrVolumeTJ = np.loadtxt(strTemp + 'TJ/VolumeTJ.txt')
        arrVolume12BV = np.loadtxt(strTemp + '12BV/Volume12BV.txt')
        arrVolume13BV = np.loadtxt(strTemp + '13BV/Volume13BV.txt')
        intEnd = len(arrVolumeTJ[1,:])
        print(T, u)
        plt.plot(arrVolumeTJ[0, intStart:intEnd], arrVolumeTJ[1, intStart:intEnd],c=lstColours[0])
        intEnd = len(arrVolume12BV[1,:])
        plt.plot(arrVolume12BV[0, intStart:intEnd], 2*arrVolume12BV[1, intStart:intEnd],c=lstColours[1])
        intEnd = len(arrVolume13BV[1,:])
        plt.plot(arrVolume13BV[0, intStart:intEnd], 2*arrVolume13BV[1, intStart:intEnd],c=lstColours[2])
        intEnd = np.min([len(arrVolume12BV[1,:]),len(arrVolume13BV[1,:])])
      #  plt.plot(arrVolumeTJ[0, intStart:intEnd], (arrVolume12BV[1, intStart:intEnd] +arrVolume13BV[1, intStart:intEnd]),c='purple',linestyle='dashed')
        #plt.ylim([0,350])
        plt.xlim([0,100000])
        plt.legend(lstNewLegend,markerscale=4)
        plt.xlabel(strTime)
        plt.ylabel(strVolume)
        plt.show()
# %%
#strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis221/Sigma9_9_9/Temp'
#strRoot = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
#strRoot = '/home/paul/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
#strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
#strRoot = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp'
#lstTemp = [450, 500, 550, 600, 650, 700, 750]
#lstU = [0.005,0.0075,0.01,0.0125] #used for Sigma7_7_49
#lstU = [0.01, 0.02, 0.03, 0.04]
dctTJ = dict()
strType = 'TJ'
for T in lstTemp:
    for u in lstU:
        strU = str(u).split('.')[1]
        strDir = strRoot + str(T) + '/u' + strU + '/' + strType + '/'
        strLogFile = strDir + strType + '.log'
        objLog = LT.LAMMPSLog(strLogFile)
        print(strU, T, objLog.GetColumnNames(1))

# %%
def PopulateTJDictionary(strRoot: str, lstTemp: list, lstU: list, strType: str) -> dict():
    dctReturn = dict()
    for T in lstTemp:
        for u in lstU:
            strU = str(u).split('.')[1]
            strDir = strRoot + str(T) + '/u' + strU + '/' + strType + '/'
            print(strDir)
            strLogFile = strDir + strType + '.log'
            strVolumeFile = strDir + 'Volume' + strType + '.txt'
            print(strLogFile)
            objData = LT.LAMMPSData(
                strDir + '1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
            objAnalysis = objData.GetTimeStepByIndex(-1)
            arrCellVectors = objAnalysis.GetCellVectors()
            arrLog = LT.LAMMPSLog(strLogFile)
            print(strVolumeFile)
            arrVolume = np.loadtxt(strVolumeFile)
            objCSLMobility = gf.CSLMobility(
                arrCellVectors, arrLog.GetValues(1), arrVolume, strType, T, u)
            #objCSLMobility.FindMobilities(1, 4*4.05)
            dctReturn[str(T) + ',' + strU] = objCSLMobility
            #objLog = objCSLMobility.GetLogObject()
            # print(objLog.GetColumnNames(1),strU,str(T))
    return dctReturn
# %%
def PopulateGBDictionary(strRoot: str, lstTemp: list, lstU: list, strType1: str, strType2: str, arrTJCellVectors: np.array) -> dict():
    dctReturn = dict()
    for T in lstTemp:
        for u in lstU:
            strU = str(u).split('.')[1]
            strDir1 = strRoot + str(T) + '/u' + strU + '/' + strType1 + '/'
            strDir2 = strRoot + str(T) + '/u' + strU + '/' + strType2 + '/'
            strLogFile1 = strDir1 + strType1 + '.log'
            strLogFile2 = strDir2 + strType2 + '.log'
            strVolumeFile1 = strDir1 + 'Volume' + strType1 + '.txt'
            strVolumeFile2 = strDir2 + 'Volume' + strType2 + '.txt'
            fltArea = np.linalg.norm(
                np.cross(arrTJCellVectors[0], arrTJCellVectors[2]))
            arrLog1 = LT.LAMMPSLog(strLogFile1).GetValues(1)
            arrVolume1 = np.loadtxt(strVolumeFile1)
            arrLog2 = LT.LAMMPSLog(strLogFile2).GetValues(1)
            arrVolume2 = np.loadtxt(strVolumeFile2)
            tupShape = np.shape(arrLog1)
            arrLog = np.zeros(tupShape)
            arrLog[:, 0] = arrLog1[:, 0]
            arrLog[:, 1] = (arrLog1[:, 1] + arrLog2[:, 1])/2
            arrLog[:, 2] = arrLog1[:, 2] + arrLog2[:, 2]
            arrLog[:, 3] = arrLog1[:, 3] + arrLog2[:, 3]
            arrLog[:, 4] = (arrLog1[:, 4] + arrLog2[:, 4])/2
            intLength = np.min([len(arrVolume1[0, :]), len(arrVolume2[0, :])])
            arrVolume = arrVolume1[:, :intLength] + arrVolume2[:, :intLength]
            arrVolume[2, :] = arrVolume[1, :intLength]/fltArea
            arrVolume[0, :] = arrVolume1[0, :intLength]
            objCSLMobility = gf.CSLMobility(
                arrTJCellVectors, arrLog, arrVolume, strType1, T, u)
            #objCSLMobility.FindMobilities(1, 4*4.05)
            dctReturn[str(T) + ',' + strU] = objCSLMobility
            #objLog = objCSLMobility.GetLogObject()
            # print(objLog.GetColumnNames(1),strU,str(T))
    return dctReturn
# %%
def MakeTJandBCDictionaries(strDir, lstinTemp, lstinU):
    dctTJ = PopulateTJDictionary(strDir, lstinTemp, lstinU, 'TJ')
    dct12BV = PopulateTJDictionary(strDir, lstinTemp, lstinU, '12BV')
    dct13BV = PopulateTJDictionary(strDir, lstinTemp, lstinU, '13BV')
    return dctTJ, dct12BV, dct13BV
# %%
lstTemp = [450, 500, 550, 600, 650, 700, 750]
lstU = [0.01, 0.02, 0.03, 0.04]
#lstU = [0.005,0.0075,0.01,0.0125]#,0.015]
strRoot9_9_9 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp'
#strRoot7_7_49 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
#strRoot21_21_49 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp'
dctTJ, dctBV12, dctBV13 = MakeTJandBCDictionaries(strRoot9_9_9, lstTemp, lstU)

# %%
#with open('/home/p17992pt/dct7_7_49BV13All', 'wb') as fp:
#    pickle.dump(dctBV13, fp)
#%%
with open('/home/p17992pt/dct21_21_49BV12All', 'rb') as fp:    
    dctBV12 = pickle.load(fp)
#%%
def QuickMobilityEstimate(strRoot: str, lstTemp: list, lstU: list, strType1: str) -> dict():
    dctReturn = dict()
    for T in lstTemp:
        for u in lstU:
            strU = str(u).split('.')[1]
            strDir1 = strRoot + str(T) + '/u' + strU + '/' + strType1 + '/'
            strLogFile1 = strDir1 + strType1 + '.log'
            strVolumeFile1 = strDir1 + 'Volume' + strType1 + '.txt'
            arrLog1 = LT.LAMMPSLog(strLogFile1).GetValues(1)
            arrVolume1 = np.loadtxt(strVolumeFile1)
            fltArea = arrVolume[0, 1]/arrVolume[0, 2]
            objRange = slice(100, 200, 1)
            lstVnOut, lstdUBydTOut = DoubleBootstrapEstimate(
                arrVolumeSpeed[0, objRange], -arrVolumeSpeed[2, objRange], arrLogValues[objRange, 0], -arrLogValues[objRange, 2], 1000)
            #objLog = objCSLMobility.GetLogObject()
            # print(objLog.GetColumnNames(1),strU,str(T))
    return dctReturn

# %%
def PartitionByTemperatureSimple(dctAny: dict(), intTemp, uLower: float, uUpper: float, blnSyntheticUOnly=False):
    lstVn = []
    lstU = []
    for a in dctAny.keys():
        if (dctAny[a].GetTemp() == intTemp) and (dctAny[a].GetPEParameter() <= uUpper) and (dctAny[a].GetPEParameter() >= uLower):
            intFinish = dctAny[a].GetLowVolumeCutOff(1, 8*4.05)
            intFinish = np.min([700, intFinish])
            intStart = 400  # int(intFinish/2)
            dctAny[a].SetLinearRange(intStart, intFinish)
            objRange = dctAny[a].GetLinearRange()
            arrVolumeSpeed = dctAny[a].GetVolumeSpeed()
            arrLogValues = dctAny[a].GetLogValues()
            lstVnOut, lstUOut = DoubleBootstrapEstimate(arrVolumeSpeed[0, objRange], -arrVolumeSpeed[2, objRange],                                                arrVolumeSpeed[1, objRange], arrLogValues[objRange, 2], 10**4,FitLine)
            # intFinish - intStart)
            popt, pop = optimize.curve_fit(
                FitLine, arrVolumeSpeed[0, objRange], arrVolumeSpeed[2, objRange])
            plt.scatter(arrVolumeSpeed[0, objRange],
                        arrVolumeSpeed[2, objRange])
            plt.xlabel('Time in fs')
            plt.ylabel('Mean normal displacement in \AA')
            plt.plot(arrVolumeSpeed[0, objRange], FitLine(
                arrVolumeSpeed[0, objRange], *popt), c='black')
            print('Dist vs Time' + str(intTemp) +
                  str(dctAny[a].GetPEParameter()))
            plt.show()
            popt2, pop2 = optimize.curve_fit(
                FitLine, arrVolumeSpeed[1, objRange], arrLogValues[objRange, 2])
            plt.scatter(arrVolumeSpeed[1, objRange], arrLogValues[objRange, 2])
            plt.ylabel('Potential energy in eV')
            plt.xlabel('Volume in \AA$^{3}$')
            print('PE vs Volume' + str(intTemp) +
                  str(dctAny[a].GetPEParameter()))
            plt.show()
          #  lstU.append(popt2[0])
            # lstU.append(lstUOut)
            if blnSyntheticUOnly:
                lstU.append(dctAny[a].GetPEParameter()*4*(4.05**(-3)))
                lstVn.append(lstVnOut)
            else:
                lstU.append(lstUOut)
                lstVn.append(lstVnOut)
    # as there are two grain boundaries moving towards each other the speed of each grain boundary is half the value
    return lstU, list(np.array(lstVn)/2)
#%%
def PartitionByTemperature(dctAny: dict(), intTemp, uLower: float, uUpper: float, blnSyntheticUOnly=False):
    lstVn = []
    lstU = []
    for a in dctAny.keys():
        if (dctAny[a].GetTemp() == intTemp) and (dctAny[a].GetPEParameter() <= uUpper) and (dctAny[a].GetPEParameter() >= uLower):
            intFinish = dctAny[a].GetLowVolumeCutOff(1, 8*4.05)
            intFinish = np.min([700, intFinish])
            intStart = 400  # int(intFinish/2)
            dctAny[a].SetLinearRange(intStart, intFinish)
            objRange = dctAny[a].GetLinearRange()
            arrVolumeSpeed = dctAny[a].GetVolumeSpeed()
            arrLogValues = dctAny[a].GetLogValues()
            lstVnOut, lstUOut = DoubleBootstrapEstimate(arrVolumeSpeed[0, objRange], -arrVolumeSpeed[2, objRange],                                                arrVolumeSpeed[1, objRange], arrLogValues[objRange, 2], 10**4,FitLine)
            # intFinish - intStart)
            popt, pop = optimize.curve_fit(
                FitLine, arrVolumeSpeed[0, objRange], arrVolumeSpeed[2, objRange])
            plt.scatter(arrVolumeSpeed[0, objRange],
                        arrVolumeSpeed[2, objRange])
            plt.xlabel('Time in fs')
            plt.ylabel('Mean normal displacement in \AA')
            plt.plot(arrVolumeSpeed[0, objRange], FitLine(
                arrVolumeSpeed[0, objRange], *popt), c='black')
            print('Dist vs Time' + str(intTemp) +
                  str(dctAny[a].GetPEParameter()))
            plt.show()
            popt2, pop2 = optimize.curve_fit(
                FitLine, arrVolumeSpeed[1, objRange], arrLogValues[objRange, 2])
            plt.scatter(arrVolumeSpeed[1, objRange], arrLogValues[objRange, 2])
            plt.ylabel('Potential energy in eV')
            plt.xlabel('Volume in \AA$^{3}$')
            print('PE vs Volume' + str(intTemp) +
                  str(dctAny[a].GetPEParameter()))
            plt.show()
          #  lstU.append(popt2[0])
            # lstU.append(lstUOut)
            if blnSyntheticUOnly:
                lstU.append(dctAny[a].GetPEParameter()*4*(4.05**(-3)))
                lstVn.append(lstVnOut)
            else:
                lstU.append(lstUOut)
                lstVn.append(lstVnOut)
    # as there are two grain boundaries moving towards each other the speed of each grain boundary is half the value
    return lstU, list(np.array(lstVn)/2)
# %%
def GetVolumeOrLAMMPSLog(inCSL: gf.CSLMobility, lstVolume: list, lstLAMMPS: list, intStart=100):
    arrLogValues = inCSL.GetLogValues()
    arrVolumeSpeed = inCSL.GetVolumeSpeed()
    intMax = np.min([len(arrLogValues[:, 0]), len(arrVolumeSpeed[0])])
    if len(lstLAMMPS) == 0:
        x = arrVolumeSpeed[lstVolume[0], 100:intMax]
        y = arrVolumeSpeed[lstVolume[1], 100:intMax]
    elif len(lstVolume) == 0:
        x = arrLogValues[100:intMax, lstLAMMPS[0]]
        y = arrLogValues[100:intMax, lstLAMMPS[1]]
    else:
        x = arrVolumeSpeed[lstVolume[0], 100:intMax]
        y = arrLogValues[100:intMax, lstLAMMPS[0]]
    return x, y


# %%
lstLegend = []
lstpopt = []
lstpopt2 = []
lstMeanPop = []
lstStdPop = []
for k in lstU:
    strKey = '550' + ',' + str(k).split('.')[1]
    x, y = dctTJ[strKey].GetVolumeOrLAMMPSLog([0, 2], [])
    popt, pop = optimize.curve_fit(FitLine, x, y)
    plt.plot(x, y)
    plt.plot(x, FitLine(x, *popt))
    lstLegend.append(k)
    lstpopt.append(popt[0]*4.05**3/4)
    x2, y2 = dctTJ[strKey].GetVolumeOrLAMMPSLog([1], [2])
    lstValues = BootstrapEstimate(x2, y2, 1000)
    popt2, pop2 = optimize.curve_fit(FitLine, x2, y2)
    lstMeanPop.append(np.mean(lstValues))
    lstStdPop.append(np.std(lstValues))
    lstpopt2.append(popt2[0])
plt.legend(lstLegend)
plt.show()
plt.scatter(np.array(lstU)*4/(4.05**3), -np.array(lstpopt), c='black')
# plt.scatter(np.array(lstpopt2),-np.array(lstpopt),c='purple')
plt.scatter(np.array(lstMeanPop), -np.array(lstpopt), c='purple')
plt.xlabel('Driving force in eV per atom')
plt.ylabel('Mean normal velocity in $\AA$ per fs')
plt.legend(['Synthetic', 'Calculated'])
plt.show()
plt.scatter(np.array(lstU)*4/(4.05**3), np.array(lstMeanPop), c='purple')
plt.xlabel('Synthetic driving force in eV \AA$^{-3}$')
plt.ylabel('Calculated driving force in eV \AA$^{-3}$')
plt.axline((0, 0), slope=1, c='black')
plt.show()
# %%
lstAllMeanPop = []
lstAllMeanStd = []
dctNow  = dctTJ
for t in lstTemp[:]:
    lstLegend = []
    lstpopt = []
    lstpopt2 = []
    lstMeanPop = []
    lstStdPop = []
    for k in lstU:
        strKey = str(t) + ',' + str(k).split('.')[1]
        x, y = dctNow[strKey].GetVolumeOrLAMMPSLog([0, 2], [])
        popt, pop = optimize.curve_fit(FitLine, x, y)
        # plt.plot(x,y)
        # plt.plot(x,FitLine(x, *popt))
        # lstLegend.append(k)
        lstpopt.append(popt[0]*4.05**3/4)
        x2, y2 = dctNow[strKey].GetVolumeOrLAMMPSLog([1], [2])
        lstValues = BootstrapEstimate(x2, y2, 1000)
        popt2, pop2 = optimize.curve_fit(FitLine, x2, y2)
        lstMeanPop.append(np.mean(lstValues))
        lstStdPop.append(2*np.std(lstValues))
        lstpopt2.append(popt2[0])
    lstAllMeanPop.append(lstMeanPop)
    lstAllMeanStd.append(lstStdPop)

for l in range(len(lstAllMeanPop)):
    plt.plot(np.array(lstU)*4/(4.05**3),
             np.array(lstAllMeanPop[l]), marker='x')
plt.legend(lstTemp)
# for j in range(len(lstAllMeanPop)):
#      plt.errorbar(np.array(lstU)*4/(4.05**3),np.array(lstAllMeanPop[j]),lstAllMeanStd[j],capsize=5)
plt.xlabel('$P_s$ in eV \AA$^{-3}$')
plt.ylabel('$P_t$ in eV \AA$^{-3}$')
plt.xlim([0.0002, 0.0008])
plt.ylim([0.000, 0.0008])
plt.axline((0.00, 0.00), slope=1, c='black')
#plt.xlim([0.0004, 0.0025])
#plt.ylim([0.0004, 0.0025])
#plt.axline((0.004, 0.004), slope=1, c='black')
plt.show()
#%%
np.savetxt('/home/p17992pt/9PsPtBV13.txt',np.array([x,y,x2,y2]))
arrP = np.loadtxt('/home/p17992pt/9PsPtBV13.txt')
print(np.shape(arrP))
# %%
def WriteSyntheticMobilityValues(lstInTemp, dctAny: dict(), lstU: list):
    lstMobility = []
    lstMobilityStd = []
    for j in lstInTemp:
        uLower = lstU[0]
        if j == -1:  # comment out as this removes new grain data point for sigma 9-9-9
            uUpper = lstU[-2]
        else:

            uUpper = lstU[-1]
        #uUpper = lstU[-1]
        #uLower = lstU[0]
        tupValues = PartitionByTemperature(dctAny, j, uLower, uUpper, False)
        # plt.title(str(j))
        # plt.scatter(tupValues[0], tupValues[1])
        # #plt.scatter(tupValues[0],tupValues[1])
        # plt.ylim([0,0.005])
        # plt.xlim([0,0.08*4*4.05**(-3)])
        # plt.show()
        intL = len(tupValues[1][1])
        lstSyntheticU = []
        arrOrder = np.argsort(list(map(lambda x: np.mean(x),tupValues[0])))
        for i in arrOrder:
            lstSyntheticU.append([4*lstU[i]*4.05**(-3)]*intL)
        lstMeanV = list(map(lambda x: np.mean(x), tupValues[1]))
        lstStdV = list(map(lambda x: np.mean(x), tupValues[1]))
        #uPressure = 4*np.array(lstU)*4.05**(-3)
        #popt,pop = optimize.curve_fit(FitLine,tupValues[0],tupValues[1])
        # plt.plot(np.array(tupValues[0]),FitLine(np.array(tupValues[0]),popt[0],popt[1]))
        # lstMobility.append(popt[0])
        #lstValues = BlockBootstrapEstimate(tupValues[0],tupValues[1])
        lstValues = BlockBootstrapEstimate(lstSyntheticU, tupValues[1],FitProportional)
        lstMobility.append(np.mean(lstValues))
        lstMobilityStd.append(2*np.std(lstValues))
    return lstMobility, lstMobilityStd

# %%
def WriteMobilityValues(lstInTemp, dctAny: dict, uLower: float, uUpper: float, blnSyntheticUOnly=False):
    lstMobility = []
    lstMobilityStd = []
    for j in lstInTemp:
        if j ==-1: #comment out as this removes new grain data point for sigma 9-9-9
            tupValues = PartitionByTemperature(
            dctAny, j, uLower, 0.03, blnSyntheticUOnly)
        else:
            tupValues = PartitionByTemperature(
            dctAny, j, uLower, uUpper, blnSyntheticUOnly)
        for i in range(len(tupValues[0])):
            # plt.title(str(j))
            print(str(j))
            plt.scatter(tupValues[0][i], tupValues[1][i])
        # plt.scatter(tupValues[0],tupValues[1])
        plt.ylim([0, 0.005])
        plt.xlim([0, 0.08*4*4.05**(-3)])
        plt.show()

        #popt,pop = optimize.curve_fit(FitLine,tupValues[0],tupValues[1])
        # plt.plot(np.array(tupValues[0]),FitLine(np.array(tupValues[0]),popt[0],popt[1]))
        # lstMobility.append(popt[0])
        lstValues = BlockBootstrapEstimate(
            tupValues[0], tupValues[1], FitProportional)
        #lstValues = BootstrapEstimate(tupValues[0],tupValues[1],100)
        plt.hist(lstValues)
        plt.xlabel('$m_t$')
        lstMobility.append(np.mean(lstValues[0]))
        lstMobilityStd.append(2*np.std(lstValues[0]))
    return lstMobility, lstMobilityStd
#%%
with open('/home/p17992pt/dct7_7_49TJAll', 'rb') as fp:    
    dctTJ = pickle.load(fp)

# %%
lstTemp =[450,500,550,600,650,700,750]
#lstU = [0.005, 0.0075, 0.01, 0.0125]#,0.015]
lstU = [0.01, 0.02, 0.03, 0.04]
lstMobTJ, lstErrorTJ = WriteMobilityValues(lstTemp, dctTJ, lstU[0],lstU[-1])
#lstMobTJ21 = WriteMobilityValues(lstNewTemp, dctTJ21)
# %%
lstMobBV12, lstErrorBV12 = WriteSyntheticMobilityValues(
    lstTemp, dctBV12, lstU)

# %%
lstMobBV13, lstErrorBV13 = WriteSyntheticMobilityValues(
    lstTemp, dctBV13, lstU)


# %%
# %%
#lstMobGB = WriteMobilityValues(lstNewTemp, dctGB7)
lstMobBVs = []
lstMobBVs.append(lstMobBV12)
lstMobBVs.append(lstMobBV13)
arrBV = np.vstack(lstMobBVs)
arrMins = np.min(arrBV, axis=0)
# %%

# %%
PlotMobilities(lstTemp, lstMobTJ, lstMobBV12, lstMobBV13, np.array(
    lstErrorTJ), np.array(lstErrorBV12), np.array(lstErrorBV13),[0.2,2.5])
# %%
PlotMobilities(lstTemp[:], lstMobTJ[:], [], [],
               np.array(lstErrorTJ[:]), [], [])
# %%
PlotMobilities(lstTemp, lstMobTJ, [], [], np.array(
    lstErrorTJ), [], [0.5, 1.0])

# %%
PlotArrhenius(lstTemp[:-3], lstMobTJ[:-3], True)
# %%
lstAllValues = []
lstAllValues.append(np.array(lstMobTJ))
lstAllValues.append(np.array(lstMobBV12))
lstAllValues.append(np.array(lstMobBV13))
lstAllValues.append(np.array(lstErrorTJ))
lstAllValues.append(np.array(lstErrorBV12))
lstAllValues.append(np.array(lstErrorBV13))
#%%
np.savetxt('/home/p17992pt/Sigma7_7_49u00tou0125SProp.txt', np.vstack(lstAllValues))
# %%
lstTemp = [450, 500, 550, 600, 650, 700, 750]
arr21 = np.loadtxt('/home/p17992pt/Sigma9_9_9u01tou03Prop.txt')
# %%
PlotMobilities(lstTemp, arr21[0], arr21[1], arr21[2],
               arr21[3], arr21[4], arr21[5],[0.3,0.7])
# %%
PlotAllArhenius(lstTemp[:], arr21[0], True)
# %%


class AnimateGBs(object):
    def __init__(self, strDir: str, arrCellVectors: np.array):
        self.__strRoot = strDir
        self.__CellVectors = arrCellVectors
        self.__blnTJ = True
        self.__bln12 = True
        self.__bln13 = True
        self.__ScatterSize = 0.5

    def Animate(self, i):
        self.__ax.clear()
        intStep = 500*i
        objPeriodicTree1 = gf.PeriodicWrapperKDTree(
            arrPoints12, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors), 4*4.05, ['p', 'p', 'p'])
        if self.__blnTJ:
            strDir = self.__strRoot + 'TJ/'
            arrPoints12 = np.loadtxt(
                strDir + 'Mesh12TJ' + str(intStep) + '.txt')
            objPeriodicTree12 = gf.PeriodicWrapperKDTree(
                arrPoints12, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors), 4*4.05, ['p', 'p', 'p'])
            arrPoints13 = np.loadtxt(
                strDir + 'Mesh13TJ' + str(intStep) + '.txt')
            objPeriodicTree13 = gf.PeriodicWrapperKDTree(
                arrPoints13, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors), 4*4.05, ['p', 'p', 'p'])
            arrPoints23 = np.loadtxt(
                strDir + 'Mesh23TJ' + str(intStep) + '.txt')
            objPeriodicTree23 = gf.PeriodicWrapperKDTree(
                arrPoints23, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors), 4*4.05, ['p', 'p', 'p'])
            # self.__ax.scatter(*tuple(zip(*arrPoints12)),c='b')
            # self.__ax.scatter(*tuple(zip(*arrPoints13)),c='b')
            # self.__ax.scatter(*tuple(zip(*arrPoints23)),c='b')
            self.__ax.scatter(
                *tuple(zip(*objPeriodicTree12.GetExtendedPoints())), c='b')
            self.__ax.scatter(
                *tuple(zip(*objPeriodicTree13.GetExtendedPoints())), c='b')
            self.__ax.scatter(
                *tuple(zip(*objPeriodicTree23.GetExtendedPoints())), c='b')
        if self.__bln12:
            strDir = self.__strRoot + '12BV/'
            arrPoints12 = np.loadtxt(
                strDir + 'Mesh1212BV' + str(intStep) + '.txt')
            arrPoints12 = arrPoints12 + self.__CellVectors[0]/2
            self.__ax.scatter(*tuple(zip(*arrPoints12)), c='r')
        if self.__bln13:
            strDir = self.__strRoot + '13BV/'
            arrPoints13 = np.loadtxt(
                strDir + 'Mesh1213BV' + str(intStep) + '.txt')
            self.__ax.scatter(*tuple(zip(*arrPoints13)), c='g')

    def WriteFile(self, strFilename: str, blnTJ: bool, bln12: bool, bln13: bool, intFrames):
        self.__blnTJ = blnTJ
        self.__bln12 = bln12
        self.__bln13 = bln13
        fig, ax = plt.subplots()
        self.__ax = ax
        ani = animation.FuncAnimation(
            fig, self.Animate, interval=500, frames=intFrames)
        writergif = animation.PillowWriter(fps=10)
        ani.save(strFilename, writer=writergif)

# %%


class AnimateTJs(object):
    def __init__(self, strDir: str, arrCellVectors: np.array):
        self.__strRoot = strDir
        self.__CellVectors = arrCellVectors
        self.__blnTJ = True
        self.__bln12 = True
        self.__bln13 = True
        self.__ScatterSize = 0.5
        self.__bln3d = False
        self.__TimeStep = 500

    def SetTimeStep(self, inTimeStep):
        self.__TimeStep = inTimeStep

    def GetTimeStep(self):
        return self.__TimeStep

    def FindTripleLines(self, intStep):
        strDir = self.__strRoot + 'TJ/'
        arrPoints12 = np.loadtxt(strDir + 'Mesh12TJ' + str(intStep) + '.txt')
        arrPoints13 = np.loadtxt(strDir + 'Mesh13TJ' + str(intStep) + '.txt')
        arrPoints23 = np.loadtxt(strDir + 'Mesh23TJ' + str(intStep) + '.txt')
        lstAllMeshPoints = []
        lstAllMeshPoints.append(arrPoints12)
        lstAllMeshPoints.append(arrPoints13)
        lstAllMeshPoints.append(arrPoints23)
        intPos = 0
        for k in lstAllMeshPoints:
            clustering = DBSCAN(2*4.05).fit(k)
            arrLabels = clustering.labels_
            arrUniqueLabels, arrCounts = np.unique(
                arrLabels, return_counts=True)
            arrRows1 = np.where(arrCounts > 10)[0]
            arrRows2 = np.where(
                np.isin(arrLabels, arrUniqueLabels[arrRows1]))[0]
            lstAllMeshPoints[intPos] = k[arrRows2]
            intPos += 1
        intTJs = len(lstAllMeshPoints)
        lstAllTJMesh = []
        for i in range(intTJs):
            lstOverlapIndices = []
            intCount = 0
            objTreei = gf.PeriodicWrapperKDTree(lstAllMeshPoints[i], self.__CellVectors, gf.FindConstraintsFromBasisVectors(
                self.__CellVectors), 2*4.05, ['p', 'p', 'p'])
            for j in range(intTJs):
                if i != j:
                    objTreej = gf.PeriodicWrapperKDTree(lstAllMeshPoints[j], self.__CellVectors, gf.FindConstraintsFromBasisVectors(
                        self.__CellVectors), 2*4.05, ['p', 'p', 'p'])
                    arrIndices, arrDistances = objTreei.Pquery_radius(
                        objTreej.GetExtendedPoints(), 4*4.05)
                    lstIndices = mf.FlattenList(arrIndices)
                    if len(lstIndices) > 0:
                        if len(lstOverlapIndices) > 0:
                            lstOverlapIndices = list(
                                set(lstOverlapIndices).intersection(set(lstIndices)))
                            intCount += 1
                        else:
                            lstOverlapIndices = lstIndices
                    else:
                        print("Missing mesh points")
            if intCount == 1:
                arrIndices = objTreei.GetPeriodicIndices(lstOverlapIndices)
                #arrIndices = mf.FlattenList(arrIndices)
                #arrPoints = objTreei.GetExtendedPoints()[lstOverlapIndices,:]
                arrPoints = objTreei.GetOriginalPoints()[arrIndices, :]
                arrPoints = np.unique(arrPoints, axis=0)
                lstAllTJMesh.append(arrPoints)
        lstTripleLines = self.GroupTripleLines(np.vstack(lstAllTJMesh))
        return lstTripleLines

    def GroupTripleLines(self, arrPoints: np.array):
        lstTripleLines = []
        objTreePositions = gf.PeriodicWrapperKDTree(
            self.__OriginalPositions, self.__CellVectors, gf.FindConstraintsFromBasisVectors(self.__CellVectors), 4*4.05, ['p', 'p', 'p'])
        arrDistances, arrIndices = objTreePositions.Pquery(arrPoints, k=1)
        arrIndices = np.array(mf.FlattenList(arrIndices))
        lstIndices = objTreePositions.GetPeriodicIndices(arrIndices)
        arrPeriodicIndices = np.array(lstIndices)
        arrExtendedPoints = objTreePositions.GetExtendedPoints()
        arrOriginalPoints = objTreePositions.GetOriginalPoints()
        arrOriginalIndices = objTreePositions.Pquery(arrOriginalPoints)
        arrOriginalIndices = np.unique(mf.FlattenList(arrOriginalIndices))
        # for j in range(len(self.__OriginalPositions)):
        for j in arrOriginalIndices.astype('int'):
            arrRows = np.where(arrPeriodicIndices == j)[0]
            if len(arrRows):
                arrNewIndices = arrIndices[arrRows]
                arrNewPoints = arrPoints[arrRows]
                arrNewExtendedPoints = arrExtendedPoints[arrNewIndices, :]
                # lstTripleLines.append(arrNewExtendedPoints)
                # from the original point to the extended point
                arrTranslations = arrNewExtendedPoints - arrOriginalPoints[j]
                arrTranslations[:, 2] = np.zeros(len(arrTranslations))
                arrReturn = np.unique(arrNewPoints-arrTranslations, axis=0)
                # move points back to the position closest to the original triple line positions
                lstTripleLines.append(arrReturn)
            else:
                print('error frame ' + str(self.__strRoot))
        return lstTripleLines

    def FindMeanTripleLinePositions(self, intSteps: int, intStepSize: int):
        lstReturnPoints = []
        lstMissingSteps = []
        for i in range(intSteps):
            lstTripleLines = self.FindTripleLines(intStepSize*i)
            if len(lstTripleLines) == 4:
                lstNewPositions = list(
                    map(lambda x: np.mean(x, axis=0), lstTripleLines))
                lstReturnPoints.append(lstNewPositions)
            else:
                lstMissingSteps.append(i)
        return lstReturnPoints, lstMissingSteps

    def Animate(self, i):
        self.__ax.clear()
        intStep = 500*i
        lstAllPoints = self.FindTripleLines(intStep)
        lstColours = ['b', 'g', 'r', 'k']
        k = 0
        if len(lstAllPoints) == 4:
            lstNewPositions = list(
                map(lambda x: np.mean(x, axis=0), lstAllPoints))
           # self.__OriginalPositions = np.array(lstNewPositions)
            for j in lstAllPoints:
                if len(j) > 1:
                    self.__ax.set_aspect('auto')
                    if self.__bln3d:
                        self.__ax.scatter(*tuple(zip(*j)), c=lstColours[k])
                    else:
                        self.__ax.scatter(
                            *tuple(zip(*j[:, :2])), c=lstColours[k])
                    self.__ax.set_xbound([-100, self.__CellVectors[0][0]+100])
                    self.__ax.set_ybound([-100, self.__CellVectors[1][1]+100])
                k += 1
        else:
            print("error missing junction lines only " +
                  str(len(lstAllPoints)) + ' frame ' + str(i))
        print(str(i))

    def SetOriginalPositions(self, inPositions):
        self.__OriginalPositions = inPositions

    def WriteFile(self, strFilename: str, arrOriginalPositions: np.array, intFrames: int, bln3d=False):
        self.__OriginalPositions = arrOriginalPositions
        if bln3d:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            self.__bln3d = True
        else:
            fig, ax = plt.subplots()
        self.__ax = ax
        ani = animation.FuncAnimation(
            fig, self.Animate, interval=2000, frames=intFrames)
        writergif = animation.PillowWriter(fps=10)
        ani.save(strFilename, writer=writergif)


# %%
arrCellVectors = dctTJ7['550,015'].GetCellVectors()
objTJAnimate = AnimateTJs(strRoot7_7_49 + '550/u015/', arrCellVectors)
objTJAnimate.SetOriginalPositions(np.array([0.5*arrCellVectors[2], 0.5*(arrCellVectors[0]+arrCellVectors[2]), 0.5*(
    arrCellVectors[1]+arrCellVectors[2]), 0.5*(arrCellVectors[0]+arrCellVectors[1]+arrCellVectors[2])]))
lstReturnPoints, lstMissingSteps = objTJAnimate.FindMeanTripleLinePositions(
    1000, 100)
# %%
lstTime = list(range(0, 100000, 100))
intTJ = 3
arrPoints = np.vstack(lstReturnPoints)
#for j in range(len(lstReturnPoints)):sbatch
#    plt.scatter(lstReturnPoints[j][intTJ][0],lstReturnPoints[j][intTJ][1], c='black')
plt.plot(arrPoints[intTJ:1000:4, 0], arrPoints[intTJ:1000:4, 1])
plt.axis('scaled')
plt.show()
plt.clf()
plt.cla()
plt.close()
# %%
print(np.where(arrPoints[2:800:4, 0] < 1)[0])
objTJAnimate.FindMeanTripleLinePositions(1000, 100)
# %%


def WriteTJAnimations(indctTJ: dict(), indct12BV: dict(), indct13BV: dict(), inRootDir: str, inSaveDir: str):
    for a in indctTJ:
        lstVolumeCutOff = []
        lstVolumeCutOff.append(indctTJ[a].GetLowVolumeCutOff(1, 4*4.05))
        lstVolumeCutOff.append(indct12BV[a].GetLowVolumeCutOff(1, 4*4.05))
        lstVolumeCutOff.append(indct13BV[a].GetLowVolumeCutOff(1, 4*4.05))
        intFrames = np.min(lstVolumeCutOff)
        strTemp = str(indctTJ[a].GetTemp())
        strU = str(indctTJ[a].GetPEString())
        strDirAnim = inRootDir + strTemp + '/u' + strU + '/'
        arrCellVectors = indctTJ[a].GetCellVectors()
        objTJ = AnimateTJs(strDirAnim, arrCellVectors)
        objTJ.WriteFile(inSaveDir + strTemp + 'u' + strU + '.gif', np.array([0.5*arrCellVectors[2], 0.5*(arrCellVectors[0]+arrCellVectors[2]), 0.5*(
            arrCellVectors[1]+arrCellVectors[2]), 0.5*(arrCellVectors[0]+arrCellVectors[1]+arrCellVectors[2])]), intFrames, False)


# %%
strRootDir = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strFileDir = r'/home/p17992pt/MobilityImages/Sigma7_7_49/TJAll'
WriteTJAnimations(dctTJ7, dct12BV7, dct13BV7, strRootDir, strFileDir)
# %%


def WriteGBAnimations(indctTJ: dict(), indct12BV: dict(), indct13BV: dict(), inRootDir: str, inSaveDir: str):
    for a in indctTJ:
        lstVolumeCutOff = []
        lstVolumeCutOff.append(indctTJ[a].GetLowVolumeCutOff(1, 4*4.05))
        lstVolumeCutOff.append(indct12BV[a].GetLowVolumeCutOff(1, 4*4.05))
        lstVolumeCutOff.append(indct13BV[a].GetLowVolumeCutOff(1, 4*4.05))
        intFrames = np.min(lstVolumeCutOff)
        strTemp = str(indctTJ[a].GetTemp())
        strU = str(indctTJ[a].GetPEString())
        strDirAnim = inRootDir + strTemp + '/u' + strU + '/'
        arrCellVectors = indctTJ[a].GetCellVectors()
        objGB = AnimateGBs(strDirAnim, arrCellVectors)
        objGB.WriteFile(inSaveDir + strTemp + 'u' + strU +
                        '.gif', True, True, True, intFrames)


# %%
strRootDir = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strFileDir = r'/home/p17992pt/MobilityImages/Sigma7_7_49/GBAll'
WriteGBAnimations(dctTJ7, dct12BV7, dct13BV7, strRootDir, strFileDir)
# %%

# %%
# Checks mesh points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
strRoot7_7_49 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp'
strType = 'TJ'
objData = LT.LAMMPSData(
    strRoot7_7_49 + '450/u005/TJ/1Min.lst', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
arrCellVectors = objAnalysis.GetCellVectors()
arrPoints12 = np.loadtxt(strRoot7_7_49 + '450/u005/TJ/Mesh12TJ46500.txt')
objTree12 = gf.PeriodicWrapperKDTree(
    arrPoints12, arrCellVectors, gf.FindConstraintsFromBasisVectors(arrCellVectors), 20, ['p', 'p', 'p'])
ax.scatter(*tuple(zip(*objTree12.GetExtendedPoints())))
arrPoints13 = np.loadtxt(strRoot7_7_49 + '450/u005/TJ/Mesh13TJ46500.txt')
objTree13 = gf.PeriodicWrapperKDTree(
    arrPoints13, arrCellVectors, gf.FindConstraintsFromBasisVectors(arrCellVectors), 20, ['p', 'p', 'p'])
ax.scatter(*tuple(zip(*objTree13.GetExtendedPoints())))
arrPoints23 = np.loadtxt(strRoot7_7_49 + '450/u005/TJ/Mesh23TJ46500.txt')
objTree23 = gf.PeriodicWrapperKDTree(
    arrPoints23, arrCellVectors, gf.FindConstraintsFromBasisVectors(arrCellVectors), 20, ['p', 'p', 'p'])
ax.scatter(*tuple(zip(*objTree23.GetExtendedPoints())))
plt.show()
# %%P
# Checking linear trends
dctAny = dct12BV
for a in dctAny.keys():
    if dctAny[a].GetTemp() == 550 and dctAny[a].GetPEParameter() <= 0.02:
        plt.title(str(a))
        arrValues = dctAny[a].GetLogValues()
        arrVolumeSpeed = dctAny[a].GetVolumeSpeed()
        intFinish = dctAny[a].GetLowVolumeCutOff(1, 4*4.05)
        dctAny[a].SetLinearRange(10, intFinish)
        objSlice = dctAny[a].GetLinearRange()
        spl = UnivariateSpline(
            arrVolumeSpeed[0, objSlice], arrVolumeSpeed[1, objSlice])
        # spl.set_smoothing_factor(10)
        plt.plot(arrVolumeSpeed[0, 5:intFinish], spl(
            arrVolumeSpeed[0, 5:intFinish]), 'g', lw=3)
        plt.scatter(arrVolumeSpeed[0, 5:intFinish],
                    arrVolumeSpeed[2, 5:intFinish], c='black')
        plt.show()
        popt2, pop = optimize.curve_fit(
            FitCurve, arrValues[5*5:5*intFinish, 0], arrValues[5*5:5*intFinish, 2])
        plt.scatter(arrValues[5*5:5*intFinish, 0],
                    arrValues[5*5:5*intFinish, 2])
        plt.plot(arrValues[5*5:5*intFinish, 0],
                 FitCurve(arrValues[5*5:5*intFinish, 0], *popt2), c='black')
        plt.show()
# %%
dctAny = dctTJ7
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() >= 0.005 and dctAny[a].GetTemp() == 550:
        arrRows = dctAny[a].GetOverlapRows(1)
        arrLogValues = dctAny[a].GetLogValues()
        arrVolumeSpeed = dctAny[a].GetVolumeSpeed()
        arrTime = arrLogValues[arrRows, 0]
        arrPE = arrLogValues[arrRows, 2]
        arrVolume = arrVolumeSpeed[1, :]
        intFinish = dctAny[a].GetLowVolumeCutOff(1, 4*4.05)
        plt.title(str(dctAny[a].GetPEParameter()) + ' V against t')
        popt1C, pop1C = optimize.curve_fit(
            FitCurve, arrTime[5:intFinish], arrVolume[5:intFinish])
        plt.plot(arrTime[5:intFinish], FitCurve(
            arrTime[5:intFinish], *popt1C), c='black')
        plt.scatter(arrTime[10:intFinish], arrVolume[10:intFinish])
        plt.show()
        plt.title('PE against V')
        popt2C, pop2C = optimize.curve_fit(
            FitLine, arrVolume[5:intFinish], arrPE[5:intFinish])
        plt.plot(arrVolume[5:intFinish], FitLine(
            arrVolume[5:intFinish], *popt2C), c='black')
        plt.scatter(arrVolume[5:intFinish], arrPE[5:intFinish])
        plt.show()
# %%
# comparison of Triple line cell and bicrystal cell
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() <= 0.02:
        plt.scatter(dctGB[a].GetTemp(), dctGB[a].GetMobility(), c='red')
        plt.scatter(dctTJ[a].GetTemp(), dctTJ[a].GetMobility(), c='black')
plt.legend(['12BV', 'TJ'])
plt.show()
# %%
for a in dctTJ.keys():
    if dctTJ[a].GetTemp() == 600:
        plt.scatter(dctTJ[a].GetPEPerVolume(1, 4*4.05), -
                    dctTJ[a].GetNormalSpeed(1, 4*4.05), c='red')
        #
#plt.legend(['12BV', 'TJ'])
plt.show()
# %%
# arrhenius plot of mobility
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.05:
        plt.scatter(1/dct12BV[a].GetTemp(),
                    np.log(dct12BV[a].GetMobility()), c='red')
        plt.scatter(1/dctTJ[a].GetTemp(),
                    np.log(dctTJ[a].GetMobility()), c='black')
plt.legend(['12BV', 'TJ'])
plt.show()
# %%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.05:
        plt.scatter(dctTJR[a].GetTemp(), dctTJR[a].GetMobility(), c='red')
        plt.scatter(dctTJ[a].GetTemp(), dctTJ[a].GetMobility(), c='black')
plt.legend(['TJR', 'TJ'])
plt.show()
# %%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.05:
        plt.scatter(1/dctTJR[a].GetTemp(),
                    np.log(dctTJR[a].GetMobility()), c='red')
        plt.scatter(1/dctTJ[a].GetTemp(),
                    np.log(dctTJ[a].GetMobility()), c='black')
plt.legend(['TJR', 'TJ'])
plt.show()
# %%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetTemp() == 450:
        plt.scatter(dctTJR[a].GetPEPerVolume(), np.log(
            dctTJR[a].GetMobility()), c='red')
        plt.scatter(1/dctTJ[a].GetTemp(),
                    np.log(dctTJ[a].GetMobility()), c='black')
plt.legend(['TJR', 'TJ'])
plt.show()
# %%
dctAny = dctTJ
for a in dctAny.keys():
    if dctAny[a].GetPEParameter() == 0.045:
        plt.scatter(dct12BV[a].GetTemp(), +4*dct12BV[a].GetPEParameter()
                    * 4.05**(-3) - dct12BV[a].GetPEPerVolume(), c='red')
        plt.scatter(dctTJ[a].GetTemp(), +4*dctTJ[a].GetPEParameter()
                    * 4.05**(-3) - dctTJ[a].GetPEPerVolume(), c='black')
plt.legend(['12BV', 'TJ'])
plt.show()
# %%
# %matplotlib qt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
lstPEPerV = []
lstTemp = []
lstMob = []
for a in dctAny:
    if dctAny[a].GetPEParameter() != 0.01:
        lstPEPerV.append(1000*dctAny[a].GetPEPerVolume())
        lstTemp.append(dctAny[a].GetTemp())
        lstMob.append(dctAny[a].GetMobility())
        #ax.scatter(dctAny[a].GetPEPerVolume(),dctAny[a].GetTemp(), dctAny[a].GetMobility())
ax.scatter(lstPEPerV, lstTemp, lstMob)
ax.plot(lstPEPerV, lstTemp, lstMob)
surf = ax.plot_trisurf(lstPEPerV, lstTemp, lstMob)
fig.colorbar(surf)

fig.tight_layout()
plt.show()  # or:
