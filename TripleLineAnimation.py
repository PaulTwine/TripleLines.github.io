#%%
import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D
import copy as cp
import pickle
# from scipy import signal
# from scipy import spatial
from scipy import optimize
# from scipy import interpolate
from matplotlib import animation
import MiscFunctions as mf
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import os
# from scipy.signal import savgol_filter
# from scipy.signal import hilbert
from scipy.signal import butter
from scipy.signal import filtfilt
import pickle
# from sympy import Matrix
# from sympy.matrices.normalforms import smith_normal_form
#%%
def FitLine(x,a,b):
    return x*a + b
#%%
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams.update({'font.size': 15}) #normally 15
plt.rcParams['figure.dpi'] = 300
strAxisDisplacement = 'Distance in \AA'
strAxisTime='Time in fs'
strStdDeviationOfRadialDistance='Standard deviation of the mesh point radial distances in \AA'
lstColours=['firebrick','darkgoldenrod','navy','purple']
strKappa = '$\kappa$ in \AA'
strKappaResiduals = 'Residuals of $\kappa$ in \AA'
strResidual = 'Residuals of the standard deviation of \n the mesh point radial distances in \AA'
strProjected = 'Projected distance parallel to $\mathbf{j}$ in multiples of $|\mathbf{c}_2|$'
#%%
class TripleLineAnimation(object):
    def __init__(self, strDir: str, arrCellVectors: np.array, arrInitialTJPosistions : np.array,fltWrapperWidth = 25):
        self.__strRoot = strDir
        self.__blnFixedSize = False
        self.__CellVectors = arrCellVectors
        self.__BasisConversion = np.linalg.inv(arrCellVectors)
        self.__TripleLineTree = KDTree
        (arrInitialTJPosistions)
        self.__InitalTJPositions = arrInitialTJPosistions
        self.__TJTreeInitialPositions = gf.PeriodicWrapperKDTree(arrInitialTJPosistions, self.__CellVectors,gf.FindConstraintsFromBasisVectors(self.__CellVectors),fltWrapperWidth)
        lstFiles = os.listdir(strDir)
        lstTJFiles =[]
        dctFiles = dict()
        for i in lstFiles:
            if i.startswith('TJ') and i.endswith('.txt'):
                intTimeStep = int(i[7:-4])
                if intTimeStep in dctFiles.keys():
                    lstCurrentFiles = dctFiles[intTimeStep]
                    lstCurrentFiles.append(i)
                else:
                    dctFiles[intTimeStep] = [i]
        self.__TJFiles = dctFiles
        self.__dctPoints = dict()
    def GetPointsDictionary(self):
        return self.__dctPoints
    def TripleLinesToAnimate(self, lstRange):
        self.__TripleLines = lstRange   
    def Animate(self,i):
        self.__ax.clear()
        if self.__blnFixedSize:
            self.__ax.set_xlim([0,self.__CellVectors[0,0]])
            self.__ax.set_ylim([0,self.__CellVectors[1,1]])
            self.__ax.set_zlim([0,self.__CellVectors[2,2]])
        self.__ax.axis('off')
        intStep = self.__Start + i*self.__Step
        lstPoints = []
        n = 0
        dctTJPoints = dict()
        for j in np.sort(self.__TJFiles[intStep]):
            try:
                arrPoints = np.loadtxt(self.__strRoot + str(j) )
                if len(arrPoints) > 0:
                    arrDistances, arrIndices = self.__TJTreeInitialPositions.Pquery(arrPoints)
                    arrIndices = mf.FlattenList(arrIndices)
                    arrPeriodicIndices = self.__TJTreeInitialPositions.GetPeriodicIndices(arrIndices)
                    arrIndices, arrCounts = np.unique(arrPeriodicIndices,return_counts=True)
                    arrTJNumber = arrPeriodicIndices[np.argmax(arrCounts)]
                    if arrTJNumber in self.__TripleLines:
                        arrMovedPoints = gf.PeriodicShiftAllCloser(self.__InitalTJPositions[arrTJNumber], arrPoints, self.__CellVectors, self.__BasisConversion,['pp','pp','ff'])
                        lstPoints.append(arrMovedPoints)
            except:
                    print('Missing file step ', intStep, ' and triple line ' , j)
            n +=1    
            lstColours = ['r','g','b','black']
        if len(lstPoints) > 1:
            arrPoints = np.unique(np.vstack(lstPoints),axis=0)
        elif len(lstPoints) == 1:
            arrPoints = lstPoints[0]
        else:
            print('Error missing points',j)
        arrMean = np.mean(arrPoints,axis=0)
        self.__dctPoints[intStep] = np.unique(arrPoints,axis=0)
        #self.__ax.scatter(x_fine,y_fine,z_fine,c='black')
        self.__ax.scatter(*tuple(zip(*arrPoints)),s=4,c='grey')
        self.__ax.plot([arrMean[0],arrMean[0]],[arrMean[1],arrMean[1]],[0,self.__CellVectors[2,2]],c='black')
        # for p in lstPoints:
        #     self.__ax.scatter(*tuple(zip(*p)),s=4)
        gf.EqualAxis3D(self.__ax)
        # for p in list(dctPoints.keys()):
        #     arrOutPoints = np.vstack(dctPoints[p])
        #     self.__ax.scatter(*tuple(zip(*arrOutPoints)),c=lstColours[p])
    def WriteFile(self,strFilename: str,intFrames: int, intStart: int, intStep: int):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        self.__ax = ax
        self.__Start = intStart
        self.__Step = intStep
        ani = animation.FuncAnimation(fig, self.Animate,interval=2000, frames=intFrames) 
        writergif = animation.PillowWriter(fps=20)
        ani.save(strFilename,writer=writergif)
    def ClearPointsDictionary(self):
        self.__dctPoints = dict()
#%%
strDirectory1 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp600/u03/TJ/'
objLAMMPSDat = LT.LAMMPSDat(strDirectory1 + 'TJ.dat')
arrCellVectors = objLAMMPSDat.GetCellVectors()
arrTJPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
objTJAnimation = TripleLineAnimation(strDirectory1, arrCellVectors,arrTJPositions)
dct21_21_49Points = dict()
with open('/home/p17992pt/dct212149L.dct', 'rb') as f:
    dct21_21_49Points = pickle.load(f)
# for j in range(4):
#     objTJAnimation.TripleLinesToAnimate([j])
#     objTJAnimation.WriteFile('/home/p17992pt/STJSigma21_21_49_' +str(j)+ '.gif',1000,0,100)
#     dct21_21_49Points[j] = objTJAnimation.GetPointsDictionary()
#     objTJAnimation.ClearPointsDictionary()
#%%

#%%
strDirectory2 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp600/u015/TJ/'
objLAMMPSDat = LT.LAMMPSDat(strDirectory2 + 'TJ.dat')
arrCellVectors = objLAMMPSDat.GetCellVectors()
arrTJPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
objTJAnimation = TripleLineAnimation(strDirectory2, arrCellVectors,arrTJPositions)
dct7_7_49Points = dict()
with open('/home/p17992pt/dct7749L.dct', 'rb') as f:
    dct7_7_49Points = pickle.load(f)
# for j in range(4):
#     objTJAnimation.TripleLinesToAnimate([j])
#     objTJAnimation.WriteFile('/home/p17992pt/MTJSigma7_7_49_' +str(j)+ '.gif',1000,0,100)
#     dct7_7_49Points[j] = objTJAnimation.GetPointsDictionary()
#     objTJAnimation.ClearPointsDictionary()
#%%
strDirectory3 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp600/u03L/TJ/'
objLAMMPSDat = LT.LAMMPSDat(strDirectory3 + 'TJ.dat')
arrCellVectors = objLAMMPSDat.GetCellVectors()
arrTJPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
objTJAnimation = TripleLineAnimation(strDirectory3, arrCellVectors,arrTJPositions)
dct9_9_9Points = dict()
#with open('/home/p17992pt/dct999L.dct', 'rb') as f:
#    dct9_9_9Points = pickle.load(f)
for j in range(4):
    objTJAnimation.TripleLinesToAnimate([j])
    objTJAnimation.WriteFile('/home/p17992pt/MTJSigma9_9_9_' +str(j)+ 'L.gif',540,0,100)
    dct9_9_9Points[j] = objTJAnimation.GetPointsDictionary()
    objTJAnimation.ClearPointsDictionary()
#%%
dctCurrent = dct7_7_49Points
#%%
##distance travelled
c=0
lstLegend = []
lstTimeSteps = []
for k in dctCurrent:
    lstLegend.append(k+1)
    lstDistances = []
    lstTimeSteps = []
    lstDisplacement = []
    lstPositions = []
    dctPts = dctCurrent[k]
    for i in dctPts.keys():
        arrMean = np.mean(dctPts[i],axis=0)[:2]
        arrCentred = dctPts[i][:,:2] - arrMean
        lstPositions.append(arrMean)
        lstDisplacement.append(np.linalg.norm(arrMean-arrTJPositions[k,:2]))
        lstTimeSteps.append(i)
        #pop,popt = optimize.curve_fit(gf.LinearRule,lstTimeSteps[500:],lstDisplacement[500:])
    plt.plot(*tuple(zip(*np.vstack(lstPositions[:]))),c=lstColours[c])
    #plt.plot(lstTimeSteps[100:],lstDisplacement[100:],c=lstColours[c])
    #plt.plot(lstTimeSteps[50:],lstPositions[50:])
    #plt.plot(lstTimeSteps[800:], gf.LinearRule(np.array(lstTimeSteps[800:]),*pop))
    #print(k,pop)
    c +=1
#plt.xlabel(strAxisTime)
#plt.ylabel(strAxisDisplacement)
#plt.ylim([0,150])
plt.ylim([0,arrCellVectors[1,1]/2])
plt.xlim([0,arrCellVectors[0,0]/2])
#plt.xticks(np.linspace(0,arrCellVectors[1,1]/2,y))
plt.axis('square')
plt.legend(lstLegend)
plt.show()
#%%
## distance time graphs
intTimeStop = 1000
intTimeStart = 0
arrTJPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
dctCurrentDistances = dict()
dctCurrentSpeeds = dict()
c=0
lstLegend = []
lstCentered = []
for k in dctCurrent:
    lstLegend.append(k+1)
    lstDistances = []
    lstTimeSteps = []
    lstPositions = []
    dctPts = dctCurrent[k]
    for i in dctPts.keys():
        #arrMean = np.mean(dctPts[i],axis=0)[:2]
        #arrCentred = dctPts[i][:,:2] - arrMean
        arrMean = np.mean(dctPts[i],axis=0)[:2]
        arrCentred = dctPts[i][:,1] - arrMean[1]
        lstPositions.append(arrMean)
        #lstDistances.append(np.linalg.norm(arrMean-arrTJPositions[k,:2]))
        lstDistances.append(np.linalg.norm(arrMean[1]-arrTJPositions[k,1]))
        lstTimeSteps.append(i)
    pop,popt = optimize.curve_fit(gf.LinearRule,lstTimeSteps[intTimeStart:intTimeStop],lstDistances[intTimeStart:intTimeStop])
    lstValues = mf.BootstrapEstimate(lstTimeSteps[intTimeStart:intTimeStop],lstDistances[intTimeStart:intTimeStop],intTimeStop-intTimeStart)
    print(np.mean(lstValues),np.std(lstValues))
    #plt.plot(*tuple(zip(*np.vstacklstPositions[65:]))),c=lstColours[c])
    plt.plot(lstTimeSteps[:],lstDistances[:],c=lstColours[c])
    #plt.plot(lstTimeSteps[50:],lstPositions[50:])
    #plt.plot(lstTimeSteps[800:], gf.LinearRule(np.array(lstTimeSteps[800:]),*pop))
    plt.ylim([0,145])
    dctCurrentSpeeds[k] = pop
    print(k,pop)
    dctCurrentDistances[c] = lstDistances
    c +=1
#plt.plot(arrVolume21_21_49[0,0:],arrVolume21_21_49[2,1000::-1]/2-np.min(arrVolume21_21_49[2,1000::-1]/2))
plt.xlabel(strAxisTime)
plt.ylabel(strAxisDisplacement)
plt.ylim([0,150])
#plt.ylim([0,10000])
plt.xlim([0,100000])
#plt.axis('square')
plt.legend(lstLegend)
plt.show()
#%%
arrVolume21_21_49 = np.loadtxt('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp600/u03L/TJ/VolumeTJ.txt')
objLog21_21_49 = LT.LAMMPSLog('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp600/u03L/TJ/TJ.log')
objDat= LT.LAMMPSDat('/home/p17992pt/csf4_scratch/CSLTJMobility/Axis511/Sigma9_9_9/Temp600/u03L/TJ/TJ.dat')
plt.plot(arrVolume21_21_49[0,:],arrVolume21_21_49[2,::-1]/2)
arrCellVectors  = objDat.GetCellVectors()
fltArea = np.linalg.norm(np.cross(arrCellVectors[0],arrCellVectors[2]))
lstValues = mf.BootstrapEstimate(arrVolume21_21_49[0,400:550],arrVolume21_21_49[2,400:550:]/2,1000)
print(np.mean(lstValues),np.std(lstValues))
print(arrVolume21_21_49[1,:]/arrVolume21_21_49[2,:])
plt.show()
plt.plot(arrVolume21_21_49[2,::-1],objLog21_21_49.GetValues(1)[:,2])
lstValues = mf.BootstrapEstimate(arrVolume21_21_49[2,400:550:],objLog21_21_49.GetValues(1)[400:550,2],150)
print(np.mean(lstValues)/fltArea,np.std(lstValues)/fltArea)
print(arrVolume21_21_49[1,:]/arrVolume21_21_49[2,:])

#%%
#Radial displacement
c=0
lstLegend = []
dctCurrentRadial = dict()
dctCurrentRadialSmooth = dict()
fltPeriod = np.round((4.05*np.sqrt(2))/(100*np.abs(dctCurrentSpeeds[0][0])))
for k in dctCurrent:
    lstLegend.append(k+1)
    lstRadialDistances = []
    lstRadialDistancesSmooth = []
    lstTimeSteps = []
    lstDisplacement = []
    lstPositions = []
    dctPts = dctCurrent[k]
    for i in dctPts.keys():
        arrMPoints = dctPts[i]
        #arrMPoints = gf.MergeTooCloseAtoms(dctPts[i],arrCellVectors, 4.05/np.sqrt(2)*0.5)
        arrMean = np.mean(arrMPoints,axis=0)[:2]
        arrCentred = arrMPoints[:,:2] - arrMean
        lstPositions.append(arrMean)
        lstDisplacement.append(np.linalg.norm(arrMean[:2]-arrTJPositions[k,:2]))
        # if np.mod(c,2) ==0:
        #     lstDisplacement.append(np.linalg.norm(arrMean))
        # else:
        #     lstDisplacement.append(np.linalg.norm(arrCellVectors[1,:2]/2 - np.linalg.norm(arrMean[1])))       
        arrRadialDistances = np.std(arrCentred)
        #arrRadialDistances = np.median(np.abs(arrCentred))
        lstRadialDistances.append(arrRadialDistances)
        lstTimeSteps.append(i)
    #pop,popt = optimize.curve_fit(gf.LinearRulen,lstTimeSteps[800:],lstDisplacement[800:])
    #plt.plot(lstTimeSteps[800:], gf.LinearRule(np.array(lstTimeSteps[800:]),*pop))
    #print(k,pop)
    #plt.ylim([0.5,3.2]) #for long triple lines plt.ylim([0.5,3.2])
    arrAllRadialDistances = np.array(lstRadialDistances)
   #arrAllRadialDistances = arrAllRadialDistances/(np.mean(arrAllRadialDistances))
    a,b = butter(4,2/fltPeriod,'lowpass',False)
    arrSmooth = filtfilt(a,b,arrAllRadialDistances)
    lstRadialDistancesSmooth.append(arrSmooth) 
    plt.xlabel(strAxisTime)
    #plt.ylabel(strStdDeviationOfRadialDistance)
    plt.ylabel(strKappa)
    popt, pop = optimize.curve_fit(FitLine,np.array(lstTimeSteps),arrAllRadialDistances)
    print(popt)
    lstValues = mf.BootstrapEstimate(np.array(lstTimeSteps),arrAllRadialDistances,1000)
    print(np.mean(lstValues),np.std(lstValues))
    arrProjection = FitLine(np.array(lstTimeSteps), *popt)
    plt.plot(lstTimeSteps,arrProjection,c='black',linestyle='dashed')
    #plt.plot(lstTimeSteps[:], np.mean(lstRadialDistances[:])*np.ones(len(lstRadialDistances[:])),c='black')
    plt.plot(lstTimeSteps, arrAllRadialDistances,c=lstColours[c],alpha=0.7)
    plt.plot(lstTimeSteps, arrSmooth,c='black')
    #plt.plot(lstTimeSteps, savgol_filter(lstRadialDistances,21,5),c='grey')
    print(np.mean(lstRadialDistances),np.std(lstRadialDistances))
    plt.xlim([0,100000])
    plt.ylim([0.5,3])
    intLimit = np.max(np.ceil(dctCurrentDistances[c]/y)).astype('int')
    for k in range(1,intLimit):
        intPoint = np.argmin(np.abs(dctCurrentDistances[c]/y-k*np.ones(len(dctCurrentDistances[c]))))
        plt.axvline(lstTimeSteps[intPoint],c='black',linestyle='dotted')
    plt.show()
    #plt.show()
    dctCurrentRadial[c]=arrAllRadialDistances
    dctCurrentRadialSmooth[c] = arrSmooth-arrProjection
    c +=1
    
    #print(k, np.mean(lstDistances), np.std(lstDistances))
    #plt.plot(lstTimeSteps[50:],lstDisplacement[50:])
    #plt.axis('equal')
    #plt.xlim([0,arrCellVectors[0,0]/2])
    #plt.ylim([0,arrCellVectors[1,1]/2])
    #plt.plot(np.vstack(lstPositions)[50:,0],np.vstack(lstPositions)[50:,1])
    
#plt.legend(lstLegend)
#plt.show()
#%%
n = 0
plt.plot(dctCurrentDistances[n]/y,dctCurrentRadialSmooth[n],c=lstColours[n])
plt.ylabel(strKappaResiduals)
plt.xlabel(strProjected)
#plt.xticks(list(range(intEnd+2)))
for j in range(8):
    plt.axvline(j,c='black',linestyle='dashed')
plt.xticks(list(range(8)))
plt.xlim([0,7])
plt.ylim([-np.max(dctCurrentRadialSmooth[n]),np.max(dctCurrentRadialSmooth[n])])
plt.show()

#%%
objCSL = gl.CSLTripleLine(np.array([1,1,1]), ld.FCCCell)
arrCell = objCSL.FindTripleLineSigmaValues(200)
intIndex = np.where(np.all(arrCell[:,:,0].astype('int')==[7,7,49],axis=1))[0][0]
arrCSL = arrCell[intIndex]
objCSL.GetTJSigmaValue(arrCSL)
objCSL.GetTJBasisVectors(intIndex,True)
arrCellBasis = objCSL.GetCSLBasisVectors()
arrEdgeVectors, arrTransform = gf.ConvertToLAMMPSBasis(arrCellBasis)
y= 4.05*arrEdgeVectors[1,1]
print(np.round(4.05*arrEdgeVectors,5))
#%%
c = 3
arrRadial = dctCurrentRadialSmooth[c]
y= 4.05*arrEdgeVectors[1,1]
arrPositions = np.array(dctCurrentDistances[c])/y
arrRows = np.argsort(arrPositions)
arrPositions = arrPositions[arrRows]
intEnd = np.max(np.floor(arrPositions)).astype('int')
arrStart = 0#max(np.where(arrPositions[:] <=1)[0])
arrEnd = min(np.where(arrPositions[:] >= intEnd))[0]
arrEnd = len(arrPositions)-1
fltMean = np.mean(arrRadial)
arrUsedPositions = arrPositions[arrStart:arrEnd]
arrUsedRadial = arrRadial[arrStart:arrEnd]
#arrUsedRadial = np.abs(hilbert(arrUsedRadial))
#arrRadial -= fltMean
#arrRadial = np.abs(hilbert(np.array(dctCurrentRadial[c])[arrRows]))
# arrPhase = np.unwrap(np.angle(arrRadial))
# arrFrequency = np.diff(arrPhase)/(2*np.pi*(intEnd-1))
# plt.plot(arrRadial,arrFrequency)
# plt.show()
#plt.hist(arrRadial)
#plt.show()
#arrRadial = np.array(dctCurrentRadial[c])[arrRows]

#arrRadial -= fltMean
#tck, u= interpolate.splprep([arrPositions[arrStart:arrEnd],arrRadial[arrStart:arrEnd]],s=2)
#fit = interpolate.splev(np.linspace(0,1,9), tck)
popt, pop = optimize.curve_fit(FitLine,arrUsedPositions,arrUsedRadial)
print(popt)
for j in range(8):
    plt.axvline(j,c='black',linestyle='dotted')
plt.xticks(list(range(8)))
plt.xlim([0,7])
arrProjection = FitLine(arrUsedPositions, *popt)
#plt.plot(arrUsedPositions,arrProjection,c='black')
plt.plot(arrUsedPositions,arrUsedRadial-arrProjection,c=lstColours[c])
plt.axhline(0,0,linestyle='dashed',c='black')
fltMax = np.max(arrUsedRadial-arrProjection)
#plt.plot(arrUsedPositions, savgol_filter(arrUsedRadial,intEnd*2+1,3,mode='interp'),c='grey')
#plt.axhline(fltMean,c='black')
#plt.plot(fit[0],fit[1])
print(fltMax)
#fltMax = 0.85
plt.xticks(list(range(8)))
plt.ylim([-0.65,0.85])
plt.ylabel(strKappaResiduals)
plt.xlabel(strProjected)
plt.show()
#%%
objData = LT.LAMMPSData(strDirectory1 + '1Sim90000.dmp', 1, 4.05, LT.LAMMPSAnalysis3D)
objAnalysis = objData.GetTimeStepByIndex(-1)
#%%
#%%
arrScope = arrUsedRadial-arrProjection
sp = np.fft.rfft(arrScope)
pwr = np.abs(sp)**2
acorr = np.fft.ifft(pwr).real
#print(acorr)

freq = np.fft.rfftfreq(len(arrScope))
#plt.scatter(freq, np.sqrt(sp.real**2 + sp.imag**2))
plt.scatter(freq,np.abs(sp))
plt.xlim([0,16/1000])
#%%
def FunctionFourier(arrX: np.array, arrFourier: np.array):
    arrReturn = np.zeros(len(arrX))
    intL = len(arrFourier)
    for j in range(intL):
        arrReturn += arrFourier[j]*np.exp(2*np.pi*arrX*j*np.imag(1))/(intL)
    return arrReturn
arrFunction = FunctionFourier(np.linspace(1,3,100),sp)
plt.plot(np.linspace(0,3,100),arrFunction)
plt.show()
#%%
# objCSL = gl.CSLTripleLine(np.array([1, 1, 1]), ld.FCCCell)
# arrCell = objCSL.FindTripleLineSigmaValues(200)
# intIndex = np.where(np.all(arrCell[:, :, 0].astype(
#     'int') == [7, 7, 49], axis=1))[0][0]
# arrCSL = arrCell[intIndex]
# objCSL.GetTJSigmaValue(arrCSL)
# objCSL.GetTJBasisVectors(intIndex)
# arrBasis = objCSL.GetSimulationCellBasis()
# y = 4.05*arrBasis[1,1]
# arrNormalised = np.array(lstDistances)-np.mean(lstDistances)
# arrNormalised *=signal.windows.hann(len(arrNormalised))
# fft = np.fft.rfft(arrNormalised,norm='ortho')
# plt.plot(abs(fft))
# %%
import statsmodels.api as sm
intTJ = 0
arrDistance0 = np.array(dctCurrentDistances[intTJ])
intStart =1
intEnd = 2
intPeriods = 7
arrRows1 = np.where((arrDistance0 < intStart*y +1) & (arrDistance0 > intStart*y-1))[0]
arrRows2 = np.where((arrDistance0 < intEnd*y +1) & (arrDistance0 > intEnd*y-1))[0]
end = np.median(arrRows2).astype('int')
start = np.median(arrRows1).astype('int')
arrRadial0 = np.array(dctCurrentRadial[intTJ])
arrRadial0 = arrRadial0[start:intPeriods*end]
arrDistance0 = arrDistance0[start:intPeriods*end]
#arrNormalised = arrNormalised/np.max(arrNormalised)
#arrNormalised *=signal.windows.hann(len(arrNormalised))

print(np.argmax(fft),arrNormalised[np.argmax(fft)])
objSpline = interpolate.BSpline(np.sort(arrDistance0[::25]), arrRadial0[np.argsort(arrDistance0[::25])],3,extrapolate='periodic')


lstVlines = []
intLimit = np.round(np.max(arrDistance0)/y).astype('int')+1
for j in range(1,intPeriods+1):
    lstVlines.append(j*y)
    #print(arrNormalised[start + j*intPeriods])
plt.vlines(lstVlines,min(arrRadial0),max(arrRadial0),colors='black')
plt.plot(arrDistance0,arrRadial0)
#plt.plot(arrDistance0,objSpline(arrDistance0))
lowess = sm.nonparametric.lowess(arrRadial0,arrDistance0, frac=0.1)
plt.plot(lowess[:,0],lowess[:,1])
plt.show()
arrNormalised = arrRadial0-np.mean(arrRadial0)
fft = np.fft.fft(arrNormalised,norm='ortho')
freq = np.fft.fftfreq(len(arrNormalised),np.max(arrNormalised)-np.min(arrNormalised))
plt.scatter(freq[:20],np.abs(fft[:20]))
intMax = np.argmax(np.abs(fft)) 
print(2*np.pi/freq[intMax])
plt.show()

# %%
with open('/home/p17992pt/dct212149.dct', 'wb') as f:
    pickle.dump(dct21_21_49Points, f)
# %%
with open('/home/p17992pt/dct7749L.dct', 'rb') as f:
    loaded_dict = pickle.load(f)
# %%
