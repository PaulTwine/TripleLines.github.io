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
from scipy import signal
from scipy import spatial
from scipy import optimize
from scipy.interpolate import UnivariateSpline
from matplotlib import animation
import MiscFunctions as mf
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import os
# from sympy import Matrix
# from sympy.matrices.normalforms import smith_normal_form
#%%
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams['figure.dpi'] = 300
strAxisDisplacement = 'Distance in \AA'
strAxisTime='Time in fs'
strStdDeviationOfRadialDistance='Standard deviation of the radial distance in \AA'
lstColours=['firebrick','darkgoldenrod','navy','purple']
#%%
class TripleLineAnimation(object):
    def __init__(self, strDir: str, arrCellVectors: np.array, arrInitialTJPosistions : np.array,fltWrapperWidth = 25):
        self.__strRoot = strDir
        self.__blnFixedSize = False
        self.__CellVectors = arrCellVectors
        self.__TripleLineTree = KDTree(arrInitialTJPosistions)
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
        intStep = self.__Start + i*self.__Step
        lstPoints = []
        n = 0
        for j in np.sort(self.__TJFiles[intStep]):
            if n in self.__TripleLines:
                try:
                    arrPoints = np.loadtxt(self.__strRoot + str(j) )
                    if len(arrPoints) > 0:
                        arrPoints = gf.AddPeriodicWrapper(arrPoints,self.__CellVectors,20)
                        clustering = DBSCAN(2*4.05).fit(arrPoints)
                        arrLabels = clustering.labels_
                        arrUniqueLabels = np.unique(arrLabels)
                        lstMeans = []
                        lstCluster = []
                        for a in np.sort(arrUniqueLabels):
                            if a !=-1:
                                arrRows = np.where(arrLabels == a)[0]
                                arrMerged = arrPoints[arrRows]
                                #arrMerged = gf.MergeTooCloseAtoms(arrPoints[arrRows],self.__CellVectors,0.5*4.05/np.sqrt(2),5)
                                lstCluster.append(arrMerged)
                                #lstCluster.append(arrPoints[arrRows])
                                lstMeans.append(np.mean(lstCluster[-1],axis=0))
                        arrDistances, arrIndices = self.__TripleLineTree.query(np.vstack(lstMeans))
                        arrMin = np.argmin(arrDistances)
                        lstPoints.append(lstCluster[arrMin])                 
                except:
                    print('Missing file step ', intStep, ' and triple line ' , j)
            n +=1    
            lstColours = ['r','g','b','black']
        self.__dctPoints[intStep] = np.vstack(lstPoints)
        for p in lstPoints:
            self.__ax.scatter(*tuple(zip(*p)))
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
strDirectory1 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma21_21_49/Temp650/u03L/TJ/'
objLAMMPSDat = LT.LAMMPSDat(strDirectory1 + 'TJ.dat')
arrCellVectors = objLAMMPSDat.GetCellVectors()
arrPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
objTJAnimation = TripleLineAnimation(strDirectory1, arrCellVectors,arrPositions)
dct21_21_49Points = dict()
for j in range(4):
    objTJAnimation.TripleLinesToAnimate([j])
    objTJAnimation.WriteFile('/home/p17992pt/TJSigma21_21_49_' +str(j)+ '.gif',1000,0,100)
    dct21_21_49Points[j] = objTJAnimation.GetPointsDictionary()
    objTJAnimation.ClearPointsDictionary()
#%%

#%%
strDirectory2 = '/home/p17992pt/csf4_scratch/CSLTJMobility/Axis111/Sigma7_7_49/Temp600/u015L/TJ/'
objLAMMPSDat = LT.LAMMPSDat(strDirectory2 + 'TJ.dat')
objTJAnimation = TripleLineAnimation(strDirectory2, arrCellVectors,arrPositions)
arrCellVectors = objLAMMPSDat.GetCellVectors()
arrPositions = np.array([0.5*arrCellVectors[2],0.5*(arrCellVectors[2]+arrCellVectors[1]),0.5*(arrCellVectors[2]+arrCellVectors[0]),0.5*(arrCellVectors[2]+arrCellVectors[1]+arrCellVectors[0])])
objTJAnimation = TripleLineAnimation(strDirectory2, arrCellVectors,arrPositions)
dct7_7_49Points = dict()
for j in range(4):
    objTJAnimation.TripleLinesToAnimate([j])
    objTJAnimation.WriteFile('/home/p17992pt/TJSigma7_7_49_' +str(j)+ '.gif',1000,0,100)
    dct7_7_49Points[j] = objTJAnimation.GetPointsDictionary()
    objTJAnimation.ClearPointsDictionary()


#%%
##distance travelled
dctCurrent = dct21_21_49Points
c=0
lstLegend = []
lstCentered = []
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
        lstDisplacement.append(np.linalg.norm(arrMean-arrPositions[k,:2]))
        lstTimeSteps.append(i)
    #pop,popt = optimize.curve_fit(gf.LinearRule,lstTimeSteps[500:],lstDisplacement[500:])
    plt.plot(*tuple(zip(*np.vstack(lstPositions[65:]))),c=lstColours[c])
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
plt.axis('square')
plt.legend(lstLegend)
plt.show()
#%%
## distance time graphs
dctCurrent = dct21_21_49Points
c=0
lstLegend = []
lstCentered = []
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
        lstDisplacement.append(np.linalg.norm(arrMean-arrPositions[k,:2]))
        lstTimeSteps.append(i)
    pop,popt = optimize.curve_fit(gf.LinearRule,lstTimeSteps[500:],lstDisplacement[500:])
    #plt.plot(*tuple(zip(*np.vstack(lstPositions[65:]))),c=lstColours[c])
    plt.plot(lstTimeSteps[100:],lstDisplacement[100:],c=lstColours[c])
    #plt.plot(lstTimeSteps[50:],lstPositions[50:])
    #plt.plot(lstTimeSteps[800:], gf.LinearRule(np.array(lstTimeSteps[800:]),*pop))
    print(k,pop)
    c +=1
plt.xlabel(strAxisTime)
plt.ylabel(strAxisDisplacement)
#plt.ylim([0,150])
#plt.ylim([0,arrCellVectors[1,1]/2])
#plt.xlim([0,arrCellVectors[0,0]/2])
#plt.axis('square')
#plt.legend(lstLegend)
plt.show()

#%%
c=0
lstLegend = []
dctCurrent = dct21_21_49Points
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
        lstDisplacement.append(np.linalg.norm(arrMean-arrPositions[k,:2]))
        # if np.mod(c,2) ==0:
        #     lstDisplacement.append(np.linalg.norm(arrMean))
        # else:
        #     lstDisplacement.append(np.linalg.norm(arrCellVectors[1,:2]/2 - np.linalg.norm(arrMean[1])))
        arrDistances = np.std(arrCentred)
        lstDistances.append(arrDistances)
        lstTimeSteps.append(i)
    #pop,popt = optimize.curve_fit(gf.LinearRule,lstTimeSteps[800:],lstDisplacement[800:])
    #plt.plot(lstTimeSteps[800:], gf.LinearRule(np.array(lstTimeSteps[800:]),*pop))
    #print(k,pop)
    plt.ylim([1.0,4])
    plt.xlabel(strAxisTime)
    plt.ylabel(strStdDeviationOfRadialDistance)
    plt.plot(lstTimeSteps[:], np.mean(lstDistances[:])*np.ones(len(lstDistances[:])),c='black')
    plt.plot(lstTimeSteps, lstDistances,c=lstColours[c])
    plt.show()
    print(np.mean(lstDistances))
    #plt.show()
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
arrNormalised = np.array(lstDistances)-np.mean(lstDistances)
arrNormalised *=signal.windows.hann(len(arrNormalised))
fft = np.fft.rfft(arrNormalised,norm='ortho')
plt.plot(abs(fft))
# %%
print(np.argmax(lstDistances))
for k in dct21_21_49Points:
    dctTJPoints = dct21_21_49Points[k]
    plt.scatter(*tuple(zip(*dctTJPoints[10200])))
    plt.axis('equal')
    plt.show()
# %%
objCSL = gl.CSLTripleLine(np.array([1,1,1]),ld.FCCCell)

print(lstDisplacement[0:])
# %%
