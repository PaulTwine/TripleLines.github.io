#%%
import numpy as np
import GeometryFunctions as gf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import MiscFunctions as mf
#%%
intFontSize = 16
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{bm}')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 16})
plt.legend(fontsize=12)
#%%
def FitLine(x, a, b):
    return a*x + b
#%%
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
    plt.ylabel('$m_t$ in $\AA^4$ eV$^{-1}$ fs$^{-1}$')
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
def NormalValuesList(arrValues :np.array, arrStd: np.array, intSize :int):
    lstReturn = []
    for i in range(len(arrValues)):
        arrAdjust = np.random.normal(arrValues[i],arrStd[i],intSize)
        lstReturn.append(arrAdjust)
    arrReturn = np.vstack(lstReturn)
    return np.transpose(arrReturn)
#%%
intStart = 0
intEnd = 3
slcRange = slice(intStart,intEnd+1,1)
arr9 = np.loadtxt('/home/p17992pt/FinalSigma7_7_49u005tou0125.txt')
lstTemp = [450,500,550,600,650,700,750]
PlotMobilities(lstTemp,arr9[0],arr9[1],arr9[2],arr9[3],arr9[4],arr9[5],[-0.3,2.3])
#%%
#
#lstTemp = [450,500,600,650,700,750]
#arr9 = arr9[:, [0,1,3,4,5,6]]
lstColours = ['black','darkolivegreen', 'saddlebrown'] #swapped order to match the array of TJ, 12BC and 13BC
arrITemp = 1/np.array(lstTemp[:])
 
for i in range(3):
    arrRows = np.where(arr9[i,:] > 0)[0]
    arrPlotTemp = arrITemp[arrRows] 
    arrPlotLog = np.log(arr9[i,:][arrRows])
    plt.scatter(arrPlotTemp,arrPlotLog,c=lstColours[i])
    plt.legend(['TJ','B$_{1,2}$', 'B$_{1,3}$'])
    arrStd = np.log((arr9[i+3,:][arrRows])/(arr9[i,:][arrRows])+np.ones(len(arrRows)))
#plt.scatter(arrITemp,np.log(arr9[0,:]),c=lstColours[0]) 
for j in range(3):
    lstPopt0 = []
    lstPopt1 = []
    arrOut = NormalValuesList(arr9[j,:],arr9[j+3,:]/2,10000)
    arrBootStrapRows = mf.BootStrapRows(intEnd-intStart,15000)
    arrLogOut = np.log(np.abs(arrOut))
    for i in range(len(arrOut)):
        #arrBootRows = arrBootStrapRows[i] +intStart
        arrBootRows = np.array(list(range(intEnd-intStart)))
        arrBootRows = (arrBootRows + np.ones(len(arrBootRows))*intStart).astype('int')
        arrLog = np.log(arrOut[i][arrBootRows])
        arrCurrentTemp = arrITemp[arrBootRows]
        popt,pop = curve_fit(FitLine,arrCurrentTemp,arrLog)
        lstPopt0.append(popt[0])
        lstPopt1.append(popt[1])
    fltM = np.mean(lstPopt0)
    fltC = np.mean(lstPopt1)
    #plt.scatter(arrITemp[slcRange],np.mean(arrLogOut,axis=0),c=lstColours[j])
    if j in [0]:
        plt.plot(arrITemp,FitLine(arrITemp,fltM,fltC),c=lstColours[j])
    plt.errorbar(arrITemp[arrRows],np.log(arr9[j,:][arrRows]),arrStd,c=lstColours[j],ls='none',capsize=5)
    print(np.mean(lstPopt0),np.std(lstPopt0))
plt.axvline(arrITemp[intStart],c='black',ls='dashed')
plt.axvline(arrITemp[intEnd],c='black',ls='dashed') 
plt.xlabel('Inverse temperature K$^{-1}$')
plt.ylabel('$\ln(m_t)$')
plt.ylim([-1,1]) 
plt.tight_layout()
plt.show()

# %%
