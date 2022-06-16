import sys
import numpy as np
from scipy import stats
import GeometryFunctions as gf

def UpdateTemplate(lstOriginal: list, lstNew: list, strOldFilename: str,strNewFilename: str):
    fIn = open(strOldFilename, 'rt')
    fData = fIn.read()
    for j in range(len(lstOriginal)):
        fData = fData.replace(lstOriginal[j], lstNew[j])
    fIn.close()
    fIn = open(strNewFilename, 'wt')
    fIn.write(fData)
    fIn.close()

def WriteAnnealTemplate(strDirectory: str, strFilename: str, intTemp: int):
    strLogFile =  strFilename + '.log'
    str1MinDumpFile = '1Min*.dmp'
    str2MinDumpFile = '2Min*.dmp'
    strDumpFile = '1Sim*.dmp'
    strDatFile = strFilename + '.dat'
    strFirstMin = '1Min.lst'
    strLastMin = '2Min.lst'
    strLAMMPS =''
    strLAMMPS += 'units metal\n'
    strLAMMPS += 'dimension 3\n'
    strLAMMPS += 'boundary p p p\n'
    strLAMMPS += 'atom_style atomic\n'
    strLAMMPS += 'box tilt large\n'
    strLAMMPS += 'read_data ' + strDatFile +'\n'
    strLAMMPS += 'log ' + strLogFile +  '\n'
    strLAMMPS += 'pair_style eam/alloy\n'
    strLAMMPS += 'pair_coeff * * Al03.eam.alloy Al\n'
    strLAMMPS += 'neighbor 0.3 bin\n'
    strLAMMPS += 'neigh_modify delay 10\n'
    strLAMMPS += 'thermo 100\n'
    strLAMMPS += 'compute pe1 all pe/atom\n'
    strLAMMPS += 'compute v all voronoi/atom\n'
    strLAMMPS += 'compute pt all ptm/atom default 0.15 all\n'
    strLAMMPS += 'compute st all stress/atom NULL virial\n'
    strLAMMPS += 'dump 1 all custom 100 ' + str1MinDumpFile + ' id x y z vx vy vz c_pe1 c_v[1] c_pt[1] c_pt[4] c_pt[5] c_pt[6] c_pt[7] c_st[1] c_st[2] c_st[3] c_st[4] c_st[5] c_st[6]\n'
    strLAMMPS += 'min_style fire\n'
    strLAMMPS += 'timestep 0.002\n'
    strLAMMPS += 'min_modify integrator verlet tmax 6.0 dmax 0.1\n'
    strLAMMPS += 'minimize 0.0 1.0e-6 10000 20000\n'
    strLAMMPS += 'write_dump all custom ' + strFirstMin + ' id x y z vx vy vz c_pe1 c_v[1] c_pt[1] c_pt[4] c_pt[5] c_pt[6] c_pt[7] c_st[1] c_st[2] c_st[3] c_st[4] c_st[5] c_st[6]\n'
    strLAMMPS += 'undump 1 \n'
    strLAMMPS += 'reset_timestep 0\n'
    strLAMMPS += 'timestep 0.001\n'
    strLAMMPS += 'dump 2 all custom 100 ' + strDumpFile + ' id x y z vx vy vz c_pe1 c_v[1] c_pt[1] c_pt[4] c_pt[5] c_pt[6] c_pt[7] c_st[1] c_st[2] c_st[3] c_st[4] c_st[5] c_st[6]\n'
    strLAMMPS += 'velocity all create ' + str(intTemp) + ' 24577\n'
    strLAMMPS += 'fix 2 all nvt temp ' + str(intTemp) + ' ' + str(intTemp) + ' $(100.0*dt)\n'
    strLAMMPS += 'run 100000\n'
    strLAMMPS += 'unfix 2 \n'
    strLAMMPS += 'undump 2 \n'
    strLAMMPS += 'timestep 0.002\n'
    strLAMMPS += 'dump 3 all custom 100 ' + str2MinDumpFile + ' id x y z vx vy vz c_pe1 c_v[1] c_pt[1] c_pt[4] c_pt[5] c_pt[6] c_pt[7] c_st[1] c_st[2] c_st[3] c_st[4] c_st[5] c_st[6]\n'
    strLAMMPS += 'minimize 0.0 1.0e-6 10000 20000\n'
    strLAMMPS += 'write_dump all custom ' + strLastMin + ' id x y z vx vy vz c_pe1 c_v[1] c_pt[1] c_pt[4] c_pt[5] c_pt[6] c_pt[7] c_st[1] c_st[2] c_st[3] c_st[4] c_st[5] c_st[6]\n'
    fIn = open(strDirectory + strFilename + '.in', 'wt')
    fIn.write(strLAMMPS)
    fIn.close()

def ConfidenceInterval(arrValues: np.array, fltAlpha: float, blnInside = True)-> np.array:
    mu,st = stats.norm.fit(arrValues)
    tupValues = stats.norm.interval(alpha=fltAlpha, loc=mu, scale=st)
    if blnInside:
        arrRows = np.where((arrValues > tupValues[0]) & (arrValues < tupValues[1]))[0]
    else:
        arrRows = np.where((arrValues < tupValues[0] ) | (arrValues > tupValues[1]))[0]
    return arrRows
def MatchPairsOfIDs(lstOldIDs: list, lstNewIDs: list):
        arrMatrix = np.zeros([len(lstNewIDs), len(lstOldIDs)])
        for i in range(len(lstNewIDs)):
            for j in range(len(lstOldIDs)):
                arrMatrix[i,j] = len(set(lstOldIDs[j]).intersection(lstNewIDs[i]))
        lstMatched = []
        for j in arrMatrix:
                blnMatched = False
                counter = 0
                while not(blnMatched) and counter < len(j):
                    k = gf.FindNthLargestPosition(j, counter)
                    if k not in lstMatched:
                        lstMatched.append(k[0])
                        blnMatched = True
                    counter += 1
        return lstMatched
def FlattenList(t):
    return [item for sublist in t for item in sublist]



        