import numpy as np
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSDump as LD
import sys
a1=4.05
a2= np.sqrt(3)*a1
strPMFile = 'dump.eam' + str(sys.argv[1]) + 'PM'
objData = LD.LAMMPSData(strPMFile,1)
objProcess = objData.GetTimeStepByIndex(-1)
objProcess.CategoriseAtoms()
h = objProcess.CellHeight
objProcess.FindTripleLines(a2,3*a2, 3)
objProcess.MergePeriodicTripleLines(a2)
with open('SummaryData'+ str(sys.argv[1]), 'w+') as fdata:
    for j in objProcess.GetUniqueTripleLineIDs():
        fltEnergy, fltRadius, intNumberOfAtoms = objProcess.FindTripleLineEnergy(j,a1/4,a1)
        lstGBIDs = []
        for h in objProcess.GetUniqueTripleLines(j).GetUniqueAdjacentGrainBoundaries():
            lstGBIDs.extend(objProcess.FindGBAtoms(h,2*objProcess.GetUniqueTripleLines(j).GetRadius(),3*a1))
        fltGBEnergy = np.mean(objProcess.GetAtomsByID(lstGBIDs)[:,7]))
        fdata.write('{},{},{},{},{},{}\n'.format(j, objProcess.GetUniqueTripleLines(j).GetCentre(),fltEnergy,fltRadius,intNumberOfAtoms,fltGBEnergy))                                       
    fdata.close()
