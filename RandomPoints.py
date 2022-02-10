import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import copy as cp
import sys
fig = plt.figure(figsize=plt.figaspect(1)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
ax = fig.gca(projection='3d')
a  = 4.05
strDirectory = str(sys.argv[1])
intSigma = int(sys.argv[2])
lstAxis = eval(str(sys.argv[3]))
intIncrements = 10# int(sys.argv[4])
arrAxis = np.array(lstAxis)
objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
objSigma.MakeCSLCell(intSigma)
arrBasis = a*objSigma.GetBasisVectors()
fltArea = np.linalg.norm(np.cross(arrBasis[1],arrBasis[2]))
fltDistance = np.sqrt(fltArea/(2*intIncrements))
blnStop = False
arrRandom = np.array([(0.5-np.random.ranf())*arrBasis[1]+(0.5-np.random.ranf())*arrBasis[2]])
arrConstraints = gf.FindConstraintsFromBasisVectors(arrBasis)
objPeriodicTree = gf.PeriodicWrapperKDTree(arrRandom, arrBasis,arrConstraints,fltDistance)
i = 0
intPoints = 0
while (blnStop == False and i < 100000):
    lstPoints = []
    if intPoints < 3:
        arrStart = 0.5*(arrBasis[1]+arrBasis[2])
    elif 3 <= intPoints < 5:
        arrStart = 0.5*(arrBasis[1]-arrBasis[2])
    elif 3 <= intPoints < 5:
        arrStart = 0.5*(-arrBasis[1]+arrBasis[2])
    elif 3 <= intPoints < 5:
        arrStart = -0.5*(arrBasis[1]+arrBasis[2])
    else:
        arrStart = np.array([(0.5-np.random.ranf())*arrBasis[1]+(0.5-np.random.ranf())*arrBasis[2]])
    arrRandom = 0.5*np.array([np.random.beta(2,2)*arrBasis[1]+np.random.beta(2,2)*arrBasis[2]]) +arrStart    
    arrIndices = objPeriodicTree.Pquery_radius(arrRandom, fltDistance)[0]
    if len(np.unique(arrIndices)) == 1 and len(arrIndices[0]) == 0:
        lstPoints.append(objPeriodicTree.GetOriginalPoints())
        lstPoints.append(arrRandom)
        arrPoints = np.vstack(lstPoints)
        intPoints +=1
        if len(arrPoints) == intIncrements +1:
            blnStop = True
        else:
            objPeriodicTree = gf.PeriodicWrapperKDTree(arrPoints, arrBasis,arrConstraints,fltDistance)
    i +=1
#ax.scatter(*tuple(zip(*objPeriodicTree.GetOriginalPoints())))
#gf.EqualAxis3D(ax)
#plt.show()
np.savetxt(strDirectory + 'AllRandomPoints',arrPoints,fmt='%f')

for  k in range(intIncrements):
    np.savetxt(strDirectory + str(k) + '/RandomDisplacement.txt', arrPoints[k], fmt='%f')