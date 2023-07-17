import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import sys
from mpl_toolkits.mplot3d import Axes3D

lstAxis = [0,0,1] #eval(str(sys.argv[4]))
arrAxis = np.array(lstAxis)

objSigma = gl.SigmaCell(arrAxis,ld.FCCCell)
arrValues = objSigma.GetSigmaValues(5)[:5,0].astype('int')
lstBurgers = []
lstSigma = []
for k in arrValues:
    objSigma.MakeCSLCell(k)
    fltAngle1, fltAngle2 = objSigma.GetLatticeRotations()
    arrBasisVectors = gf.StandardBasisVectors(3)
    lstBases = []
    lstBases.append(np.linalg.inv(gf.RotateVectors(fltAngle1,arrAxis,arrBasisVectors)))
    lstBases.append(np.linalg.inv(gf.RotateVectors((fltAngle1+fltAngle2)/2,arrAxis,arrBasisVectors)))
    lstBases.append(np.linalg.inv(gf.RotateVectors(fltAngle2,arrAxis,arrBasisVectors)))
    arrTensor = gf.TripleLineTensor(arrAxis,lstBases,[-np.pi/6,np.pi/2,7*np.pi/6])
    arrBurgers = np.matmul(arrTensor, np.array([1,0,0]))
    lstBurgers.append(np.dot(arrBurgers,arrBurgers))
    #lstBurgers.append(np.linalg.norm(arrBurgers))
    lstSigma.append(k)
    print(np.linalg.eig(arrTensor))

plt.scatter(lstSigma,lstBurgers)
plt.show()
