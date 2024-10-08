import numpy as np
import matplotlib.pyplot as plt
import LatticeDefinitions as ld
import GeometryFunctions as gf
import GeneralLattice as gl
import LAMMPSTool as LT
import copy as cp
import sys
import os

strDirIn = sys.argv[1]
strTemplate = sys.argv[2]

for j in os.listdir(strDirIn):
    if j.endswith('.dmp'):
        fIn = open(strDirIn + strTemplate, 'rt')
        fData = fIn.read()
        fData = fData.replace('read.dat', j[:-3]+ 'dat')
        fData = fData.replace('read.dmp', j[:-3] + 'dmp')
        fData = fData.replace('read*.dmp', j[:-5] + 'New*.dmp')
        fData = fData.replace('read.lst',  j[:-3] + 'lst')
        fData = fData.replace('read.log',  j[:-3] + 'log')
        fIn.close()
        fIn = open(strDirIn + j[:-3]+'in', 'w+')
        fIn.write(fData)
        fIn.close()



